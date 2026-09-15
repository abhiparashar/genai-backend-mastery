package com.genai.gateway.controller;

import com.genai.gateway.dto.Dtos.ChatRequest;
import com.genai.gateway.dto.Dtos.ChatResponse;
import com.genai.gateway.dto.Dtos.ErrorResponse;
import com.genai.gateway.service.AiClient;
import com.genai.gateway.service.ConversationService;
import jakarta.validation.Valid;
import java.time.Duration;
import java.util.UUID;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.support.WebExchangeBindException;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

/**
 * Public chat API. Java owns auth, validation, conversation state and cost
 * attribution; Python owns the model call.
 */
@RestController
@RequestMapping("/api/chat")
public class ChatController {

    private static final Logger log = LoggerFactory.getLogger(ChatController.class);
    private static final String CORRELATION_HEADER = "X-Correlation-ID";

    private final AiClient aiClient;
    private final ConversationService conversations;

    public ChatController(AiClient aiClient, ConversationService conversations) {
        this.aiClient = aiClient;
        this.conversations = conversations;
    }

    /** Synchronous completion. */
    @PostMapping
    public Mono<ResponseEntity<ChatResponse>> chat(
            @Valid @RequestBody ChatRequest request,
            @RequestHeader(value = CORRELATION_HEADER, required = false) String incomingId,
            @RequestHeader(value = "X-Tenant-ID", defaultValue = "default") String tenantId) {

        // Accept an inbound id so a trace that started at the edge continues
        // through Java and into Python as one thread of logs.
        String correlationId = incomingId != null ? incomingId : UUID.randomUUID().toString();

        return aiClient
                .chat(request, correlationId, tenantId)
                .flatMap(
                        response ->
                                conversations
                                        .append(request, response, tenantId)
                                        .thenReturn(response))
                .map(
                        response ->
                                ResponseEntity.ok()
                                        .header(CORRELATION_HEADER, correlationId)
                                        .body(response));
    }

    /**
     * Streaming passthrough, SSE in and SSE out.
     *
     * <p>{@code Flux<ServerSentEvent<String>>} rather than {@code Flux<String>}:
     * keeping the event NAME means a client can dispatch on
     * {@code token}/{@code usage}/{@code error} instead of sniffing JSON keys.
     *
     * <p>Two things this must not do:
     *
     * <ul>
     *   <li><b>Never collect the flux.</b> {@code collectList()} or
     *       {@code block()} anywhere in this chain buffers the whole answer and
     *       silently converts streaming back into a 20-second wait -- the most
     *       common way an SSE passthrough is broken without anyone noticing.
     *   <li><b>Never retry.</b> Once a chunk has reached the browser, replaying
     *       the request duplicates text the user already read.
     * </ul>
     */
    @PostMapping(path = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<String>> stream(
            @Valid @RequestBody ChatRequest request,
            @RequestHeader(value = CORRELATION_HEADER, required = false) String incomingId,
            @RequestHeader(value = "X-Tenant-ID", defaultValue = "default") String tenantId) {

        String correlationId = incomingId != null ? incomingId : UUID.randomUUID().toString();

        Flux<ServerSentEvent<String>> tokens =
                aiClient
                        .streamChat(request, correlationId, tenantId)
                        .map(chunk -> ServerSentEvent.<String>builder().event("token").data(chunk).build());

        // A heartbeat keeps proxies and load balancers from closing an idle
        // connection during a long first-token wait. Merged rather than
        // concatenated so it interleaves with real tokens.
        Flux<ServerSentEvent<String>> heartbeat =
                Flux.interval(Duration.ofSeconds(15))
                        .map(tick -> ServerSentEvent.<String>builder().event("heartbeat").data("{}").build());

        return Flux.merge(tokens, heartbeat)
                // takeUntil on the token stream completing, so the infinite
                // heartbeat does not keep the response open forever.
                .takeUntilOther(tokens.then())
                .concatWithValues(ServerSentEvent.<String>builder().event("done").data("[DONE]").build())
                .doOnCancel(() -> log.info("client disconnected cid={}", correlationId));
    }

    /**
     * Maps validation failures to the shared error envelope.
     *
     * <p>Spring's default body is a Spring-shaped object; clients would then
     * need one parser for gateway errors and another for Python's. One shape
     * means one error handler.
     */
    @ExceptionHandler(WebExchangeBindException.class)
    public ResponseEntity<ErrorResponse> onValidationError(WebExchangeBindException exception) {
        String detail =
                exception.getFieldErrors().stream()
                        .findFirst()
                        .map(error -> error.getField() + ": " + error.getDefaultMessage())
                        .orElse("invalid request");
        return ResponseEntity.badRequest()
                .body(ErrorResponse.of("invalid_request", detail, "-"));
    }

    /**
     * Propagates upstream client errors rather than masking them.
     *
     * <p>A 402 (budget exhausted) or 422 from Python is information the caller
     * needs. Rewriting it to a 500 destroys that and invites a retry loop.
     */
    @ExceptionHandler(WebClientResponseException.class)
    public ResponseEntity<ErrorResponse> onUpstreamError(WebClientResponseException exception) {
        HttpStatus status = HttpStatus.resolve(exception.getStatusCode().value());
        if (status == null || status.is5xxServerError()) {
            log.error("upstream server error: {}", exception.getMessage());
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body(ErrorResponse.of("upstream_error", "the AI service failed", "-"));
        }
        return ResponseEntity.status(status)
                .body(ErrorResponse.of("invalid_request", exception.getResponseBodyAsString(), "-"));
    }
}
