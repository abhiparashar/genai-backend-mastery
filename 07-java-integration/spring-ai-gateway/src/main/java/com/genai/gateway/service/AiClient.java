package com.genai.gateway.service;

import com.genai.gateway.config.AiServiceProperties;
import com.genai.gateway.dto.Dtos.ChatRequest;
import com.genai.gateway.dto.Dtos.ChatResponse;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.github.resilience4j.reactor.circuitbreaker.operator.CircuitBreakerOperator;
import io.github.resilience4j.reactor.retry.RetryOperator;
import io.github.resilience4j.retry.Retry;
import io.github.resilience4j.retry.RetryRegistry;
import java.time.Duration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

/**
 * Calls the Python AI service with the resilience policy the contract requires.
 *
 * <h2>The retry rule that matters</h2>
 *
 * Retry ONLY 429, 502, 503, 504 and connect/read timeouts. NEVER 400, 401, 402,
 * 413 or 422. Those are our bug or our quota; retrying sends the same bad
 * request again, triples the latency, and guarantees the same failure.
 *
 * <h2>Why 4xx must not open the circuit breaker</h2>
 *
 * This is the most common breaker misconfiguration there is. If client errors
 * count as failures, one caller sending malformed requests trips the breaker
 * and takes a perfectly healthy upstream offline for everyone else. The
 * breaker measures UPSTREAM health, so only 5xx and timeouts may count.
 *
 * <h2>Timeout budget</h2>
 *
 * The gateway's time limiter (35s) must EXCEED the Python service's own LLM
 * timeout (30s). Invert those and the gateway cancels work that was about to
 * succeed, producing timeouts that vanish when you look at the upstream logs.
 */
@Service
public class AiClient {

    private static final Logger log = LoggerFactory.getLogger(AiClient.class);

    private final WebClient webClient;
    private final AiServiceProperties properties;
    private final CircuitBreaker circuitBreaker;
    private final Retry retry;

    public AiClient(
            WebClient aiWebClient,
            AiServiceProperties properties,
            CircuitBreakerRegistry circuitBreakerRegistry,
            RetryRegistry retryRegistry) {
        this.webClient = aiWebClient;
        this.properties = properties;
        this.circuitBreaker = circuitBreakerRegistry.circuitBreaker("aiService");
        this.retry = retryRegistry.retry("aiService");

        this.circuitBreaker
                .getEventPublisher()
                .onStateTransition(
                        event ->
                                // Log loudly: a breaker opening is the single most
                                // useful signal that an upstream is in trouble, and
                                // it belongs on a dashboard, not buried in DEBUG.
                                log.warn(
                                        "circuit breaker {} -> {}",
                                        event.getStateTransition().getFromState(),
                                        event.getStateTransition().getToState()));
    }

    /** Synchronous-style completion, returned as a Mono. */
    public Mono<ChatResponse> chat(ChatRequest request, String correlationId, String tenantId) {
        return webClient
                .post()
                .uri("/v1/chat")
                .header("X-Correlation-ID", correlationId)
                .header("X-Tenant-ID", tenantId)
                .bodyValue(request)
                .retrieve()
                .bodyToMono(ChatResponse.class)
                // Order matters: timeout innermost so EACH attempt is bounded,
                // then retry, then the breaker observing the final outcome.
                .timeout(Duration.ofSeconds(properties.timeoutSeconds()))
                .transformDeferred(RetryOperator.of(retry))
                .transformDeferred(CircuitBreakerOperator.of(circuitBreaker))
                .doOnSubscribe(s -> log.debug("calling ai-service cid={}", correlationId))
                .onErrorResume(error -> fallback(error, correlationId));
    }

    /**
     * Streaming passthrough.
     *
     * <p>Deliberately NOT retried. Once the first chunk has reached the browser,
     * replaying the request would duplicate text the user already saw. Streams
     * fail fast and the client decides whether to start over.
     *
     * <p>The breaker still applies: it protects the upstream from a flood of new
     * stream attempts while it is down.
     */
    public Flux<String> streamChat(ChatRequest request, String correlationId, String tenantId) {
        return webClient
                .post()
                .uri("/v1/chat/stream")
                .accept(MediaType.TEXT_EVENT_STREAM)
                .header("X-Correlation-ID", correlationId)
                .header("X-Tenant-ID", tenantId)
                .bodyValue(request)
                .retrieve()
                .bodyToFlux(String.class)
                .transformDeferred(CircuitBreakerOperator.of(circuitBreaker))
                .doOnCancel(
                        // Cancellation propagates to the upstream HTTP request, so a
                        // browser closing the tab stops us paying for tokens nobody
                        // will read. This is free with Reactor and easy to lose if
                        // you buffer the stream into a list first.
                        () -> log.info("client cancelled stream cid={}", correlationId))
                .onErrorResume(
                        error -> {
                            log.warn("stream failed cid={}: {}", correlationId, error.toString());
                            return Flux.just("The AI service is temporarily unavailable.");
                        });
    }

    /**
     * Typed fallback.
     *
     * <p>Returning degraded content beats propagating a 500: the user gets a
     * sentence instead of a stack trace, and the caller's own SLA survives an
     * upstream incident. Client errors are re-thrown untouched, because a 422
     * is information the caller needs, not a failure to paper over.
     */
    private Mono<ChatResponse> fallback(Throwable error, String correlationId) {
        if (error instanceof WebClientResponseException http && http.getStatusCode().is4xxClientError()) {
            return Mono.error(error);
        }
        log.error("ai-service unavailable cid={}: {}", correlationId, error.toString());
        return Mono.just(ChatResponse.degraded(correlationId));
    }

    /** Exposed for the readiness probe and tests. */
    public String circuitState() {
        return circuitBreaker.getState().name();
    }
}
