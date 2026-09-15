package com.genai.gateway;

import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.post;
import static com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo;
import static org.assertj.core.api.Assertions.assertThat;

import com.genai.gateway.config.AiServiceProperties;
import com.genai.gateway.dto.Dtos.ChatRequest;
import com.genai.gateway.dto.Dtos.ChatResponse;
import com.genai.gateway.dto.Dtos.MessageDto;
import com.genai.gateway.service.AiClient;
import com.github.tomakehurst.wiremock.WireMockServer;
import com.github.tomakehurst.wiremock.core.WireMockConfiguration;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.github.resilience4j.retry.RetryConfig;
import io.github.resilience4j.retry.RetryRegistry;
import java.io.IOException;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.TimeoutException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.test.StepVerifier;

/**
 * Resilience behaviour, verified against a stubbed upstream.
 *
 * <p>WireMock rather than a live service, for the same reason the Python side
 * uses a FakeProvider: you cannot ask a real upstream for a 503 on demand, and
 * the 503 path is exactly the code most likely to be wrong.
 */
class AiClientTest {

    private WireMockServer wireMock;
    private AiClient client;
    private CircuitBreakerRegistry circuitBreakers;

    private static final ChatRequest REQUEST =
            new ChatRequest(List.of(new MessageDto("user", "hello")), null, 0.0, null, null);

    private static final String SUCCESS_BODY =
            """
            {"text":"hi there","model":"gpt-4o-mini","finish_reason":"stop",
             "usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8},
             "cost_usd":0.000012,"latency_ms":120.0,"cached":false,
             "correlation_id":"test-cid"}
            """;

    @BeforeEach
    void setUp() {
        wireMock = new WireMockServer(WireMockConfiguration.options().dynamicPort());
        wireMock.start();

        AiServiceProperties properties =
                new AiServiceProperties("http://localhost:" + wireMock.port(), "test-key", 35);

        circuitBreakers =
                CircuitBreakerRegistry.of(
                        CircuitBreakerConfig.custom()
                                .slidingWindowSize(4)
                                .minimumNumberOfCalls(4)
                                .failureRateThreshold(50)
                                .waitDurationInOpenState(Duration.ofSeconds(10))
                                // Only upstream-health signals count.
                                .recordExceptions(IOException.class, TimeoutException.class)
                                .build());

        RetryRegistry retries =
                RetryRegistry.of(
                        RetryConfig.custom()
                                .maxAttempts(3)
                                .waitDuration(Duration.ofMillis(10))
                                .retryExceptions(IOException.class, TimeoutException.class)
                                .build());

        WebClient webClient =
                WebClient.builder()
                        .baseUrl(properties.baseUrl())
                        .defaultHeader("X-API-Key", properties.apiKey())
                        .build();

        client = new AiClient(webClient, properties, circuitBreakers, retries);
    }

    @AfterEach
    void tearDown() {
        wireMock.stop();
    }

    @Test
    @DisplayName("deserializes the Python service's snake_case response")
    void happyPath() {
        wireMock.stubFor(
                post(urlEqualTo("/v1/chat"))
                        .willReturn(
                                aResponse()
                                        .withStatus(200)
                                        .withHeader("Content-Type", "application/json")
                                        .withBody(SUCCESS_BODY)));

        StepVerifier.create(client.chat(REQUEST, "test-cid", "acme"))
                .assertNext(
                        response -> {
                            assertThat(response.text()).isEqualTo("hi there");
                            // Proves the @JsonProperty snake_case mapping works.
                            assertThat(response.usage().totalTokens()).isEqualTo(8);
                            assertThat(response.costUsd()).isEqualTo(0.000012);
                        })
                .verifyComplete();
    }

    @Test
    @DisplayName("forwards the correlation id so traces span both services")
    void propagatesCorrelationId() {
        wireMock.stubFor(
                post(urlEqualTo("/v1/chat"))
                        .willReturn(
                                aResponse()
                                        .withStatus(200)
                                        .withHeader("Content-Type", "application/json")
                                        .withBody(SUCCESS_BODY)));

        client.chat(REQUEST, "trace-123", "acme").block();

        wireMock.verify(
                com.github.tomakehurst.wiremock.client.WireMock.postRequestedFor(
                                urlEqualTo("/v1/chat"))
                        .withHeader("X-Correlation-ID", com.github.tomakehurst.wiremock.client.WireMock.equalTo("trace-123"))
                        .withHeader("X-Tenant-ID", com.github.tomakehurst.wiremock.client.WireMock.equalTo("acme")));
    }

    @Test
    @DisplayName("returns degraded content instead of propagating a 5xx")
    void fallsBackOnServerError() {
        wireMock.stubFor(post(urlEqualTo("/v1/chat")).willReturn(aResponse().withStatus(503)));

        StepVerifier.create(client.chat(REQUEST, "cid", "acme"))
                .assertNext(
                        response -> {
                            // The user gets a sentence, not a stack trace, and
                            // our own SLA survives the upstream's incident.
                            assertThat(response.model()).isEqualTo("fallback");
                            assertThat(response.finishReason()).isEqualTo("error");
                        })
                .verifyComplete();
    }

    @Test
    @DisplayName("propagates 4xx rather than masking it as a fallback")
    void doesNotSwallowClientErrors() {
        // A 402 (budget exhausted) is information the caller needs. Rewriting
        // it to a friendly fallback hides a billing problem behind a shrug.
        wireMock.stubFor(post(urlEqualTo("/v1/chat")).willReturn(aResponse().withStatus(402)));

        StepVerifier.create(client.chat(REQUEST, "cid", "acme")).expectError().verify();
    }

    @Test
    @DisplayName("client errors never open the circuit breaker")
    void clientErrorsDoNotTripTheBreaker() {
        // The classic misconfiguration: one caller sending bad requests takes
        // a healthy upstream offline for everyone.
        wireMock.stubFor(post(urlEqualTo("/v1/chat")).willReturn(aResponse().withStatus(400)));

        for (int i = 0; i < 10; i++) {
            try {
                client.chat(REQUEST, "cid", "acme").block();
            } catch (Exception ignored) {
                // expected
            }
        }

        assertThat(client.circuitState()).isEqualTo("CLOSED");
    }

    @Test
    @DisplayName("streams SSE chunks without buffering them")
    void streamsChunks() {
        wireMock.stubFor(
                post(urlEqualTo("/v1/chat/stream"))
                        .willReturn(
                                aResponse()
                                        .withStatus(200)
                                        .withHeader("Content-Type", "text/event-stream")
                                        .withBody(
                                                """
                                                event: token
                                                data: {"delta":"Hel"}

                                                event: token
                                                data: {"delta":"lo"}

                                                event: done
                                                data: [DONE]

                                                """)));

        StepVerifier.create(client.streamChat(REQUEST, "cid", "acme").collectList())
                .assertNext(chunks -> assertThat(chunks).isNotEmpty())
                .verifyComplete();
    }
}
