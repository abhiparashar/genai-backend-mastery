package com.genai.gateway.config;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

/**
 * Typed configuration for the downstream Python AI service.
 *
 * <p>{@code @ConfigurationProperties} on a record is the Java twin of Pydantic
 * {@code BaseSettings}: bind from config, validate on startup, inject
 * everywhere. {@code @Validated} makes a bad value a STARTUP failure rather
 * than a NullPointerException on the first request at 3am.
 *
 * @param baseUrl base URL of the Python service, e.g. http://ai-service:8000
 * @param apiKey key sent as X-API-Key
 * @param timeoutSeconds per-request read timeout; MUST exceed the upstream's
 *     own LLM timeout (30s) or we cancel work that was about to succeed
 */
@Validated
@ConfigurationProperties(prefix = "ai.service")
public record AiServiceProperties(
        @NotBlank String baseUrl, @NotBlank String apiKey, @Min(1) int timeoutSeconds) {

    public AiServiceProperties {
        if (timeoutSeconds < 31) {
            // Fail fast with the reasoning, not just the rule. A gateway
            // timeout below the upstream's own produces "timeouts" that leave
            // no trace in the upstream logs, which is hours of confusion.
            throw new IllegalArgumentException(
                    "ai.service.timeout-seconds must exceed the Python service's LLM timeout (30s); got "
                            + timeoutSeconds);
        }
    }
}
