package com.genai.gateway.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.Valid;
import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import java.util.List;

/**
 * Wire contract shared with the Python service.
 *
 * <p>Java {@code record}s here, which are the closest thing to a Pydantic model:
 * immutable, concise, with generated equals/hashCode/toString. The validation
 * annotations are Bean Validation, and Spring rejects a bad body with a 400
 * before the controller method runs -- exactly what Pydantic does with a 422.
 *
 * <p>Validating HERE as well as in Python is deliberate, not duplication. The
 * gateway is the trust boundary: rejecting a malformed request costs a
 * microsecond locally versus a network round trip, and it means the Python
 * service can be locked down to internal traffic only.
 */
public final class Dtos {

    private Dtos() {}

    /** One conversation turn. */
    public record MessageDto(
            @NotBlank @Pattern(regexp = "system|user|assistant") String role,
            @NotBlank @Size(max = 32_000) String content) {}

    /**
     * Request body.
     *
     * <p>{@code messages[]} rather than a bare prompt: it maps 1:1 onto both
     * provider payloads and is multi-turn native. A {@code prompt} + {@code system}
     * shape cannot express an assistant turn, so history has to be smuggled in
     * out of band.
     */
    public record ChatRequest(
            @NotEmpty @Size(max = 100) @Valid List<MessageDto> messages,
            String model,
            @DecimalMin("0.0") @DecimalMax("2.0") Double temperature,
            @Min(1) @Max(8192) Integer maxTokens,
            @Size(max = 128) String conversationId) {

        /** Null-safe accessor so the service layer never branches on null. */
        public double temperatureOrDefault() {
            return temperature == null ? 0.0 : temperature;
        }
    }

    /**
     * Token accounting.
     *
     * <p>{@code @JsonProperty} maps Python's snake_case onto Java's camelCase.
     * Doing it per-field rather than with a global naming strategy keeps the
     * mapping explicit and greppable -- when the contract changes, the diff
     * shows exactly which field moved.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record UsageDto(
            @JsonProperty("input_tokens") int inputTokens,
            @JsonProperty("output_tokens") int outputTokens,
            @JsonProperty("total_tokens") int totalTokens) {

        public static UsageDto empty() {
            return new UsageDto(0, 0, 0);
        }
    }

    /**
     * Successful completion.
     *
     * <p>{@code ignoreUnknown = true} is a deliberate compatibility choice: the
     * Python service can add a response field without breaking every deployed
     * gateway. Strict deserialization turns an additive change into an outage.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ChatResponse(
            String text,
            String model,
            @JsonProperty("finish_reason") String finishReason,
            UsageDto usage,
            @JsonProperty("cost_usd") double costUsd,
            @JsonProperty("latency_ms") double latencyMs,
            boolean cached,
            @JsonProperty("correlation_id") String correlationId) {

        /** Fallback returned when the upstream is unavailable. */
        public static ChatResponse degraded(String correlationId) {
            return new ChatResponse(
                    "The AI service is temporarily unavailable. Please try again shortly.",
                    "fallback",
                    "error",
                    UsageDto.empty(),
                    0.0,
                    0.0,
                    false,
                    correlationId);
        }
    }

    /** The shared error envelope. */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ErrorDetail(
            String code, String message, @JsonProperty("correlation_id") String correlationId) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ErrorResponse(ErrorDetail error) {
        public static ErrorResponse of(String code, String message, String correlationId) {
            return new ErrorResponse(new ErrorDetail(code, message, correlationId));
        }
    }
}
