package com.genai.gateway.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.genai.gateway.dto.Dtos.ChatRequest;
import com.genai.gateway.dto.Dtos.ChatResponse;
import com.genai.gateway.dto.Dtos.MessageDto;
import java.time.Duration;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.redis.core.ReactiveStringRedisTemplate;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

/**
 * Conversation history in Redis, with a TTL and a token-aware trim.
 *
 * <h2>Why this lives in Java, not Python</h2>
 *
 * State, sessions and storage are exactly what the Java side is already good
 * at. Putting them here keeps the Python service STATELESS, which means it
 * can be scaled, restarted and deployed freely without anyone losing a
 * conversation. That boundary is the whole point of the hybrid architecture.
 *
 * <h2>Why a TTL is mandatory</h2>
 *
 * Conversations without an expiry grow forever. A chat product with 100k
 * users and no TTL fills Redis, and Redis evicting keys under memory pressure
 * is a far worse outage than a conversation politely expiring after a day.
 *
 * <h2>Everything here is reactive</h2>
 *
 * A blocking Redis call inside a WebFlux handler stalls the event loop and
 * silently destroys the concurrency the whole gateway exists to provide. It
 * is exactly the same mistake as {@code time.sleep()} inside asyncio.
 */
@Service
public class ConversationService {

    private static final Logger log = LoggerFactory.getLogger(ConversationService.class);
    private static final Duration TTL = Duration.ofHours(24);
    private static final int MAX_TURNS = 20;

    private final ReactiveStringRedisTemplate redis;
    private final ObjectMapper objectMapper;

    public ConversationService(ReactiveStringRedisTemplate redis, ObjectMapper objectMapper) {
        this.redis = redis;
        this.objectMapper = objectMapper;
    }

    private static String key(String tenantId, String conversationId) {
        // Tenant in the key prefix, always. It makes cross-tenant reads
        // structurally impossible rather than a filtering mistake away.
        return "conv:%s:%s".formatted(tenantId, conversationId);
    }

    /** Append the latest exchange, trim to the window, and refresh the TTL. */
    public Mono<Void> append(ChatRequest request, ChatResponse response, String tenantId) {
        if (request.conversationId() == null) {
            return Mono.empty();
        }
        String redisKey = key(tenantId, request.conversationId());
        MessageDto lastUser = request.messages().get(request.messages().size() - 1);

        return Mono.fromCallable(
                        () ->
                                List.of(
                                        objectMapper.writeValueAsString(lastUser),
                                        objectMapper.writeValueAsString(
                                                new MessageDto("assistant", response.text()))))
                .flatMapMany(entries -> redis.opsForList().rightPushAll(redisKey, entries))
                // Trim to the newest MAX_TURNS*2 entries. Bounding here means
                // the prompt we rebuild later cannot grow without limit, which
                // is the O(N^2) token cost every chat product rediscovers.
                .then(redis.opsForList().trim(redisKey, -MAX_TURNS * 2L, -1))
                // Refresh on write, so an ACTIVE conversation survives while an
                // abandoned one still expires.
                .then(redis.expire(redisKey, TTL))
                .doOnError(error -> log.warn("failed to persist conversation: {}", error.toString()))
                // History is a nice-to-have. Losing it must never fail the
                // user's request, which already succeeded.
                .onErrorResume(error -> Mono.just(false))
                .then();
    }

    /** Load prior turns for a conversation, oldest first. */
    public Flux<MessageDto> history(String tenantId, String conversationId) {
        return redis.opsForList()
                .range(key(tenantId, conversationId), 0, -1)
                .flatMap(this::parse)
                .onErrorResume(
                        error -> {
                            log.warn("failed to load conversation: {}", error.toString());
                            return Flux.empty();
                        });
    }

    private Mono<MessageDto> parse(String raw) {
        try {
            return Mono.just(objectMapper.readValue(raw, MessageDto.class));
        } catch (JsonProcessingException exception) {
            // Skip a corrupt entry rather than failing the whole conversation.
            log.warn("dropping unparseable history entry");
            return Mono.empty();
        }
    }

    public Mono<Long> clear(String tenantId, String conversationId) {
        return redis.delete(key(tenantId, conversationId));
    }
}
