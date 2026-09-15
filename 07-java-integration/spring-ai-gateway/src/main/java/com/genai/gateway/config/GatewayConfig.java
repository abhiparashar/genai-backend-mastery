package com.genai.gateway.config;

import io.netty.channel.ChannelOption;
import io.netty.handler.timeout.ReadTimeoutHandler;
import java.time.Duration;
import java.util.concurrent.TimeUnit;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.ExchangeFilterFunction;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.http.client.HttpClient;
import reactor.netty.resources.ConnectionProvider;

/**
 * WebClient and connection-pool configuration.
 *
 * <p>The defaults are not good enough for an upstream that takes 30 seconds to
 * answer, and the failure mode of leaving them alone is subtle: connection
 * exhaustion under load, which looks like the upstream being slow.
 */
@Configuration
public class GatewayConfig {

    @Bean
    public WebClient aiWebClient(AiServiceProperties properties) {
        ConnectionProvider pool =
                ConnectionProvider.builder("ai-service")
                        // Sized for concurrent SLOW calls. The default (16) is
                        // tuned for fast services; with 30-second LLM calls it
                        // becomes the bottleneck long before the upstream does.
                        .maxConnections(200)
                        // Queue rather than fail instantly when the pool is full,
                        // but bound the wait so a stampede surfaces as an error
                        // instead of unbounded latency.
                        .pendingAcquireMaxCount(500)
                        .pendingAcquireTimeout(Duration.ofSeconds(10))
                        // Recycle idle connections before the upstream or a load
                        // balancer silently closes them. Without this you get
                        // intermittent "connection reset" on the first request
                        // after a quiet period -- a classic, maddening bug.
                        .maxIdleTime(Duration.ofSeconds(55))
                        .maxLifeTime(Duration.ofMinutes(10))
                        .evictInBackground(Duration.ofSeconds(30))
                        .build();

        HttpClient httpClient =
                HttpClient.create(pool)
                        .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 5_000)
                        // Read timeout must exceed the upstream's own LLM timeout
                        // or we cancel work that was about to succeed.
                        .doOnConnected(
                                conn ->
                                        conn.addHandlerLast(
                                                new ReadTimeoutHandler(
                                                        properties.timeoutSeconds(),
                                                        TimeUnit.SECONDS)))
                        .compress(true)
                        // Streaming requires this: without it Reactor Netty may
                        // buffer the response and SSE stops being incremental.
                        .responseTimeout(Duration.ofSeconds(properties.timeoutSeconds()));

        return WebClient.builder()
                .baseUrl(properties.baseUrl())
                .clientConnector(new ReactorClientHttpConnector(httpClient))
                .defaultHeader("X-API-Key", properties.apiKey())
                .filter(logRequest())
                .codecs(
                        configurer ->
                                // Default is 256KB; a long conversation plus RAG
                                // context exceeds that and fails with an opaque
                                // DataBufferLimitException.
                                configurer.defaultCodecs().maxInMemorySize(4 * 1024 * 1024))
                .build();
    }

    /** Logs the outbound call without ever logging the API key. */
    private ExchangeFilterFunction logRequest() {
        return ExchangeFilterFunction.ofRequestProcessor(
                request -> {
                    org.slf4j.LoggerFactory.getLogger(GatewayConfig.class)
                            .debug("-> {} {}", request.method(), request.url());
                    return reactor.core.publisher.Mono.just(request);
                });
    }
}
