package com.genai.gateway;

import com.genai.gateway.config.AiServiceProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

/**
 * Gateway entry point.
 *
 * <p>Java owns auth, validation, conversation state, cost attribution and
 * resilience. Python owns the model call. That split is the argument of this
 * whole module: it puts each concern where the ecosystem is strongest, and it
 * lets the Python service stay stateless and independently deployable.
 */
@SpringBootApplication
@EnableConfigurationProperties(AiServiceProperties.class)
public class AiGatewayApplication {

    public static void main(String[] args) {
        SpringApplication.run(AiGatewayApplication.class, args);
    }
}
