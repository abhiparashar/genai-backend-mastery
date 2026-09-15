# Layered build: dependencies and application code land in separate image
# layers, so a code-only change rebuilds a few MB rather than the whole fat jar.
#
# Uses a Maven base image rather than the Maven Wrapper. The wrapper is the
# better choice in a real repo (it pins the Maven version for everyone), but
# it needs .mvn/ and mvnw committed, and shipping a Dockerfile that references
# files which do not exist is worse than being explicit about the dependency.
# To switch: commit the wrapper with `mvn wrapper:wrapper`, then use
# `./mvnw` below with a plain JDK base image.

FROM maven:3.9-eclipse-temurin-17 AS builder
WORKDIR /build

# Resolve dependencies first, from the pom alone. Docker caches this layer, so
# it is reused on every build where dependencies did not change.
COPY pom.xml .
RUN mvn -B dependency:go-offline

COPY src ./src
RUN mvn -B clean package -DskipTests \
    && java -Djarmode=layertools -jar target/*.jar extract --destination extracted

FROM eclipse-temurin:17-jre-alpine
RUN addgroup -S app && adduser -S -G app app
WORKDIR /app

# Ordered least- to most-frequently-changed. That ordering is the entire
# reason layered jars are worth the extra build step.
COPY --from=builder /build/extracted/dependencies/ ./
COPY --from=builder /build/extracted/spring-boot-loader/ ./
COPY --from=builder /build/extracted/snapshot-dependencies/ ./
COPY --from=builder /build/extracted/application/ ./

USER app
EXPOSE 8080

# MaxRAMPercentage, never -Xmx. With a fixed heap the JVM sizes itself against
# the HOST's memory and gets OOM-killed by the container limit; the percentage
# form respects the cgroup limit, which is the single most common way a JVM
# misbehaves in Kubernetes.
ENV JAVA_OPTS="-XX:MaxRAMPercentage=75 -XX:+UseG1GC"

# Liveness, not readiness: a healthcheck that probes dependencies restarts the
# container whenever a dependency blips.
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD wget -qO- http://localhost:8080/actuator/health/liveness || exit 1

ENTRYPOINT ["sh", "-c", "exec java $JAVA_OPTS org.springframework.boot.loader.launch.JarLauncher"]
