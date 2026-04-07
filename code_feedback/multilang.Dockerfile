# Use Ubuntu 22.04 for a stable multi-lang environment
FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies for Python, C, C++, Java, and Node.js
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    openjdk-17-jdk \
    curl \
    && curl -sL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /usr/src/app

# Default command (will be overridden by dynamic_analyzer.py)
CMD ["bash"]
