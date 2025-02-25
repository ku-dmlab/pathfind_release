# Ubuntu 22.04 LTS (Jammy Jellfish)
FROM mambaorg/micromamba:1.4.3-jammy

# Tell apt-get we're never going to be able to give manual feedback.
ARG DEBIAN_FRONTEND=noninteractive
# Preset timezone that is required for some packages.
ENV TZ=Asia/Seoul

# We won't use default $MAMBA_USER.
USER root
# Install system pacakges.
RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get -y install --no-install-recommends \
        apt-utils \
        bash-completion \
        build-essential \
        ca-certificates \
        curl \
        less \
        man-db \
        manpages \
        manpages-dev \
        git \
        vim \
        unzip \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome. You can ignore user _apt Permission denied error.
RUN wget "https://dl.google.com/linux/chrome/deb/pool/main/g/google-chrome-stable/google-chrome-stable_114.0.5735.198-1_amd64.deb" \
    && apt-get update \
    && apt-get -y install --no-install-recommends ./google-chrome-stable_114.0.5735.198-1_amd64.deb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Override CUDA for Mamba package resolution.
ARG CONDA_OVERRIDE_CUDA="11.7"

# Install Python packages in requirements.txt.
COPY dev.yaml /tmp/dev.yaml
RUN micromamba install -y -n base -f /tmp/dev.yaml \
    && micromamba clean -y --all
