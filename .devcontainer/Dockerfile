ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION
ARG POETRY_VERSION=1.5.1
LABEL maintainer="RISC Software GmbH"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -yq \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    git \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/*

ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install pyenv
RUN set -ex \
    && curl https://pyenv.run | bash \
    && pyenv update \
    && eval "$(pyenv init -)" && pyenv --version

# Install python and set interpreter
RUN pyenv install -s ${PYTHON_VERSION} && pyenv global ${PYTHON_VERSION}
RUN pip install --upgrade pip

# Install and configure Poetry
ENV POETRY_VERSION=${POETRY_VERSION} \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=true
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="${PATH}:${POETRY_HOME}/bin"
