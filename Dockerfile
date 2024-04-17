# FROM python:3.12.2-slim

# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     wget \
#     build-essential \
#     libffi-dev \
#     libssl-dev \
#     && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y software-properties-common && \
#     wget https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda-repo-debian12-12-3-local_12.3.1-545.23.08-1_amd64.deb && \
#     dpkg -i cuda-repo-debian12-12-3-local_12.3.1-545.23.08-1_amd64.deb && \
#     cp /var/cuda-repo-debian12-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
#     apt-get update && \
#     apt-get -y install cuda-toolkit-12-3 

# RUN pip install --no-cache-dir poetry

# WORKDIR /app
# COPY . /app

# RUN poetry config virtualenvs.create false \
#     && poetry install --no-interaction --no-ansi

# # Adjust CUDACXX to the new CUDA installation
# ENV CUDACXX=/usr/local/cuda-12/bin/nvcc 
# ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all-major"
# ENV FORCE_CMAKE=1
# RUN pip3 install llama-cpp-python==0.2.56 --no-cache-dir --force-reinstall --upgrade

# EXPOSE 8501

# RUN apt-get remove

# CMD ["streamlit", "run", "app_qa.py", "--server.address=0.0.0.0"]

# Use an intermediate image to install dependencies
FROM debian:buster as builder

RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Your final image starts here
FROM python:3.12.2-slim

# Copy necessary binaries and libraries from the builder
COPY --from=builder /usr/bin/wget /usr/bin/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libffi.so.6 /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libssl.so.1.1 /usr/lib/x86_64-linux-gnu/

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install wget and other necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit
RUN apt-get update && apt-get install -y software-properties-common && \
    wget https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda-repo-debian12-12-3-local_12.3.1-545.23.08-1_amd64.deb && \
    dpkg -i cuda-repo-debian12-12-3-local_12.3.1-545.23.08-1_amd64.deb && \
    cp /var/cuda-repo-debian12-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && apt-get -y install cuda-toolkit-12-3 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

WORKDIR /app
COPY . /app

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Adjust CUDACXX to the new CUDA installation
ENV CUDACXX=/usr/local/cuda-12/bin/nvcc 
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all-major"
ENV FORCE_CMAKE=1
RUN pip3 install llama-cpp-python==0.2.56 --no-cache-dir --force-reinstall --upgrade

EXPOSE 8501

CMD ["streamlit", "run", "app_qa.py", "--server.address=0.0.0.0"]