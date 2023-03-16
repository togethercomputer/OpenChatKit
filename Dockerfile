# Base image
FROM ubuntu:20.04

# Set working directory
WORKDIR /app

# Update and install required packages
RUN apt-get update && \
    apt-get install -y git-lfs wget && \
    rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set conda to automatically activate base environment on login
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Create OpenChatKit environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Install Git LFS
RUN git lfs install

# Copy OpenChatKit code
COPY . .

# Prepare GPT-NeoX-20B model
RUN python pretrained/GPT-NeoX-20B/prepare.py

# Set entrypoint to bash shell
ENTRYPOINT ["/bin/bash"]
