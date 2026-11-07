# Speech-to-text for the voice studio: Whisper (English) with word timestamps on the GPU.
# Same PyTorch / CUDA 12.8 build as the Breeze image, which supports Blackwell (sm_120) GPUs.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
ENV HF_HOME=/models PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
EXPOSE 7862
CMD ["python", "server.py"]
