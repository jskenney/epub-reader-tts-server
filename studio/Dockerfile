# Voice studio for the EPUB reader's Server voice: a Gradio app that talks to the Breeze TTS API and
# serves the voices folder to the reader. It doesn't need a GPU.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
ENV VOICES_DIR=/data/voices BREEZE_URL=http://breeze-tts:7860 PYTHONUNBUFFERED=1 GRADIO_ANALYTICS_ENABLED=False
EXPOSE 7861
CMD ["python", "app.py"]
