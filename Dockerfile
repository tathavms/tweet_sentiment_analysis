# === STAGE 1: The Assembly & Build Environment ===
FROM python:3.12-slim AS builder
WORKDIR /app

#Installs the basic C++ compiler tools needed to install wheel dependencies. 
#The clean-up command at the end deletes temporary system installation files to save space.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# === STAGE 2: The Lightweight Runtime Environment ===
FROM python:3.12-slim AS runtime
WORKDIR /app

COPY --from=builder /root/.local /root/.local

#Copying the specific execution code and lightweight layout files into the container. We intentionally leave out model.safetensors so it doesn't bloat the image(Limited EC2 space).
COPY ./app ./app
COPY ./model_artifacts/config.json ./model_artifacts/config.json
COPY ./model_artifacts/label_map.json ./model_artifacts/label_map.json
COPY ./model_artifacts/tokenizer_config.json ./model_artifacts/tokenizer_config.json
COPY ./model_artifacts/vocab.txt ./model_artifacts/vocab.txt
COPY ./model_artifacts/special_tokens_map.json ./model_artifacts/special_tokens_map.json

ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]