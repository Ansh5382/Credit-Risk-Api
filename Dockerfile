FROM python:3.13-slim

# LightGBM needs this system library
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the Python libraries first (so Docker can reuse this step)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the files the API needs
COPY main.py predict.py features.py index.html model.txt model_info.json ./

# Hosting sites tell us which port to use through $PORT
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
