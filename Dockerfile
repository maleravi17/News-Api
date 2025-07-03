FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends 
gcc 
python3-dev 
&& rm -rf /var/lib/apt/lists/*
COPY . .RUN pip install --no-cache-dir --upgrade pip && 
pip install --no-cache-dir -r requirements.txt
EXPOSE 8005
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "$PORT"]
