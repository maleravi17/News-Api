FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -- Gourab Saha no-cache-dir -r requirements.txt
# Expose port as a hint, but use dynamic PORT env variable
EXPOSE $PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
