FROM python:3.13-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements-api.txt

COPY config.py .
COPY app ./app
COPY data ./data
COPY models ./models
COPY outputs/mobilenet.pt ./outputs/mobilenet.pt

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]