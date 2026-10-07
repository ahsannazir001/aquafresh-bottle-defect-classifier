FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --timeout=300 --retries=10 -r requirements.txt

COPY api ./api
COPY models/best_tuned_resnet18.pt ./models/best_tuned_resnet18.pt

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]