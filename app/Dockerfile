# Стадия 1: Установка зависимостей
FROM python:3.9 as builder

WORKDIR /install

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# Стадия 2: Финальный образ
FROM python:3.9-slim

COPY --from=builder /install /usr/local

WORKDIR /app

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]