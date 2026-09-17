# Образ для CLI-прогноза. Данные и обученная модель монтируются
# при запуске (volume), а не встраиваются в образ — это соответствует
# тому, что датасеты и веса в реальных проектах не хранят в контейнере.
FROM python:3.11-slim

# Python не буферизует stdout — логи видны сразу
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Зависимости отдельным слоем — кэшируются между сборками,
# пока requirements.txt не изменится
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код проекта
COPY src/ ./src/
COPY predict.py ./

# Точка входа — CLI прогноза. Аргументы передаются через docker run.
ENTRYPOINT ["python", "predict.py"]
CMD ["--help"]