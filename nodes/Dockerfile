# nodes/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY nodes/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY nodes ./nodes
COPY artifacts ./artifacts

CMD ["python", "-m", "nodes.drone_producer"]
