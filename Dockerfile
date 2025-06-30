FROM python:3.11-slim


WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py inference.py ./
COPY pipe/ ./pipe/

EXPOSE 5000
ENV PORT=5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120"]

