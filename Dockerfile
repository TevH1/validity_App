FROM python:3.11-slim


WORKDIR /usr/src/app


RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.7.1+cpu

RUN pip install --no-cache-dir sentence-transformers==4.1.0

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py inference.py ./
COPY pipe/ ./pipe/

EXPOSE 5000
ENV PORT=5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120"]
