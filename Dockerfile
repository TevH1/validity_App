FROM python:3.10-slim

WORKDIR /usr/src/app


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


COPY app.py ./
COPY pipe/    ./pipe/

EXPOSE 5000


ENV PORT=5000



#    This looks for the Flask “app” object inside app.py.

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2"]

