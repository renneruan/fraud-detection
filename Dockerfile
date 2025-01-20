FROM python:3.11.11-slim-bullseye

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN chmod +x set_env.sh && ./set_env.sh

CMD ["python", "app.py"]