FROM python:3.11-alpine

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN pip cache purge
    
CMD ["python", "app.py"]