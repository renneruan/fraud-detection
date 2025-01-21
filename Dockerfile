FROM python:3.11-slim

RUN apt update -y && apt install awscli -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN pip cache purge
    
CMD ["python", "app.py"]