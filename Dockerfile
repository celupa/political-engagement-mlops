FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt
# upgrading flask fixes dependency conflicts between mage-ai and mlflow
RUN pip install flask --upgrade