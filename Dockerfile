FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt
# upgrading flask fixes dependency conflicts between mage-ai and mlflow
RUN pip install flask --upgrade

EXPOSE 5000
EXPOSE 6789

CMD bash -c "\
    mage start \
        mageai/poleng \
        --host 0.0.0.0 --port 6789 & \
    mlflow server \
        --backend-store-uri sqlite:///mlflow/mlflow.db \
        --default-artifact-root ./mlflow/mlruns/poleng \
        --host 0.0.0.0 --port 5000"