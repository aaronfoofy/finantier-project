FROM python:3.9-slim
COPY ./src /app/src
COPY ./requirements.txt /app
COPY ./model.pkl /app
COPY ./encoder.pkl /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8000

CMD [ "uvicorn", "src.main:app", "--host=0.0.0.0","--reload"]
