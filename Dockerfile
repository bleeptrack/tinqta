# syntax=docker/dockerfile:1

FROM python:3
WORKDIR /tinqta
COPY . .
RUN pip install -r requirements.txt 
CMD ["python", "server.py"]
EXPOSE 5000
