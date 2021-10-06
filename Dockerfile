FROM jjanzic/docker-python3-opencv

WORKDIR /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]