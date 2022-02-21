FROM python:3.8-slim

LABEL maintainer="Saeed Karimi <ee.saeed.karimi@gmail.com>"

# Install dependencies
RUN apt-get -y update
RUN apt-get install -y g++
RUN apt-get install -y ffmpeg

# Set working directory
WORKDIR /app

# Install pip packages.
COPY ./requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# uncomment a line in insightface library
RUN sed -i "s/#'arcface_mfn_v1': arcface_mfn_v1/'arcface_mfn_v1': arcface_mfn_v1/" /usr/local/lib/python3.8/site-packages/insightface/model_zoo/model_zoo.py

# Copy project
COPY . .

# Copy face detection and recognition models to directory
RUN mkdir -p /root/.insightface/models
RUN cp -r models/ /root/.insightface/
