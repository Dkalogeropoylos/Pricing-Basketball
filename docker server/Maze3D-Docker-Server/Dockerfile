FROM python:3.8-slim

WORKDIR /usr/data
RUN mkdir -p /usr/app-data

COPY requirements.txt /usr/src/app/requirements.txt
WORKDIR /usr/src/app
RUN pip install -r requirements.txt

COPY ./ /usr/src/app

EXPOSE 5050

CMD gunicorn server:app

# docker build -t comet.app.orbitsystems.gr/maze_http_server:1.0.3 .
# docker run -d -p "8080:5050" comet.app.orbitsystems.gr/maze_http_server:1.0.3
# docker push comet.app.orbitsystems.gr/maze_http_server:1.0.3