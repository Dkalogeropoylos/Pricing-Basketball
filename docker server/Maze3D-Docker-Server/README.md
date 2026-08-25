# Maze Experiment Http Server

Instructions to install docker in a debian based system [here](/docker_install.md)

## Download

```shell
git clone https://github.com/Dkoutrintzes/Maze3D-Docker-Server
cd maze_server
```
## Build and run

```shell

# build the docker image with some tag ex. maze_http_server
docker build -t maze_http_server:1.0.0 .

# run the docker image with -d for detached 
# and -p for port mappings -p "host:container"
docker run -d -p "8080:5050" maze_http_server:1.0.0
```

Tested on python 3.12