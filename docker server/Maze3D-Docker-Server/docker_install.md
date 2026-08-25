# Installation for Docker and Docker compose

```shell
sudo apt update
sudo apt upgrade

# Install docker and docker-compose
sudo apt install docker.io docker-compose

# Enable docker service
sudo systemctl enable --now docker

# Verify that Docker is installed correctly by running the hello-world image.
sudo docker run hello-world

# Create the docker group.
sudo groupadd docker

# Add your user to the docker group.
sudo usermod -aG docker $USER

# Logout or run this to activate the changes to groups 
newgrp docker 

# Verify that Docker is running the hello-world image without sudo.
docker run hello-world
```