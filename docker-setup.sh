#!/bin/bash

# x11 forwarding for viewing cv2 graphs etc.
export DISPLAY=:1.0
xhost +local:root

# Set the path to your project directory on the host machine.
# HOST_PROJECT_DIR="/home/kazi/Works/Projects/DeepFlex"

# Check if a container named "tensorflow" is running and stop it if it is.
if [[ $(docker ps | awk '{if(NR>1) print $NF}') == 'deep_flex' ]]; then
    echo "Stopping tensorflow container"
    docker stop deep_flex
fi

# Prune all stopped containers.
echo "Pruning all containers"
docker container prune

# Run the TensorFlow Docker container with specified options.
docker run --gpus all  -d -t --name deep_flex \
-v "$(pwd):/home/kazi/Works/Projects/DeepFlex" \
--restart unless-stopped \
--net=host \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--volume="/etc/localtime:/etc/localtime:ro" \
--volume="/etc/timezone:/etc/timezone:ro" \
--volume="/tmp:/tmp" \
--entrypoint /bin/bash \
--user 1000:1000 \
deep_flex:5


# Start TensorBoard within the Docker container
docker exec -d deep_flex tensorboard --logdir=/home/kazi/Works/Projects/DeepFlex/logs

# Display a message indicating TensorBoard is running
echo "TensorBoard is running in the deep_flex container. You can access it from your host machine's web browser."