FROM tensorflow/tensorflow:2.12.0-gpu

# Update package repositories
RUN apt-get update -y

# Upgrade pip
RUN /usr/bin/python3 -m pip install --upgrade pip

# Install system dependencies
RUN apt-get install -y ffmpeg libsm6 libxext6 python3-tk x11-apps git

# Install Python packages
RUN pip3 install bootstrap-py opencv-python matplotlib pandas seaborn tqdm plotly imageio scikit-image scikit-learn tensorflow-addons focal-loss tensorflow-datasets keras-applications keras-preprocessing keras-segmentation keras-tuner tables protobuf==3.20.* plyfile

# Create a non-root user
ARG USERNAME=kazi
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV USER=kazi

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# [Optional] Add sudo support. Omit if you don't need to install software after connecting.
RUN apt-get install -y sudo \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set the default user and working directory
USER $USERNAME
WORKDIR /home/$USERNAME

# Define the entrypoint
CMD ["/bin/bash"]
