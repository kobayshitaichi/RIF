FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
LABEL mantainer="T <taichi.kobayashi@aoki-medialab.jp>"

ENV DEBIAN_FRONTEND "noninteractive"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub #15分
RUN apt update && \
    apt -y \
        -o Dpkg::Options::="--force-confdef" \
        -o Dpkg::Options::="--force-confold" dist-upgrade && \
# Mainly pyenv dependencies
    apt install -y --no-install-recommends \
        make build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev libxmlsec1-dev libxml2-dev liblzma-dev python-openssl \
        git libgl1-mesa-dev && \
# Other stuff that's handy
    apt install -y \
        python-dev libgtk2.0-dev \
        pkg-config libevent-dev automake \
        sudo software-properties-common locales \
        tmux emacs \
    apt-get -y install ffmpeg
# Set local
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:UTF-8
ENV LC_ALL en_US.UTF-8
# Manage user
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile && \
    useradd -m -d /home/ubuntu ubuntu && \
    gpasswd -a ubuntu sudo && \
    chown -R ubuntu:ubuntu /home/ubuntu
USER ubuntu
WORKDIR /home/ubuntu
ENV HOME /home/ubuntu

# Setup python environment
RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN eval "$(pyenv init --path)" && \
    pyenv install 3.8.8 && \
    pyenv rehash && \
    pyenv global 3.8.8 && \
    pip install --upgrade pip
RUN pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install ipython \
        jupyter ipdb tqdm numpy==1.19.5 pandas timm\
        opencv-python pytorch-lightning==1.8.6 torchmetrics\
        pyyaml matplotlib 
    

# CMD: when run docker
CMD ["bash"]









