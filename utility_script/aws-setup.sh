#!/bin/bash
sudo apt-get update
sudo apt-get upgrade
sudo curl -O https://bootstrap.pypa.io/get-pip.py
sudo sed -i -e '$a export\ PATH=~/.local/bin:$PATH' .profile
sudo source ~/.profile
#
sudo apt-get install python3.7 python3.7-dev
sudo python get-pip.py --user
sudo apt install awscli
sudo apt-get install tree
sudo pip3 install jupyter
#
sudo add-apt-repository ppa:thomas-schiex/blender-legacy
sudo apt-get update
sudo apt-get install blender
#
sudo apt-get install vim ssh git build-essential libfuse-dev libcurl4-openssl-dev libxml2-dev mime-support automake libtool pkg-config libssl-dev libpng-dev libfreetype6-dev s3fs python-pip python-dev python-setuptools python-numpy python-scipy libtlas-dev libatlas3gf-base
#
git clone https://github.com/s3fs-fuse/s3fs-fuse
cd ${HOME}/s3fs-fuse/
./autogen.sh
./configure --prefix=/usr --with-openssl
make
sudo make install
cd ${HOME}
sudo mkdir -p ${HOME}/mnt/s3
sudo chmod 777 ${HOME}/mnt/s3
sudo /usr/bin/s3fs seino ${HOME}/mnt/s3 -o rw,allow_other,uid=1000,gid=1000,iam_role="seino-s3"
#
mkdir -p ${HOME}/blender
aws s3 cp s3://seino/blender/ ./blender --recursive
#
sudo sed -i -e '$a sudo\ /usr/bin/s3fs#seino\ ${HOME}/mnt/s3\ fuse allow_other,uid=1000,gid=1000,iam_role="seino-s3"' /etc/fstab
#
sudo mkdir -p ${HOME}/img
sudo mkdir -p ${HOME}/ssl
#
sudo pip install tensorflow keras
sudo pip3 install keras
sudo pip install pathlib numpy matplotlib scikit-learn
#
jupyter notebook password
