#!/bin/sh
echo -n "Enter this Blender-Instance number: "
read INSTANCE
echo "OK! This instance is $INSTANCE"
sudo apt-get update
sudo apt-get install -y blender
sudo apt install -y python3-pip
pip install numpy
blender -b aws/blender/trial.blend -P aws/blender/ImageMaking.py
tar -cvf ${INSTANCE}.tar img
sed -e "s/テストメッセージです/$INSTANCE is Finished!/g" aws/line_notify.sh > line_notify.sh
bash line_notify.sh
# rm aws.tar
# rm aws/line_notify.sh
# rm -r aws
# rm -r img
# rm line_notify.sh