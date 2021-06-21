#!/bin/bash
scp -ri $1 ~/Documents/research/DeepLearning/blender/texture_gingiva ubuntu@$2:/home/ubuntu/blender
scp -i $1 ~/Documents/research/DeepLearning/blender/ulg.blend ubuntu@$2:/home/ubuntu/blender
scp -i $1 ~/Documents/research/DeepLearning/blender/ulg_rand.py ubuntu@$2:/home/ubuntu/blender
