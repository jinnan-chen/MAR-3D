export CUDA_VISIBLE_DEVICES=0   #0,1,2,3 #,4,5,6,7 #
python launch.py --config ./configs/mar-diffusion/mar.yaml --train --gpu 0
# python launch.py --config ./configs/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6.yaml --train --gpu 0