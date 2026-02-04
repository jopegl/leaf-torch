python train.py --backbone xception --lr 0.01 --workers 2 --epochs 80 --batch-size 2 --gpu-ids 0 --checkname deeplab-xception --eval-interval 1 --dataset leaf --crop-size 512 --base-size 512 --sync-bn False --resume always_saver/checkpoint_epoch_75.pth.tar

