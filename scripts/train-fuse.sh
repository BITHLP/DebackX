export CUDA_VISIBLE_DEVICES=1
mkdir -p ../logs
time=$(date "+%Y%m%d-%H%M%S")

accelerate launch --mixed_precision fp16 --main_process_port 3190 ../src/train-fuse.py --config ../configs/config-fuse.json > ../logs/${time}train.log 2>&1
