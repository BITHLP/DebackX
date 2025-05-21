export CUDA_VISIBLE_DEVICES=2,3,4,5
mkdir -p ../logs
time=$(date "+%Y%m%d-%H%M%S")

accelerate launch --mixed_precision fp16 --main_process_port 3190 ../src/train-codebook.py --config ../configs/config-codebook.json > ../logs/${time}train.log 2>&1
