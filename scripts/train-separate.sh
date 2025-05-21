export CUDA_VISIBLE_DEVICES=1
mkdir -p ../logs
time=$(date "+%Y%m%d-%H%M%S")

accelerate launch --mixed_precision fp16 --main_process_port 3190 ../src/train-separate.py --config ../configs/config-separate.json > ../logs/${time}train.log 2>&1
