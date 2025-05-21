export CUDA_VISIBLE_DEVICES=1

config=/data1/yztian/ACL2025/DebackX/models/fuse/config-fuse.json
ckpt=/data1/yztian/ACL2025/DebackX/models/fuse/checkpoint_best0.004.pt

python ../src/decode-fuse.py \
    --batch 32 --config ${config} --checkpoint ${ckpt} \
    --input_text_img_dir /data1/yztian/ACL2025/DebackX/results/translation/test/de/text \
    --input_back_img_dir /data1/yztian/ACL2025/DebackX/results/separate/test/back/en \
    --output_fuse_img_dir /data1/yztian/ACL2025/DebackX/results/fuse/test/en