export CUDA_VISIBLE_DEVICES=1

config=/data1/yztian/ACL2025/DebackX/models/separate/config-separate.json
ckpt=/data1/yztian/ACL2025/DebackX/models/separate/checkpoint_best0.017.pt

python ../src/decode-separate.py \
    --batch 32 --config ${config} --checkpoint ${ckpt} --input_img_dir /data2/yztian/IIMT30k/TimesNewRoman/test_flickr/en/image \
    --output_background_dir /data1/yztian/ACL2025/DebackX/results/separate/test/back/en \
    --output_textimg_dir /data1/yztian/ACL2025/DebackX/results/separate/test/text/en \
