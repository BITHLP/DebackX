export CUDA_VISIBLE_DEVICES=1

codebook_config=/data1/yztian/ACL2025/DebackX/models/codebook/config-codebook.json
codebook_ckpt=/data1/yztian/ACL2025/DebackX/models/codebook/checkpoint_best0.006.pt

trans_config=/data1/yztian/ACL2025/DebackX/models/translation/config-translation.json
trans_ckpt=/data1/yztian/ACL2025/DebackX/models/translation/checkpoint_last100000.pt

python ../src/decode-translation.py \
    --batch 16 --trans_config ${trans_config} --trans_checkpoint ${trans_ckpt} \
    --codebook_config ${codebook_config} --codebook_checkpoint ${codebook_ckpt} \
    --input_img_dir /data1/yztian/ACL2025/DebackX/results/separate/test/text/en \
    --output_img_dir /data1/yztian/ACL2025/DebackX/results/translation/test/de/text \
    --tit_path /data1/yztian/ACL2025/DebackX/results/translation/test/tit.de