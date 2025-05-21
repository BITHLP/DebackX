export CUDA_VISIBLE_DEVICES=1

config=/data1/yztian/ACL2025/DebackX/models/codebook/config-codebook.json
ckpt=/data1/yztian/ACL2025/DebackX/models/codebook/checkpoint_best0.006.pt

python ../src/decode-codebook.py \
    --batch 256 --config ${config} --checkpoint ${ckpt} --input_textimg_dir /data2/yztian/IIMT30k/TimesNewRoman/val/en/text \
    --output_code_dir /data1/yztian/ACL2025/DebackX/results/codebook/val/en/code \
    --output_code_file /data1/yztian/ACL2025/DebackX/results/codebook/val/en/code.txt \
    --output_reconstruct_img_dir /data1/yztian/ACL2025/DebackX/results/codebook/val/en/text

python ../src/decode-codebook.py \
    --batch 256 --config ${config} --checkpoint ${ckpt} --input_textimg_dir /data2/yztian/IIMT30k/TimesNewRoman/val/de/text \
    --output_code_dir /data1/yztian/ACL2025/DebackX/results/codebook/val/de/code \
    --output_code_file /data1/yztian/ACL2025/DebackX/results/codebook/val/de/code.txt \
    --output_reconstruct_img_dir /data1/yztian/ACL2025/DebackX/results/codebook/val/de/text

python ../src/decode-codebook.py \
    --batch 256 --config ${config} --checkpoint ${ckpt} --input_textimg_dir /data2/yztian/IIMT30k/TimesNewRoman/train/en/text \
    --output_code_dir /data1/yztian/ACL2025/DebackX/results/codebook/train/en/code \
    --output_code_file /data1/yztian/ACL2025/DebackX/results/codebook/train/en/code.txt \
    --output_reconstruct_img_dir /data1/yztian/ACL2025/DebackX/results/codebook/train/en/text

python ../src/decode-codebook.py \
    --batch 256 --config ${config} --checkpoint ${ckpt} --input_textimg_dir /data2/yztian/IIMT30k/TimesNewRoman/train/de/text \
    --output_code_dir /data1/yztian/ACL2025/DebackX/results/codebook/train/de/code \
    --output_code_file /data1/yztian/ACL2025/DebackX/results/codebook/train/de/code.txt \
    --output_reconstruct_img_dir /data1/yztian/ACL2025/DebackX/results/codebook/train/de/text
