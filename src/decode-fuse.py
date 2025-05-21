from Model import FuseDecoder
import argparse
import json
import os
import torch
from torchvision import utils as vutils
from torchvision import transforms
from PIL import Image


def load_config(config_path):
    json_file = open(config_path)
    json_dict = json.load(json_file)
    data = json_dict["data"]
    train = json_dict["train"]
    model = json_dict["model"]
    return data, train, model


def decode(batch, config_path, checkpoint_path, input_text_img_dir, input_back_img_dir, output_fuse_img_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000])
    ])
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    data_config, train_config, model_config = load_config(config_path)

    patch_size = model_config["patch_size"]
    model = FuseDecoder(patch_size)
    model.load_state_dict(ckpt["model_state"])
    model.cuda().eval()

    text_img_name_list = sorted(os.listdir(input_text_img_dir), key=lambda x:int(x.split(".")[0]))
    back_img_name_list = sorted(os.listdir(input_back_img_dir), key=lambda x:int(x.split(".")[0]))
    text_img_tensors = []
    back_img_tensors = []
    for text, back in zip(text_img_name_list, back_img_name_list):
        text_img = Image.open(os.path.join(input_text_img_dir, text)).convert("RGB")
        back_img = Image.open(os.path.join(input_back_img_dir, back)).convert("RGB")
        text_img_tensors.append(transform(text_img))
        back_img_tensors.append(transform(back_img))
    
    import time
    start = time.time()
    with torch.no_grad():
        idx = 1
        while text_img_tensors != [] and back_img_tensors != []:
            text_img_batch = []
            back_img_batch = []
            for _ in range(batch):
                if text_img_tensors == [] and back_img_tensors == []:
                    break
                else:
                    text_img_batch.append(text_img_tensors.pop(0))
                    back_img_batch.append(back_img_tensors.pop(0))
            text_img_batch = torch.stack(text_img_batch).cuda()
            back_img_batch = torch.stack(back_img_batch).cuda()
            output_dict = model(back_img_batch, text_img_batch)
            fuse_img = output_dict["img"]
            for i in fuse_img:
                new_img = i*0.5+0.5
                vutils.save_image(new_img, os.path.join(output_fuse_img_dir, str(idx)+".jpg"))
                idx += 1
    end = time.time()
    print(end-start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_text_img_dir", type=str, required=True)
    parser.add_argument("--input_back_img_dir", type=str, required=True)
    parser.add_argument("--output_fuse_img_dir", type=str, required=True)
    
    args = parser.parse_args()

    batch = args.batch
    config_path = args.config
    checkpoint_path = args.checkpoint
    input_text_img_dir = args.input_text_img_dir
    input_back_img_dir = args.input_back_img_dir
    output_fuse_img_dir = args.output_fuse_img_dir
    if not os.path.exists(output_fuse_img_dir):
        os.makedirs(output_fuse_img_dir)

    decode(batch, config_path, checkpoint_path, input_text_img_dir, input_back_img_dir, output_fuse_img_dir)
