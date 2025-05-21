from Model import Codebook
import argparse
import json
import os
import torch
from torchvision import utils as vutils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class TestDataset(Dataset):
    def __init__(self, img_dir):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000])
        ])
        self.img_list = sorted(os.listdir(img_dir), key=lambda x:int(x.split(".")[0]))
        self.img_list = [os.path.join(img_dir, i) for i in self.img_list]
    
    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert("RGB")
        return self.transform(img)

    def __len__(self):
        return len(self.img_list)


def load_config(config_path):
    json_file = open(config_path)
    json_dict = json.load(json_file)
    data = json_dict["data"]
    train = json_dict["train"]
    model = json_dict["model"]
    return data, train, model

def decode(config, batch_size, checkpoint, input_textimg_dir, output_code_dir, output_code_file, output_reconstruct_img_dir):
    data_config, train_config, model_config = load_config(config)

    patch_size = model_config["patch_size"]
    dim = model_config["dim"]
    codebook_dim = model_config["codebook_dim"]
    codebook_size = model_config["codebook_size"]
    model = Codebook(patch_size, dim, codebook_dim, codebook_size)

    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval().cuda()

    ds = TestDataset(input_textimg_dir)
    dl = DataLoader(ds, batch_size, shuffle=False)
    f = open(output_code_file, "w")
    with torch.no_grad():
        idx = 1
        for img in dl:
            output_dict = model(img.cuda())
            codes = output_dict["code"].cpu()
            imgs = output_dict["img"]
            for code, img in zip(codes, imgs):
                torch.save(code, os.path.join(output_code_dir, str(idx)+".pt"))
                new_img = img * 0.5 + 0.5
                vutils.save_image(new_img, os.path.join(output_reconstruct_img_dir, str(idx)+".jpg"))
                idx += 1
                code_str = ' '.join(map(str, code.tolist()))
                f.write(code_str+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_textimg_dir", type=str, required=True)
    parser.add_argument("--output_code_dir", type=str, required=True)
    parser.add_argument("--output_code_file", type=str, required=True)
    parser.add_argument("--output_reconstruct_img_dir", type=str, required=True)

    args = parser.parse_args()
    config = args.config
    batch = args.batch
    checkpoint = args.checkpoint
    input_textimg_dir = args.input_textimg_dir
    output_code_dir = args.output_code_dir
    output_code_file = args.output_code_file
    output_reconstruct_img_dir = args.output_reconstruct_img_dir
    
    if not os.path.exists(output_code_dir):
        os.makedirs(output_code_dir)
    if not os.path.exists(output_reconstruct_img_dir):
        os.makedirs(output_reconstruct_img_dir)
    
    decode(config, batch, checkpoint, input_textimg_dir, output_code_dir, output_code_file, output_reconstruct_img_dir)
