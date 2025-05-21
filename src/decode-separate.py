from Model import SeparateEncoder
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


def decode(batch, config_path, checkpoint_path, input_img_dir, output_background_dir, output_textimg_dir):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    data_config, train_config, model_config = load_config(config_path)

    patch_size = model_config["patch_size"]
    model = SeparateEncoder(patch_size)
    model.load_state_dict(ckpt["model_state"])
    model.cuda().eval()

    ds = TestDataset(input_img_dir)
    dl = DataLoader(ds, batch, shuffle=False)
    import time
    start = time.time()
    with torch.no_grad():
        idx = 1
        for img in dl:
            output_dict = model(img.cuda())
            back_img = output_dict["back_img"]
            text_img = output_dict["text_img"]
            for b, t in zip(back_img, text_img):
                new_back_img = b * 0.5 + 0.5
                new_text_img = t * 0.5 + 0.5
                vutils.save_image(new_back_img, os.path.join(output_background_dir, str(idx)+".jpg"))
                vutils.save_image(new_text_img, os.path.join(output_textimg_dir, str(idx)+".jpg"))
                idx += 1
    end = time.time()
    print(end-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_img_dir", type=str, required=True)
    parser.add_argument("--output_background_dir", type=str, required=True)
    parser.add_argument("--output_textimg_dir", type=str, required=True)
    
    args = parser.parse_args()

    batch = args.batch
    config_path = args.config
    checkpoint_path = args.checkpoint
    input_img_dir = args.input_img_dir
    output_background_dir = args.output_background_dir
    output_textimg_dir = args.output_textimg_dir

    if not os.path.exists(output_background_dir):
        os.makedirs(output_background_dir)
    if not os.path.exists(output_textimg_dir):
        os.makedirs(output_textimg_dir)
    
    decode(batch, config_path, checkpoint_path, input_img_dir, output_background_dir, output_textimg_dir)
