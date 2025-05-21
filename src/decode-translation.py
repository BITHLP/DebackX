from Model import Codebook
from Translation import AuxTITTransformer
import sentencepiece as sp
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


def decode(batch, trans_config, trans_checkpoint, codebook_config, codebook_checkpoint, input_img_dir, output_img_dir, tit_path):
    codebook_ckpt = torch.load(codebook_checkpoint, map_location="cpu")
    data_config, train_config, model_config = load_config(codebook_config)

    patch_size = model_config["patch_size"]
    dim = model_config["dim"]
    codebook_dim = model_config["codebook_dim"]
    codebook_size = model_config["codebook_size"]
    codebook = Codebook(patch_size, dim, codebook_dim, codebook_size)
    codebook.load_state_dict(codebook_ckpt["model_state"])
    codebook.cuda().eval()

    translation_ckpt = torch.load(trans_checkpoint, map_location="cpu")
    data_config, train_config, model_config = load_config(trans_config)

    text_sp = sp.SentencePieceProcessor(model_file=data_config['text_sp'])
    text_bos, text_eos, text_pad_id = text_sp.piece_to_id(['<s>', '</s>', '<pad>'])

    num_vocab = text_sp.piece_size()
    f = open(tit_path, "w")

    codebook_size = model_config["codebook_size"]
    num_code = codebook_size + 2
    code_d_model = model_config["code_d_model"]
    code_d_ff = model_config["code_d_ff"]
    code_n_head = model_config["code_n_head"]
    code_l = model_config["code_l"]

    text_d_model = model_config["text_d_model"]
    text_d_ff = model_config["text_d_ff"]
    text_n_head = model_config["text_n_head"]
    text_l = model_config["text_l"]
    dropout = model_config["dropout"]

    translation = AuxTITTransformer(num_vocab, num_code, text_d_model, code_d_model, text_d_ff, code_d_ff, text_n_head, code_n_head, text_l, code_l, text_pad_id, dropout)
    translation.load_state_dict(translation_ckpt["model_state"])
    translation.cuda().eval()

    ds = TestDataset(input_img_dir)
    dl = DataLoader(ds, batch, shuffle=False)
    import time
    start = time.time()
    with torch.no_grad():
        idx = 1
        for img in dl:
            codebook_output_dict = codebook(img.cuda())
            src_code = codebook_output_dict["code"]
            trans_output_dict = translation.inference_code(src_code, 8192, 96, text_eos, text_bos, text_pad_id, 64)
            tgt_code = trans_output_dict["code"]
            tgt_img = codebook.inference_img_with_code(tgt_code)["img"]
            tgt_text = trans_output_dict["text"]
            for img, text in zip(tgt_img, tgt_text):
                new_img = img * 0.5 + 0.5
                vutils.save_image(new_img, os.path.join(output_img_dir, str(idx)+".jpg"))
                idx += 1
                f.write(text_sp.decode(text.tolist())+"\n")
    end = time.time()
    print(end-start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--trans_config", type=str, required=True)
    parser.add_argument("--trans_checkpoint", type=str, required=True)
    parser.add_argument("--codebook_config", type=str, required=True)
    parser.add_argument("--codebook_checkpoint", type=str, required=True)
    parser.add_argument("--input_img_dir", type=str, required=True)
    parser.add_argument("--output_img_dir", type=str, required=True)
    parser.add_argument("--tit_path", type=str, required=True)
    args = parser.parse_args()

    batch = args.batch
    trans_config = args.trans_config
    trans_checkpoint = args.trans_checkpoint
    codebook_config = args.codebook_config
    codebook_checkpoint = args.codebook_checkpoint
    input_img_dir = args.input_img_dir
    output_img_dir = args.output_img_dir
    tit_path = args.tit_path

    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    decode(batch, trans_config, trans_checkpoint, codebook_config, codebook_checkpoint, input_img_dir, output_img_dir, tit_path)
