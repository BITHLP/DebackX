import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from PIL import Image


class TextImageBackgroundDataset(Dataset):
    # train encoder&decoder
    def __init__(self, src_img_path, tgt_img_path, background_img_path, src_text_img_path, tgt_text_img_path):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000])
        ])

        self.src_img_list = sorted(os.listdir(src_img_path), key=lambda x:int(x.split(".")[0]))
        self.src_img_list = [os.path.join(src_img_path, i) for i in self.src_img_list]

        self.tgt_img_list = sorted(os.listdir(tgt_img_path), key=lambda x:int(x.split(".")[0]))
        self.tgt_img_list = [os.path.join(tgt_img_path, i) for i in self.tgt_img_list]

        self.total_img_list = self.src_img_list + self.tgt_img_list

        self.background_img_list = sorted(os.listdir(background_img_path), key=lambda x:int(x.split(".")[0]))
        self.background_img_list = [os.path.join(background_img_path, i) for i in self.background_img_list]

        self.total_background_list = self.background_img_list + self.background_img_list

        self.src_text_img_list = sorted(os.listdir(src_text_img_path), key=lambda x:int(x.split(".")[0]))
        self.src_text_img_list = [os.path.join(src_text_img_path, i) for i in self.src_text_img_list]

        self.tgt_text_img_list = sorted(os.listdir(tgt_text_img_path), key=lambda x:int(x.split(".")[0]))
        self.tgt_text_img_list = [os.path.join(tgt_text_img_path, i) for i in self.tgt_text_img_list]

        self.total_text_img_list = self.src_text_img_list + self.tgt_text_img_list

    def __getitem__(self, index):
        # img, back, text_img
        img = Image.open(self.total_img_list[index]).convert("RGB")
        background = Image.open(self.total_background_list[index]).convert("RGB")
        text_img = Image.open(self.total_text_img_list[index]).convert("RGB")
        return self.transform(img), self.transform(background), self.transform(text_img)

    def __len__(self):
        return len(self.total_background_list)


class MonoTextImageDataset(Dataset): 
    # train codebook
    def __init__(self, src_text_img_path, tgt_text_img_path):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000])
        ])
        self.src_text_img_list = sorted(os.listdir(src_text_img_path), key=lambda x:int(x.split(".")[0]))
        self.src_text_img_list = [os.path.join(src_text_img_path, i) for i in self.src_text_img_list]

        self.tgt_text_img_list = sorted(os.listdir(tgt_text_img_path), key=lambda x:int(x.split(".")[0]))
        self.tgt_text_img_list = [os.path.join(tgt_text_img_path, i) for i in self.tgt_text_img_list]

        self.total_text_img_list = self.src_text_img_list + self.tgt_text_img_list

    def __getitem__(self, index):
        # textimage -> textimage
        text_img = Image.open(self.total_text_img_list[index]).convert("RGB")
        return self.transform(text_img), self.transform(text_img)

    def __len__(self):
        return len(self.total_text_img_list)


class TextAuxParalCodeDataset(Dataset):
    # train translation with aux text
    def __init__(self, src_code_path, tgt_code_path, text_path, code_bos, code_eos, text_pad_id, text_max_length=64):
        src_code_list = []
        src_code_file = open(src_code_path, "r")
        for l in src_code_file:
            src_code_list.append(list(map(int, l.strip().split())))

        tgt_code_list = []
        tgt_code_file = open(tgt_code_path, "r")
        for l in tgt_code_file:
            tgt_code_list.append(list(map(int, l.strip().split())))

        text_list = []
        tgt_idx_file = open(text_path, "r")
        for l in tgt_idx_file:
            text_list.append(list(map(int, l.strip().split())))

        self.code_bos = torch.tensor([code_bos])
        self.code_eos = torch.tensor([code_eos])
        self.text_pad_id = text_pad_id
        self.text_max_length = text_max_length

        filter_text_list = []
        self.filter_src_code_list = []
        self.filter_tgt_code_list = []
        for text, scode, tcode in zip(text_list, src_code_list, tgt_code_list):
            if len(text) < text_max_length:
                filter_text_list.append(text)
                self.filter_src_code_list.append(scode)
                self.filter_tgt_code_list.append(tcode)

        rm_eos_list = []
        rm_bos_list = []
        for t in filter_text_list:
            rm_eos_list.append(t[:-1])
            rm_bos_list.append(t[1:])
        self.tgt_text = []
        self.label_text = []
        for rm_eos in rm_eos_list:
            self.tgt_text.append(rm_eos + [self.text_pad_id] * (self.text_max_length - len(rm_eos)))
        for rm_bos in rm_bos_list:
            self.label_text.append(rm_bos + [self.text_pad_id] * (self.text_max_length - len(rm_bos)))

    def __getitem__(self, index):
        # src_code, tgt_code, label_code, tgt_text, label_text
        src_code = torch.tensor(self.filter_src_code_list[index])

        tgt_code = torch.concat([self.code_bos, torch.tensor(self.filter_tgt_code_list[index])])
        label_code = torch.concat([torch.tensor(self.filter_tgt_code_list[index]), self.code_eos])

        return src_code, tgt_code, label_code, torch.tensor(self.tgt_text[index]), torch.tensor(self.label_text[index])

    def __len__(self):
        return len(self.filter_src_code_list)


def get_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size, shuffle, num_workers=4)
