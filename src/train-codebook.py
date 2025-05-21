from Model import Codebook
from LoadData import MonoTextImageDataset, get_dataloader
import torch
from torch import nn
from accelerate import Accelerator
from einops import rearrange
import numpy as np
import datetime
import os
from torch.utils.tensorboard.writer import SummaryWriter
import json
import argparse
from accelerate.utils import set_seed
import lpips


class Constant_lr_schedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr):
        self.steps = 0
        self.lr = lr
    
    def step(self):
        self.steps += 1

class warmup_lr_schedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_step=0):
        self.steps = 0
        self.d_model = d_model
        self.warmup_step = warmup_step
        self.optimizer = optimizer
        self.lr = 0.0
    
    def step(self):
        self.steps += 1
        self.lr = self.d_model**-0.5 * min(self.steps**-0.5, self.steps * self.warmup_step**-1.5)
        self.optimizer.param_groups[0]['lr'] = self.lr

class inverse_sqrt_lr_schedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_step, warmup_init_lr, max_lr):
        self.steps = 0
        self.optimizer = optimizer
        self.warmup_step = warmup_step
        self.warmup_init_lr = warmup_init_lr
        self.max_lr = max_lr
        self.lrs = torch.linspace(self.warmup_init_lr, self.max_lr, self.warmup_step)
        self.lr = 0.0
    
    def step(self):
        self.steps += 1
        if self.steps < self.warmup_step:
            self.lr = self.lrs[self.steps]
        else:
            decay_factor = self.max_lr * self.warmup_step**0.5
            self.lr = decay_factor * self.steps**-0.5
        self.optimizer.param_groups[0]['lr'] = self.lr


def save_model(model, optimizer, lr_schedule, accelerator, path, type):
    unwrap_model = accelerator.unwrap_model(model)
    unwrap_optim = accelerator.unwrap_model(optimizer)
    torch.save({
        "model_state" : unwrap_model.state_dict(),
        "optim_state" : unwrap_optim.state_dict(),
        "lr_state" : lr_schedule.state_dict()
        }, os.path.join(path, "checkpoint"+type+".pt"))
    accelerator.print("save checkpoint at {}".format(os.path.join(path, "checkpoint"+type+".pt")))

def keep_top_models(save_path, metric_suffix, num=5):
    model_files = [f for f in os.listdir(save_path) if metric_suffix in f]
    metric_values = [float(f.replace(".pt", "").split(metric_suffix.format(''))[1]) for f in model_files if metric_suffix.format('') in f]

    sorted_indices = sorted(range(len(metric_values)), key=lambda k: metric_values[k])
    models_to_keep = sorted_indices[:num]
    for i, model_file in enumerate(model_files):
        if i not in models_to_keep:
            os.remove(os.path.join(save_path, model_file))

def keep_last_models(save_path, metric_suffix, num=5):
    model_files = [f for f in os.listdir(save_path) if metric_suffix in f]
    metric_values = [float(f.replace(".pt", "").split(metric_suffix.format(''))[1]) for f in model_files if metric_suffix.format('') in f]

    sorted_indices = sorted(range(len(metric_values)), key=lambda k: metric_values[k], reverse=True)
    models_to_keep = sorted_indices[:num]
    for i, model_file in enumerate(model_files):
        if i not in models_to_keep:
            os.remove(os.path.join(save_path, model_file))

def train(step, model, train_dl, valid_dl, optimizer, lr_schedule, loss_func, accelerator, writer, save_path):
    model.train()
    each_loss = []
    for img, label_text_img in train_dl:
        output_dict = model(img)
        output_img = output_dict["img"]
        output_vqloss = output_dict["vqloss"]
        l2 = (output_img - label_text_img).pow(2).mean()
        lpip = loss_func["lpip"](output_img, label_text_img).mean()
        loss = l2 + 0.1*lpip + output_vqloss

        accelerator.backward(loss)
        
        optimizer.step()
        lr_schedule.step()
        optimizer.zero_grad()

        step += 1
        if accelerator.is_local_main_process:
            each_loss.append(loss.item())
            writer.add_scalar(
                tag="train/l2", scalar_value=l2.item(), global_step=step
            )
            writer.add_scalar(
                tag="train/lpip", scalar_value=lpip.item(), global_step=step
            )
            writer.add_scalar(
                tag="train/vqloss", scalar_value=output_vqloss.item(), global_step=step
            )
            writer.add_scalar(
                tag="train/loss", scalar_value=loss.item(), global_step=step
            )
            writer.add_scalar(
                tag="train/train_lr", scalar_value=lr_schedule.lr, global_step=step
            )
        if step % 100 == 0 and step != 0:
            accelerator.print("{} step {}: loss={}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, np.mean(each_loss[-100: ])))
        if step % 1000 == 0 and step != 0:
            valid_metric = valid(step, model, valid_dl, loss_func, accelerator, writer)
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                save_model(model, optimizer, lr_schedule, accelerator, save_path, "_best{:.3f}".format(valid_metric))
                keep_top_models(save_path, "_best", 5)
                save_model(model, optimizer, lr_schedule, accelerator, save_path, "_last{}".format(step))
                keep_last_models(save_path, "_last", 5)
    accelerator.print("{} epoch end, loss={}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), np.mean(each_loss)))
    return step

def valid(step, model, valid_dl, loss_func, accelerator, writer):
    model.eval()
    each_l2 = []
    each_lpip = []
    each_vqloss = []
    each_loss = []
    with torch.no_grad():
        for img, label_text_img in valid_dl:
            output_dict = model(img)
            output_img = output_dict["img"]
            output_vqloss = output_dict["vqloss"]
            l2 = (output_img - label_text_img).pow(2).mean()
            lpip = loss_func["lpip"](output_img, label_text_img).mean()
            loss = l2 + 0.1*lpip + output_vqloss

            l2 = accelerator.gather_for_metrics(l2).mean().item()
            lpip = accelerator.gather_for_metrics(lpip).mean().item()
            vqloss = accelerator.gather_for_metrics(output_vqloss).mean().item()
            loss = accelerator.gather_for_metrics(loss).mean().item()

            each_l2.append(l2)
            each_lpip.append(lpip)
            each_vqloss.append(vqloss)
            each_loss.append(loss)

        eval_metric = np.mean(each_loss)
        
        if accelerator.is_local_main_process:
            writer.add_scalar(
                tag="valid/l2", scalar_value=np.mean(each_l2), global_step=step
            )
            writer.add_scalar(
                tag="valid/lpip", scalar_value=np.mean(each_lpip), global_step=step
            )
            writer.add_scalar(
                tag="valid/vqloss", scalar_value=np.mean(each_vqloss), global_step=step
            )
            writer.add_scalar(
                tag="valid/loss", scalar_value=np.mean(each_loss), global_step=step
            )
        accelerator.print("{} valid loss={}, metric={}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), np.mean(each_loss), eval_metric))
    model.train()
    return eval_metric

def load_config(config_path):
    json_file = open(config_path)
    json_dict = json.load(json_file)
    data = json_dict["data"]
    train = json_dict["train"]
    model = json_dict["model"]
    return data, train, model

def train_loop(config_path):
    accelerator = Accelerator(mixed_precision="fp16")
    data_config, train_config, model_config = load_config(config_path)

    patch_size = model_config["patch_size"]
    dim = model_config["dim"]
    codebook_dim = model_config["codebook_dim"]
    codebook_size = model_config["codebook_size"]

    batch_size = train_config["batch_size"]
    epoch = train_config["epoch"]
    max_update_step = train_config["max_update_step"]
    warmup_steps = train_config["warmup_steps"]
    warmup_init_lr = train_config["warmup_init_lr"] 
    max_lr = train_config["max_lr"]

    train_src_text_img_path  = data_config["train_src_text_img_path"]
    train_tgt_text_img_path  = data_config["train_tgt_text_img_path"]

    valid_src_text_img_path  = data_config["valid_src_text_img_path"]
    valid_tgt_text_img_path  = data_config["valid_tgt_text_img_path"]

    writer = None
    if accelerator.is_local_main_process:
        writer = SummaryWriter(log_dir=os.path.join(train_config["tensorboard_dir"], datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        writer.add_text("data_config", str(data_config))
        writer.add_text("train_config", str(train_config))
        writer.add_text("model_config", str(model_config))
    
    save_path = train_config["save_checkpoint_dir"]
    if accelerator.is_local_main_process:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        os.system("cp {} {}".format(config_path, save_path))
    
    model = Codebook(patch_size, dim, codebook_dim, codebook_size)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9, foreach=False)
    lr_schedule = inverse_sqrt_lr_schedule(optimizer, warmup_steps, warmup_init_lr, max_lr)

    loss_func = {"lpip": lpips.LPIPS(net="vgg", verbose=False).to(accelerator.device)}
    if os.path.exists(train_config["load_checkpoint"]):
        accelerator.print("load checkpoint at {}".format(train_config["load_checkpoint"]))
        ckpt = torch.load(train_config["load_checkpoint"], map_location=accelerator.device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        lr_schedule.load_state_dict(ckpt["lr_state"])
        
    accelerator.print(data_config)
    accelerator.print(train_config)
    accelerator.print(model_config)
    accelerator.print(model)

    train_ds = MonoTextImageDataset(train_src_text_img_path, train_tgt_text_img_path)
    valid_ds = MonoTextImageDataset(valid_src_text_img_path, valid_tgt_text_img_path)

    train_dl = get_dataloader(train_ds, batch_size, True)
    valid_dl = get_dataloader(valid_ds, batch_size//2, False)

    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl
    )
    step = lr_schedule.steps
    accelerator.wait_for_everyone()
    e = 0
    while(True):
        e += 1
        accelerator.print("{} epoch {} start".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), e))
        train_step = train(step, model, train_dl, valid_dl, optimizer, lr_schedule, loss_func, accelerator, writer, save_path)
        step = train_step
        if step >= max_update_step:
            break
        if epoch != -1 and e >= epoch:
            break
    accelerator.print("{} training end".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    accelerator.end_training()


if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()
    config = args.config
    train_loop(config)
