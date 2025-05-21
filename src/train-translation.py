from Translation import AuxTITTransformer
from LoadData import TextAuxParalCodeDataset, get_dataloader
import torch
from torch import nn
import accelerate
from accelerate import Accelerator
from einops import rearrange
import numpy as np
import datetime
import os
from torch.utils.tensorboard.writer import SummaryWriter
import json
import argparse
from accelerate.utils import set_seed
import sentencepiece as sp


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

    sorted_indices = sorted(range(len(metric_values)), key=lambda k: metric_values[k], reverse=True)
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

def train(step, model, train_dl, valid_dl, optimizer, lr_schedule, loss_func, accelerator, writer, save_path, text_pad_id):
    model.train()
    each_loss = []
    for src_code, tgt_code, label_code, tgt_text, label_text in train_dl:
        # valid_metric = valid(step, model, valid_dl, loss_func, accelerator, writer, text_pad_id)
        output_dict = model(src_code, tgt_code, tgt_text)
        output_code = output_dict["code"]
        output_text = output_dict["text"]
        code_loss = loss_func["code"](rearrange(output_code, 'b s c -> (b s) c'), rearrange(label_code, 'b s -> (b s)'))
        text_loss = loss_func["text"](rearrange(output_text, 'b s c -> (b s) c'), rearrange(label_text, 'b s -> (b s)'))
        loss = code_loss + text_loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_schedule.step()
        optimizer.zero_grad()

        step += 1
        if accelerator.is_local_main_process:
            each_loss.append(loss.item())
            writer.add_scalar(
                tag="train/loss", scalar_value=loss.item(), global_step=step
            )
            writer.add_scalar(
                tag="train/code_loss", scalar_value=code_loss.item(), global_step=step
            )
            writer.add_scalar(
                tag="train/text_loss", scalar_value=text_loss.item(), global_step=step
            )
            writer.add_scalar(
                tag="train/train_lr", scalar_value=lr_schedule.lr, global_step=step
            )
        if step % 100 == 0 and step != 0:
            accelerator.print("{} step {}: loss={}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, np.mean(each_loss[-100: ])))
        if step % 1000 == 0 and step != 0:
            valid_metric = valid(step, model, valid_dl, loss_func, accelerator, writer, text_pad_id)
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                save_model(model, optimizer, lr_schedule, accelerator, save_path, "_best{:.3f}".format(valid_metric))
                keep_top_models(save_path, "_best", 5)
                save_model(model, optimizer, lr_schedule, accelerator, save_path, "_last{}".format(step))
                keep_last_models(save_path, "_last", 5)
    accelerator.print("{} epoch end, loss={}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), np.mean(each_loss)))
    return step

def valid(step, model, valid_dl, loss_func, accelerator, writer, text_pad_id):
    model.eval()
    each_loss = []
    each_code_loss = []
    each_text_loss = []
    total_code_num = 0.
    acc_code_num = 0.
    total_text_num = 0.
    acc_text_num = 0.
    pad_text_num = 0.
    with torch.no_grad():
        for src_code, tgt_code, label_code, tgt_text, label_text in valid_dl:
            output_dict = model(src_code, tgt_code, tgt_text)
            output_code = output_dict["code"]
            output_text = output_dict["text"]
            code_loss = loss_func["code"](rearrange(output_code, 'b s c -> (b s) c'), rearrange(label_code, 'b s -> (b s)'))
            text_loss = loss_func["text"](rearrange(output_text, 'b s c -> (b s) c'), rearrange(label_text, 'b s -> (b s)'))
            loss = code_loss + text_loss

            code_loss = accelerator.gather_for_metrics(code_loss).mean().item()
            text_loss = accelerator.gather_for_metrics(text_loss).mean().item()
            loss = accelerator.gather_for_metrics(loss).mean().item()

            output_code = accelerator.gather_for_metrics(output_code)
            label_code = accelerator.gather_for_metrics(label_code) 
            output_text = accelerator.gather_for_metrics(output_text)
            label_text = accelerator.gather_for_metrics(label_text)

            each_loss.append(loss)
            each_code_loss.append(code_loss)
            each_text_loss.append(text_loss)
            # calculate code acc
            acc_code_num += (output_code.argmax(-1) == label_code).sum()
            total_code_num += label_code.numel()

            # calculate text acc
            accurate_preds = (output_text.argmax(-1) == label_text)
            total_text_num += label_text.numel()
            pad_mask = (label_text == text_pad_id)
            accurate_preds.masked_fill_(pad_mask, False)
            pad_text_num += (label_text == text_pad_id).sum().item()
            acc_text_num += accurate_preds.sum().item()

        acc_code = acc_code_num / total_code_num
        acc_text = acc_text_num / (total_text_num - pad_text_num)
        eval_metric = acc_code + acc_text
        
        if accelerator.is_local_main_process:
            writer.add_scalar(
                tag="valid/acc_code", scalar_value=acc_code, global_step=step
            )
            writer.add_scalar(
                tag="valid/acc_text", scalar_value=acc_text, global_step=step
            )
            writer.add_scalar(
                tag="valid/loss", scalar_value=np.mean(each_loss), global_step=step
            )
            writer.add_scalar(
                tag="valid/code_loss", scalar_value=np.mean(each_code_loss), global_step=step
            )
            writer.add_scalar(
                tag="valid/text_loss", scalar_value=np.mean(each_text_loss), global_step=step
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

    text_sp = sp.SentencePieceProcessor(model_file=data_config['text_sp'])
    text_bos, text_eos, text_pad_id = text_sp.piece_to_id(['<s>', '</s>', '<pad>'])

    num_vocab = text_sp.piece_size()

    codebook_size = model_config["codebook_size"]
    num_code = codebook_size + 2
    # d_model = model_config["d_model"]
    code_d_model = model_config["code_d_model"]
    code_d_ff = model_config["code_d_ff"]
    code_n_head = model_config["code_n_head"]
    code_l = model_config["code_l"]

    text_d_model = model_config["text_d_model"]
    text_d_ff = model_config["text_d_ff"]
    text_n_head = model_config["text_n_head"]
    text_l = model_config["text_l"]
    dropout = model_config["dropout"]

    batch_size = train_config["batch_size"]
    epoch = train_config["epoch"]
    max_update_step = train_config["max_update_step"]
    warmup_steps = train_config["warmup_steps"]
    warmup_init_lr = train_config["warmup_init_lr"] 
    max_lr = train_config["max_lr"]

    train_src_code_path = data_config["train_src_code_path"]
    train_tgt_code_path = data_config["train_tgt_code_path"]
    train_text_path = data_config["train_text_path"]
    valid_src_code_path = data_config["valid_src_code_path"]
    valid_tgt_code_path = data_config["valid_tgt_code_path"]
    valid_text_path = data_config["valid_text_path"]
    code_bos = codebook_size
    code_eos = codebook_size + 1

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
    
    model = AuxTITTransformer(num_vocab, num_code, text_d_model, code_d_model, text_d_ff, code_d_ff, text_n_head, code_n_head, text_l, code_l, text_pad_id, dropout)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9, foreach=False)
    lr_schedule = inverse_sqrt_lr_schedule(optimizer, warmup_steps, warmup_init_lr, max_lr)

    if train_config.get("lr_schedule") is not None:
        if train_config["lr_schedule"] == "constant":
            lr_schedule = Constant_lr_schedule(optimizer, train_config["lr"])

    loss_func = {"code": nn.CrossEntropyLoss(label_smoothing=0.1), "text": nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=text_pad_id)}
    if os.path.exists(train_config["load_checkpoint"]):
        accelerator.print("load checkpoint at {}".format(train_config["load_checkpoint"]))
        ckpt = torch.load(train_config["load_checkpoint"], map_location=accelerator.device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        lr_schedule.load_state_dict(ckpt["lr_state"])

    if os.path.exists(train_config["load_pretrain"]):
        accelerator.print("load pretrain model at {}".format(train_config["load_pretrain"]))
        ckpt = torch.load(train_config["load_pretrain"], map_location=accelerator.device)
        model.load_state_dict(ckpt["model_state"])

    accelerator.print(data_config)
    accelerator.print(train_config)
    accelerator.print(model_config)
    accelerator.print(model)

    train_ds = TextAuxParalCodeDataset(train_src_code_path, train_tgt_code_path, train_text_path, code_bos, code_eos, text_pad_id)
    valid_ds = TextAuxParalCodeDataset(valid_src_code_path, valid_tgt_code_path, valid_text_path, code_bos, code_eos, text_pad_id)

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
        train_step = train(step, model, train_dl, valid_dl, optimizer, lr_schedule, loss_func, accelerator, writer, save_path, text_pad_id)
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
