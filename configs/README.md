## 📝 Notes on Multi-Font Training

> 💡 **Note**: If you plan to train a model that **supports various fonts**, please **merge** the following components from each training set (e.g., *Times New Roman*, *Arial*, *Calibri*) into unified folders:
>
> - Source images  
> - Target images  
> - Source text images  
> - Target text images  
> - Background images

---

## ⚙️ Configuration File Descriptions

### `config-separate.json`

- `"train_src_img_path"` / `"train_tgt_img_path"`:  
  Path to the folders containing **source** / **target** images.

- `"train_src_text_img_path"` / `"train_tgt_text_img_path"`:  
  Path to the **source** / **target** text images.

- `"train_background_img_path"`:  
  Path to the **background** images.

- `"load_checkpoint"`:  
  Path to a previously saved checkpoint if resuming training.

- `"save_checkpoint_dir"`:  
  Directory where model checkpoints will be saved during training.

---

### `config-codebook.json`

- Configuration is **identical** to `config-separate.json`.

---

### `config-translation.json`

- `"train_src_code_path"` / `"train_tgt_code_path"`:  
  Paths to the decoded **code sequence files** generated from text images using  
  `DebackX/scripts/decode-codebook.sh`.

- `"train_text_path"`:  
  Path to **tokenized text ID files**, which include tokenized IDs of the target texts.

- `"text_sp"`:  
  Path to the **SentencePiece model** used for tokenization.

- `"load_pretrain"` *(optional)*:  
  If you're fine-tuning on IIMT30k after pre-training with a large-scale synthetic text-image dataset, specify the path to the pre-trained checkpoint here.

---

### `config-fuse.json`

- Configuration is **identical** to `config-separate.json`.

