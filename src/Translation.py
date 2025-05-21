import torch.nn as nn
import torch
import timm


class AuxTITTransformer(nn.Module):
    def __init__(self, num_vocab, num_code, text_d_model, code_d_model, text_d_ff, code_d_ff, text_n_head, code_n_head, text_l, code_l, text_pad_id, dropout=0.1, causal=True):
        super().__init__()
        self.src_code_embedding = Embedding(num_code, text_d_model, dropout)
        self.tgt_code_embedding = Embedding(num_code, code_d_model, dropout)
        self.text_embedding = Embedding(num_vocab, text_d_model, dropout)
        # self.code_encoder = Encoder(d_model, code_d_ff, code_n_head, code_l, dropout)
        self.code_encoder = Encoder(text_d_model, text_d_ff, text_n_head, text_l, dropout)
        self.code_decoder = Decoder(code_d_model, code_d_ff, code_n_head, code_l, dropout, causal)
        self.text_decoder = Decoder(text_d_model, text_d_ff, text_n_head, text_l, dropout, causal)
        self.code_proj = OutputLayer(code_d_model, num_code)
        self.text_proj = OutputLayer(text_d_model, num_vocab)
        self.adapter = nn.Linear(text_d_model, code_d_model)
        self.text_pad_id = text_pad_id
        self.init_params()

    def init_params(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, y, text):
        src_code_embed = self.src_code_embedding(x)
        tgt_code_embed = self.tgt_code_embedding(y)
        code_encoder_hidden = self.code_encoder(src_code_embed)

        text_embed = self.text_embedding(text)
        text_padding_mask = (text == self.text_pad_id)
        text_decoder_hidden = self.text_decoder(code_encoder_hidden, text_embed, y_padding_mask=text_padding_mask)
        text_output = self.text_proj(text_decoder_hidden)

        text_decoder_hidden_input = self.adapter(text_decoder_hidden)
        
        code_decoder_hidden = self.code_decoder(text_decoder_hidden_input, tgt_code_embed, x_padding_mask=text_padding_mask)
        code_output = self.code_proj(code_decoder_hidden)

        return {"code": code_output, "text": text_output}

    @torch.no_grad()
    def inference_code(self, x, code_bos, code_max_length, text_eos, text_bos, text_pad, text_max_length):
        batch = x.shape[0]
        y = torch.full((batch, 1), code_bos, dtype=torch.long, device=x.device)
        
        tgt_text_tensor = self.inference_text(x, text_eos, text_bos, text_pad, text_max_length)
        # src_embed = self.code_embedding(x)
        src_embed = self.src_code_embedding(x)
        encoder_hidden = self.code_encoder(src_embed)
        text_padding_mask = (tgt_text_tensor == self.text_pad_id)
        text_embed = self.text_embedding(tgt_text_tensor)
        text_hidden = self.text_decoder(encoder_hidden, text_embed, y_padding_mask=text_padding_mask)
        text_hidden = self.adapter(text_hidden)
        for _ in range(code_max_length):
            # tgt_embed = self.code_embedding(y)
            tgt_embed = self.tgt_code_embedding(y)
            decoder_hidden = self.code_decoder(text_hidden, tgt_embed, x_padding_mask=text_padding_mask)
            output = self.code_proj(decoder_hidden) # batch, seq, vocab
            logits = output[:, -1, :] # batch, vocab
            next_tokens = torch.argmax(logits, -1)
            y = torch.cat([y, next_tokens.unsqueeze(1)], dim=-1)
        return {"code": y[:, 1:], "text": tgt_text_tensor}

    @torch.no_grad()
    def inference_text(self, x, eos_id, bos_id, pad_id, max_length):
        batch = x.shape[0]
        y_ids = torch.tensor([[bos_id] for _ in range(batch)], device=x.device)
        complete_idx = {}
        ret = []
        src_embed = self.src_code_embedding(x)
        encoder_hidden = self.code_encoder(src_embed)

        for step in range(max_length):
            with torch.no_grad():
                tgt_embed = self.text_embedding(y_ids)
                text_padding_mask = (y_ids == pad_id)
                decoder_hidden = self.text_decoder(encoder_hidden, tgt_embed, y_padding_mask=text_padding_mask)
                output = self.text_proj(decoder_hidden) # batch, seq, vocab
                logits = output[:, -1, :] # batch, vocab
                step_out = torch.argmax(logits, -1)

                idx = 0
                for each_step_out in step_out:
                    if each_step_out == eos_id:
                        if complete_idx.get(idx) is None:
                            complete_idx[idx] = y_ids[idx]
                    if complete_idx.get(idx) is not None:
                        step_out[idx] = pad_id    
                    idx += 1
                
                y_ids = torch.concat((y_ids, step_out.reshape(-1, 1)), dim=-1)

                if len(complete_idx) == batch:
                    break
        # breakpoint()
        # for i in range(batch):
        #     if complete_idx.get(i) is None:
        #         complete_idx[i] = y_ids[i]
        #     ret.append(complete_idx[i].tolist())
        # return ret
        return y_ids


class Embedding(nn.Module):
    def __init__(self, num_vocab, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding_layer = nn.Embedding(num_vocab, d_model)
        self.pe = self.position_embedding(512, d_model)
        self.register_buffer("Positional Embedding", self.pe)
        self.dropout = nn.Dropout(p=dropout)
    
    def position_embedding(self, max_len, d_model):  
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2) * (torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        """
        x: B x S
        """
        embs = self.embedding_layer(x) * torch.sqrt(torch.tensor(self.d_model))
        embs_pe = self.pe[:x.size()[1], :]
        return self.dropout(embs + embs_pe.to(embs.device))


class OutputLayer(nn.Module):
    def __init__(self, d_model, num_vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, num_vocab)

    def forward(self, x):
        return self.proj(x)


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, n_head, l, dropout=0.1, causal=True):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_head = n_head
        self.l = l
        self.p = dropout
        self.causal = causal
        self.decoder_layer = self.init_decoder()

    def init_decoder(self):
        each_layer = nn.TransformerDecoderLayer(self.d_model, self.n_head, self.d_ff, dropout=self.p, batch_first=True)
        layers = nn.TransformerDecoder(each_layer, self.l)
        return layers
    
    def forward(self, x, y, x_padding_mask=None, y_padding_mask=None):
        """"
        y: B x S x D
        x: hidden from source
        """
        if self.causal:
            attn_mask = (nn.Transformer.generate_square_subsequent_mask(y.size()[1]) == -torch.inf).to(y.device)
        else:
            attn_mask = None
        hidden = self.decoder_layer(tgt=y, memory=x, tgt_mask=attn_mask, tgt_key_padding_mask=y_padding_mask, memory_key_padding_mask=x_padding_mask)
        return hidden


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, n_head, l, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_head = n_head
        self.l = l
        self.p = dropout
        self.encoder_layer = self.init_encoder()
    
    def init_encoder(self):
        each_layer = nn.TransformerEncoderLayer(self.d_model, self.n_head, self.d_ff, dropout=self.p, batch_first=True)
        layers = nn.TransformerEncoder(each_layer, self.l)
        return layers
    
    def forward(self, x, padding_mask=None):
        """
        x: B x S x D
        """
        hidden = self.encoder_layer(src=x, src_key_padding_mask=padding_mask)
        return hidden


class TiMMViTEncoder(timm.models.VisionTransformer):
    def __init__(self, dim=512, depth=8, heads=8, patch=16):
        self.encoder_dim = dim
        self.encoder_depth = depth
        self.encoder_heads = heads
        self.encoder_patch_size = patch
        self.img_size = (48, 512)
        super().__init__(img_size=self.img_size, patch_size=self.encoder_patch_size, embed_dim=self.encoder_dim, depth=self.encoder_depth, num_heads=self.encoder_heads, num_classes=0, class_token=False, global_pool="")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: img tensor
        """
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x