from torch import nn
import torch
import timm
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
from einops.layers.torch import Rearrange


class FuseDecoder(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.back_encoder = TiMMViTEncoder(patch=patch_size)
        self.text_encoder = TiMMViTEncoder(patch=patch_size)
        self.img_decoder = TiMMViTDecoder(patch=patch_size)
    
    def forward(self, back, text):
        back_hidden = self.back_encoder(back)
        text_hidden = self.text_encoder(text)
        fuse_hidden = back_hidden + text_hidden
        rec_img = self.img_decoder(fuse_hidden)
        return {"img": rec_img}


class Codebook(nn.Module):
    def __init__(self, patch_size, dim, codebook_dim, codebook_size):
        super().__init__()
        self.encoder = TiMMViTEncoder(patch=patch_size)
        self.pre_quant = nn.Linear(dim, codebook_dim)
        self.vq = VectorQuantize(dim=codebook_dim, codebook_size=codebook_size)
        self.post_quant = nn.Linear(codebook_dim, dim)
        self.decoder = TiMMViTDecoder(patch=patch_size)

    def forward(self, img):
        enc_hidden = self.encoder(img)
        pre_q = self.pre_quant(enc_hidden)
        quant, code, loss = self.vq(pre_q)
        post_q = self.post_quant(quant)
        rec_img = self.decoder(post_q)

        return {"img": rec_img, "vqloss": loss, "code": code}
    
    @torch.no_grad()
    def inference_img_with_code(self, code):
        quant = self.vq.get_codes_from_indices(code)
        post_q = self.post_quant(quant)
        rec_img = self.decoder(post_q)

        return {"img": rec_img}


class SeparateEncoder(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.back_encoder = TiMMViTEncoder(patch=patch_size)
        self.back_decoder = TiMMViTDecoder(patch=patch_size)
        self.text_encoder = TiMMViTEncoder(patch=patch_size)
        self.text_decoder = TiMMViTDecoder(patch=patch_size)
    
    def forward(self, img):
        back_hidden = self.back_encoder(img)
        rec_back = self.back_decoder(back_hidden)
        text_hidden = self.text_encoder(img)
        rec_text = self.text_decoder(text_hidden)

        return {"back_img": rec_back, "text_img": rec_text}


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


class TiMMViTDecoder(timm.models.VisionTransformer):
    def __init__(self, dim=512, depth=8, heads=8, patch=16):
        self.decoder_dim = dim
        self.decoder_depth = depth
        self.decoder_heads = heads
        self.decoder_patch_size = patch
        self.img_size = (48, 512)
        super().__init__(img_size=self.img_size, patch_size=self.decoder_patch_size, embed_dim=self.decoder_dim, depth=self.decoder_depth, num_heads=self.decoder_heads, num_classes=0, class_token=False, global_pool="")
        self.to_pixel = nn.Sequential(
            Rearrange("b (h w) c -> b c h w", h=self.img_size[0]//self.decoder_patch_size, w=self.img_size[1]//self.decoder_patch_size),
            nn.ConvTranspose2d(self.decoder_dim, 3, kernel_size=self.decoder_patch_size, stride=self.decoder_patch_size)
        )
        self.patch_embed = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: hidden tensor
        """
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.to_pixel(x)
        return x


class Embedding(nn.Module):
    def __init__(self, num_vocab, d_model, padding_idx, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding_layer = nn.Embedding(num_vocab, d_model, padding_idx)
        # nn.init.normal_(self.embedding_layer.weight, mean=0, std=d_model ** -0.5)
        self.pe = self.position_embedding(800, d_model)
        self.register_buffer("Positional Embedding", self.pe)
        self.dropout = nn.Dropout(p=dropout)
        self.init_params()

    def init_params(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # print("Init params {} with xavier_uniform".format(name))
                nn.init.xavier_uniform_(p)
    
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
        self.init_params()

    def init_params(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # print("Init params {} with xavier_uniform".format(name))
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        return self.proj(x)


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, n_head, l, dropout=0.1):
        """
        dropout; pre/post norm
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_head = n_head
        self.l = l
        self.p = dropout
        self.encoder_layer = self.init_encoder()
        self.init_params()

    def init_params(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # print("Init params {} with xavier_uniform".format(name))
                nn.init.xavier_uniform_(p)
    
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


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, n_head, l, dropout=0.1):
        """
        dropout; pre/post norm
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_head = n_head
        self.l = l
        self.p = dropout
        self.decoder_layer = self.init_decoder()
        self.init_params()

    def init_params(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # print("Init params {} with xavier_uniform".format(name))
                nn.init.xavier_uniform_(p)

    def init_decoder(self):
        each_layer = nn.TransformerDecoderLayer(self.d_model, self.n_head, self.d_ff, dropout=self.p, batch_first=True)
        layers = nn.TransformerDecoder(each_layer, self.l)
        return layers
    
    def forward(self, x, y, x_padding_mask=None, y_padding_mask=None):
        """"
        y: B x S x D
        x: hidden from source
        """
        attn_mask = (nn.Transformer.generate_square_subsequent_mask(y.size()[1]) == -torch.inf).to(y.device)
        hidden = self.decoder_layer(tgt=y, memory=x, tgt_mask=attn_mask, tgt_key_padding_mask=y_padding_mask, memory_key_padding_mask=x_padding_mask)
        return hidden