import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import math

import clip


class TransformerAutoencoder_hml(nn.Module):
    def __init__(self, transfomers_clip=False):
        super(TransformerAutoencoder_hml, self).__init__()

        self.transfomers_clip = transfomers_clip
        self.clip_model = None
        self.clip_text_model = None
        self.clip_tokenizer_transformer = None

        self.load_and_freeze_clip()


    def load_and_freeze_clip(self):
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if self.transfomers_clip:
            from transformers import AutoTokenizer, CLIPTextModel
            clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            clip_text_model.eval()
            for p in clip_text_model.parameters():
                p.requires_grad = False

            self.clip_text_model = clip_text_model
            self.clip_tokenizer_transformer = tokenizer

        clip_model, clip_preprocess = clip.load('ViT-B/32', jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        self.clip_model = clip_model

    def encode_text(self, raw_text, max_text_len=20):
        if self.transfomers_clip:
            inputs = self.clip_tokenizer_transformer(raw_text, padding='max_length',
                                                     truncation=True, return_tensors="pt").to("cuda")
            outputs = self.clip_text_model(**inputs)
            output_last_hidden_states = outputs.last_hidden_state.float()  # shape: [bs, seq_len, dim]
            # convert to shape: [seq_len, bs, dim]
            return output_last_hidden_states.permute(1, 0, 2)
        else:
            # raw_text - list (batch_size length) of strings with input text prompts
            # device = next(self.parameters()).device
            device = "cuda"
            # max_text_len = max_text_len  # Specific hardcoding for humanml dataset
            # assert max_text_len is None  # for this time, should be None
            if max_text_len is not None:
                default_context_length = 77
                context_length = max_text_len + 2  # start_token + 20 + end_token
                assert context_length < default_context_length
                texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(
                    device)  # [bs, context_length] # if n_tokens > context_length -> will truncate
                # print('texts', texts.shape)

                zero_pad = torch.zeros([texts.shape[0], default_context_length - context_length], dtype=texts.dtype,
                                       device=texts.device)
                texts = torch.cat([texts, zero_pad], dim=1)
                # print('texts after pad', texts.shape, texts)
            else:
                texts = clip.tokenize(raw_text, truncate=True).to(
                    device)  # [bs, context_length] # if n_tokens > 77 -> will truncate
            self.clip_model.float()
            text_encoding_results = self.clip_model.encode_text(texts)
            text_encoding_results = text_encoding_results.to(self.embed_text.weight.dtype)
            return text_encoding_results

    def encode_image(self, image):
        # image - torch.Tensor of shape [bs, 3, 224, 224]
        image_encoding_results = self.clip_model.encode_image(image)
        image_encoding_results = image_encoding_results.to(self.embed_image.weight.dtype)
        return image_encoding_results

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)

class TransformerAutoencoder_withCodes_hml_G2_noTraj(TransformerAutoencoder_hml):
    def __init__(self,
                 input_dim=3,
                 dropout=0.1,
                 num_emb=128,
                 max_text_len=20,
                 text_drop_rate=0,
                 image_drop_rate=0,
                 transfomers_clip=False,
                 ):
        super(TransformerAutoencoder_withCodes_hml_G2_noTraj, self).__init__(transfomers_clip=transfomers_clip)
        self.max_text_len = max_text_len
        self.input_dim = input_dim
        self.text_drop_rate = text_drop_rate
        self.image_drop_rate = image_drop_rate
        # if self.input_dim > 3:
        #    self.input_dim *= 28
        self.codes_realLength = 4  # 4

        self.model_dim1 = 512
        self.model_dim2 = 256

        self.clip_dim = 512  # clip_dim

        # self.clip_version = 'ViT-B/32'
        # self.load_and_freeze_clip(self.clip_version)
        self.embed_text = nn.Linear(self.clip_dim, self.model_dim1)
        self.embed_image = nn.Linear(self.clip_dim, self.model_dim1)

        self.linear_in = nn.Linear(self.codes_realLength * self.input_dim, self.model_dim1)

        # self.pos_encoder = PositionalEncoding(self.model_dim_part, args.dropout)
        self.pos_encoder = PositionalEncoding(self.model_dim1, dropout)

        # self.tokenEmb = TokenTypeEncoding(self.model_dim, self.pos_encoder)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.model_dim1,
            nhead=4,  # args.num_heads,
            dim_feedforward=self.model_dim1 * 4,
            dropout=dropout)
        # Define transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=4)

        self.linear_mid_codeIdx = nn.Linear(self.model_dim1, self.model_dim2)
        # self.linear_mid_traj = nn.Linear(self.model_dim1, self.model_dim2)

        self.num_emb = num_emb

        decoder_layers1 = nn.TransformerEncoderLayer(
            d_model=self.model_dim2,
            nhead=4,  # Number of heads
            dim_feedforward=self.model_dim2 * 4,
            dropout=dropout
        )
        # Define transformer decoder
        self.transformer_codeIdx = nn.TransformerEncoder(
            encoder_layer=decoder_layers1,
            num_layers=3
        )
        self.project_codeIdx = nn.Linear(self.model_dim2, self.num_emb)

    def forward(self, x, x_text, x_image=None):  # Toekn size: T
        # x: bs, T, NumJoint, 3
        # y: bs, t, NumJoint

        assert x_text is not None
        if self.text_drop_rate > 0:
            if random.random() <= self.text_drop_rate:
                x_text = [""] * x.shape[0]
        enc_text = self.encode_text(x_text, max_text_len=self.max_text_len)
        input_feature = self.embed_text(enc_text)
        if not self.transfomers_clip:
            input_feature = input_feature.unsqueeze(0)
        if x_image is not None:
            enc_image = self.encode_image(x_image)
            input_image_feature = self.embed_image(enc_image).unsqueeze(0)
            if self.image_drop_rate > 0:
                if random.random() <= self.image_drop_rate:
                    input_image_feature = input_image_feature * 0.
            input_feature = torch.cat([input_feature, input_image_feature], dim=0)



        # x = x.reshape((-1, 196, 6*3))
        bs, T, NJoints, input_dim = x.shape
        x = x.permute(2, 1, 0, 3)  # x:  NumJoint, T, bs, model_dim
        Tt = T // self.codes_realLength  # the token number in the sequence
        x = x.reshape(NJoints, Tt, self.codes_realLength, bs, input_dim)
        x = x.permute(1, 0, 3, 2, 4)
        x = x.reshape(Tt * NJoints, bs, self.codes_realLength * input_dim)

        x = self.linear_in(x)

        x = torch.cat([input_feature, x], dim=0)

        x = self.pos_encoder(x)

        result_text = self.transformer_encoder(x)
        # result = result_text[1:, :, :] # Tt*NJoints, bs, model_dim

        pre_codes = self.transformer_codeIdx(self.linear_mid_codeIdx(result_text))
        pre_codes = pre_codes[len(input_feature):, :, :]
        pre_codes = self.project_codeIdx(pre_codes)
        pre_codes = pre_codes.reshape(Tt, NJoints, bs, -1)
        pre_codes = pre_codes.permute(2, 1, 0, 3)
        return None, pre_codes

#
