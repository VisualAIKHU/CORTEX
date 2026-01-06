import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class EfficientCrossAttention(nn.Module):
    def __init__(self, in_channels, att_dim):   # in_channels = Query, att_dim = Key, Value
        super().__init__()
        # Linear layers to transform input to Query, Key, and Value
        self.query_layer = nn.Linear(in_channels, att_dim)
        self.key_layer = nn.Linear(in_channels, att_dim)
        self.value_layer = nn.Linear(in_channels, att_dim)
        self.scale = 1 / (att_dim ** 0.5)

    def forward(self, input_1, input_2):
        # Apply Linear layers to create Query, Key, and Value
        Q = self.query_layer(input_1)  
        K = self.key_layer(input_2)  
        V = self.value_layer(input_2)

        # Calculate attention scores and apply scaled dot-product
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (batch, seq_len, seq_len)
        attn_scores = torch.clamp(attn_scores, min=-1e2, max=1e2) 
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, seq_len, seq_len)

        # Compute attention output
        output = torch.bmm(attn_weights, V)  # (batch, seq_len, att_dim)
        return output
    

class ITDA(nn.Module):
    def __init__(self, in_channels, att_dim):
        super().__init__()
        self.cross_attention = EfficientCrossAttention(in_channels, att_dim)

    def forward(self, image_feats, cap_feats_list):
        # 1. Pad captions to the same length
        cap_feats_padded = pad_sequence(cap_feats_list, batch_first=True)  # (batch, max_caps, 512)
        max_caps = cap_feats_padded.size(1)

        # 2. Expand captions and image features to (batch, max_caps, 196, 512)
        cap_feats_padded = cap_feats_padded.unsqueeze(2).expand(-1, -1, 196, -1)  # (batch, max_caps, 196, 512)
        image_feats_expanded = image_feats.unsqueeze(1).expand(-1, max_caps, -1, -1)  # (batch, max_caps, 196, 512)

        # 3. Flatten for batch-wise cross-attention
        cap_feats_flat = cap_feats_padded.reshape(-1, 196, 512)  # (batch * max_caps, 196, 512)
        image_feats_flat = image_feats_expanded.reshape(-1, 196, 512)  # (batch * max_caps, 196, 512)
       
        # 4. Apply cross-attention
        attn_output = self.cross_attention(cap_feats_flat, image_feats_flat)  # (batch * max_caps, 196, 512)

        # 5. Reshape back and compute mean over valid captions
        attn_output = attn_output.view(-1, max_caps, 196, 512) # (batch, max_caps, 196, 512)
        mask = (cap_feats_padded.sum(dim=-1) != 0).float().unsqueeze(-1)  # (batch, max_caps, 196, 1) => padded caption if sum of 512-dim is 0  

        mask_sum = mask.sum(dim=1).clamp(min=1e-6)  # Ensure minimum value
        aggregated_features = (attn_output * mask).sum(dim=1) / mask_sum

        return aggregated_features  # (batch, 196, 512)


class CrossTransformer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, input1, input2):
        attn_output, attn_weight = self.attention(input1, input2, input2)
        output = input1 + self.dropout1(attn_output)
        output = self.norm1(output)
        return output


class CORTEX(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.feat_dim = cfg.model.transformer_encoder.feat_dim
        self.att_dim = cfg.model.transformer_encoder.att_dim
        self.att_head = cfg.model.transformer_encoder.att_head

        self.embed_dim = cfg.model.transformer_encoder.emb_dim

        self.img = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.att_dim, kernel_size=1, padding=0),
        )

        self.w_embedding = nn.Embedding(14, int(self.att_dim / 2))
        self.h_embedding = nn.Embedding(14, int(self.att_dim / 2))

        self.mlp = nn.Sequential(
            nn.Linear(self.att_dim, self.att_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.att_dim * 4, self.att_dim)
        )

        self.embed_fc = nn.Sequential(
            nn.Linear(self.att_dim*2, self.embed_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
        )

        self.embed_fc2 = nn.Sequential(
            nn.Linear(self.att_dim*4, self.embed_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
        )

        self.num_hidden_layers = cfg.model.transformer_encoder.att_layer
        self.transformer = nn.ModuleList([CrossTransformer(self.att_dim, self.att_head)
                                          for i in range(self.num_hidden_layers)])
        self.itda = ITDA(in_channels=self.att_dim, att_dim=self.att_dim)
        self.text_img_att = EfficientCrossAttention(in_channels=self.att_dim, att_dim=self.att_dim)
        self.fc = nn.Linear(768, 512)
        self.cross_attn = EfficientCrossAttention(in_channels=self.att_dim, att_dim=self.att_dim)
        self.self_attn = EfficientCrossAttention(in_channels=self.att_dim, att_dim=self.att_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, input_1, input_2, cap1, cap2):
        batch_size, C, H, W = input_1.size()

        cap1_origin = [self.fc(c) for c in cap1] # [n_sentences, 512]
        cap2_origin = [self.fc(c) for c in cap2] # [n_sentences, 512]

        cap1 = [c.mean(dim=0) for c in cap1_origin]  # Average each element of cap1 to (512,)
        cap2 = [c.mean(dim=0) for c in cap2_origin]  # Average each element of cap2 to (512,)

        cap1 = torch.stack(cap1, dim=0)  # (batch_size, 512)
        cap2 = torch.stack(cap2, dim=0)  # (batch_size, 512)

        cap1 = cap1.unsqueeze(1).expand(-1, H * W, -1)  # (batch_size, 196, 512)
        cap2 = cap2.unsqueeze(1).expand(-1, H * W, -1)  # (batch_size, 196, 512)


        ''''''''''''''''''''''''''''''''''''''''' Existing Change Captioning Module '''''''''''''''''''''''''''''''''''''''''''''''
        input_1 = self.img(input_1)  # (128,196, 512)
        input_2 = self.img(input_2)

        pos_w = torch.arange(W).cuda()
        pos_h = torch.arange(H).cuda()
        
        embed_w = self.w_embedding(pos_w)
        embed_h = self.h_embedding(pos_h)
        position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(W, 1, 1),
                                        embed_h.unsqueeze(1).repeat(1, H, 1)],
                                       dim=-1)

        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (batch, d_model, h, w)

        input_1 = input_1 + position_embedding  # (batch, att_dim, h, w)
        input_2 = input_2 + position_embedding

        input_1 = input_1.view(batch_size, self.att_dim, -1).permute(0, 2, 1)
        input_2 = input_2.view(batch_size, self.att_dim, -1).permute(0, 2, 1)


        #####################
        img_mask = torch.Tensor(np.ones([batch_size, H*W])).cuda()
        input_1_feat = input_1[img_mask.bool()]
        input_2_feat = input_2[img_mask.bool()]
        input_1_feat = self.mlp(input_1_feat)
        input_2_feat = self.mlp(input_2_feat)
        input_1_feat = input_1_feat
        input_2_feat = input_2_feat
        z_a_norm = (input_1_feat - input_1_feat.mean(0)) / input_1_feat.std(0)  # NxN_sxD
        z_b_norm = (input_2_feat - input_2_feat.mean(0)) / input_2_feat.std(0)  # NxN_txD
        # cross-correlation matrix
        B, D = z_a_norm.shape
        c = (z_a_norm.T @ z_b_norm)  # DxD
        c.div_(B)
        # loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = c.flatten()[1:].view(D - 1, D + 1)[:, :-1].pow_(2).sum()
        cdcr_loss = (on_diag * 1 + off_diag * 0.003)
        ##################################################

        input_1 = input_1.transpose(0, 1)
        input_2 = input_2.transpose(0, 1)

        input_1_pre = input_1 # h*w, b, att_dim
        input_2_pre = input_2

        for l in self.transformer:
            input_1, input_2 = l(input_1, input_2), l(input_2, input_1)      

        input_1_diff = (input_1_pre - input_1).permute(1, 0, 2) # bs, 196, 512
        input_2_diff = (input_2_pre - input_2).permute(1, 0, 2)
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''' ITDA Module '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        input1_cross_attn = self.cross_attn(input_1_diff, input_2_diff)   #torch.Size([128, 196, 512])
        input2_cross_attn = self.cross_attn(input_2_diff, input_1_diff)   #torch.Size([128, 196, 512])

        input1_self_attn = self.self_attn(input_1_diff, input_1_diff)  # Self-attention on input_1
        input2_self_attn = self.self_attn(input_2_diff, input_2_diff)  # Self-attention on input_2

        ### Dynamic Alignment Text-Image Cross Attention
        d_bef_feat = self.itda(input_1_diff, cap2_origin)
        d_aft_feat = self.itda(input_2_diff, cap1_origin)

        ### Static Alignment Text-Image Cross Attention
        s_bef_feat = self.itda(input_1_diff, cap1_origin)
        s_aft_feat = self.itda(input_2_diff, cap2_origin)


        # Calculate loss between cross-attention output and d_bef_feat, d_aft_feat
        img_to_txt_loss = F.mse_loss(d_bef_feat, input1_cross_attn) + F.mse_loss(d_aft_feat, input2_cross_attn) + F.mse_loss(s_bef_feat, input1_self_attn) + F.mse_loss(s_aft_feat, input2_self_attn)
       

        dynamic_att = torch.cat([d_bef_feat, d_aft_feat], -1) # (bs, 196, 1024)
        static_att = torch.cat([s_bef_feat, s_aft_feat], -1) # (bs, 196, 1024)

        dynamic_att = self.embed_fc(dynamic_att)  # (bs, 196, 512)
        static_att = self.embed_fc(static_att)    # (bs, 196, 512)

        output_align = torch.cat([dynamic_att, static_att], -1)
        output_align = self.embed_fc(output_align)
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


        '''RTE Module'''
        output_txt = torch.cat([cap1, cap2], -1)
        output_txt = self.embed_fc(output_txt)
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        output_img_diff = torch.cat([input_1_diff, input_2_diff], -1)
        output_combine = torch.cat([output_img_diff, output_txt, output_align], -1)
        output = self.embed_fc2(output_combine)

        return output, cdcr_loss, img_to_txt_loss


class AddSpatialInfo(nn.Module):

    def _create_coord(self, img_feat):
        batch_size, _, h, w = img_feat.size()
        coord_map = img_feat.new_zeros(2, h, w)
        for i in range(h):
            for j in range(w):
                coord_map[0][i][j] = (j * 2.0 / w) - 1
                coord_map[1][i][j] = (i * 2.0 / h) - 1
        sequence = [coord_map] * batch_size
        coord_map_in_batch = torch.stack(sequence)
        return coord_map_in_batch

    def forward(self, img_feat):
        coord_map = self._create_coord(img_feat)
        img_feat_aug = torch.cat([img_feat, coord_map], dim=1)
        return img_feat_aug

