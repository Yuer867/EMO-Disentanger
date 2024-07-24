import math
import torch
from torch import nn
import torch.nn.functional as F

def generate_causal_mask(seq_len, device):
    mask = (torch.triu(torch.ones(seq_len, seq_len, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask.requires_grad = False
    return mask

def generate_bidirectional_pad_mask(max_seqlen, batch_lens):
    mask = torch.zeros(len(batch_lens), max_seqlen, dtype=bool)
    for i, l in enumerate(batch_lens):
        mask[i, l:] = True
    return mask

def weight_init_normal(weight, normal_std):
  nn.init.normal_(weight, 0.0, normal_std)

def weight_init_orthogonal(weight, gain):
  nn.init.orthogonal_(weight, gain)

def bias_init(bias):
  nn.init.constant_(bias, 0.0)
  
def weights_init(m):
    classname = m.__class__.__name__
    # print ('[{}] initializing ...'.format(classname))

    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            weight_init_normal(m.weight, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            bias_init(m.bias)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            weight_init_normal(m.weight, 0.01)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            bias_init(m.bias)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            weight_init_normal(m.cluster_weight, 0.01)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            bias_init(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    weight_init_normal(m.out_projs[i], 0.02)
    elif classname.find('TXLDecoder') != -1:
        if hasattr(m, 'r_emb'):
            weight_init_normal(m.r_emb, 0.01)
        if hasattr(m, 'r_w_bias'):
            weight_init_normal(m.r_w_bias, 0.01)
        if hasattr(m, 'r_r_bias'):
            weight_init_normal(m.r_r_bias, 0.01)
        if hasattr(m, 'r_bias'):
            bias_init(m.r_bias)
    elif classname.find('LSTM') != -1:
        for param in m.parameters():
            if len(param.shape) >= 2:  # weights
                weight_init_orthogonal(param, 0.01)
            else:                      # biases
                bias_init(param)
    # else:
    #   print ('*** [ {:64} ] not initialized !!'.format(classname))


class SinusoidalPE(nn.Module):
    def __init__(self, d_embed, max_pos=20480):
        super(SinusoidalPE, self).__init__()
        self.d_embed = d_embed
        self.max_pos = max_pos

        pe = torch.zeros(max_pos, d_embed)
        position = torch.arange(0, max_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, seq_len, bsz=None):
        pos_encoding = self.pe[:seq_len, :]

        if bsz is not None:
          pos_encoding = pos_encoding.expand(seq_len, bsz, -1)

        return pos_encoding

class WordEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, emb_scale=0.5, pad_idx=None):
        super(WordEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.emb_scale = d_proj ** emb_scale

        if pad_idx is None:
            pad_idx = n_token - 1
            
        self.emb_lookup = nn.Embedding(n_token, d_embed, padding_idx=pad_idx)
        if d_proj != d_embed:
            self.emb_proj = nn.Linear(d_embed, d_proj, bias=False)
        else:
            self.emb_proj = None

    def forward(self, inp_tokens):
        inp_emb = self.emb_lookup(inp_tokens)
        
        if self.emb_proj is not None:
            inp_emb = self.emb_proj(inp_emb)

        return inp_emb.mul_(self.emb_scale)

class OctaveAwarePitchEmbedding(nn.Module):
    def __init__(self, n_octave, d_embed, d_proj, idx2event,
                 emb_scale=0.5, n_chroma=12, min_pitch=12
        ):
        super(OctaveAwarePitchEmbedding, self).__init__()

        self.n_octave = n_octave
        self.n_chroma = n_chroma
        self.min_pitch = min_pitch

        self.d_embed = d_embed
        self.d_proj = d_proj
        self.emb_scale = d_proj ** emb_scale

        self.octave_emb_lookup = nn.Embedding(
            n_octave + 1, d_embed // 2, padding_idx=n_octave
        )
        self.chroma_emb_lookup = nn.Embedding(
            n_chroma + 1, d_embed // 2, padding_idx=n_chroma
        )

        if d_proj != d_embed:
            self.emb_proj = nn.Linear(d_embed, d_proj, bias=False)
        else:
            self.emb_proj = None

        self.octave_translate_dict, self.chroma_translate_dict =\
            self.make_idx_translate_dicts(idx2event)

    def make_idx_translate_dicts(self, idx2event):
        idx2event[ len(idx2event) ] = 'PAD_None'

        octave_dict = dict()
        chroma_dict = dict()
        for idx, ev in idx2event.items():
            if not 'Note_Pitch' in ev:
                octave_dict[idx] = self.n_octave
                chroma_dict[idx] = self.n_chroma
            else:
                pitch = int(ev.split('_')[-1])
                pitch -= self.min_pitch
                octave_dict[idx] = pitch // self.n_chroma
                chroma_dict[idx] = pitch % self.n_chroma
        
        return octave_dict, chroma_dict

    def forward(self, inp_tokens):
        # st = time.time()
        orig_device = inp_tokens.device

        octave_tokens = inp_tokens.clone().cpu()
        chroma_tokens = inp_tokens.clone().cpu()

        octave_tokens.apply_(self.octave_translate_dict.get)
        chroma_tokens.apply_(self.chroma_translate_dict.get)
        octave_tokens = octave_tokens.to(orig_device)
        chroma_tokens = chroma_tokens.to(orig_device)
        # print ('[mapping] {:.3f}'.format(time.time() - st))
        
        # st = time.time()
        octave_emb = self.octave_emb_lookup(octave_tokens)
        chroma_emb = self.chroma_emb_lookup(chroma_tokens)
        inp_emb = torch.cat([octave_emb, chroma_emb], dim=-1)

        # print ('[bedding] {:.3f}'.format(time.time() - st))
        
        if self.emb_proj is not None:
            inp_emb = self.emb_proj(inp_emb)

        return inp_emb.mul_(self.emb_scale)

def get_min_max_pitch_idx(idx2event):
    min_idx, max_idx = len(idx2event), 0

    for k, v in idx2event.items():
        if 'Note_Pitch' in v:
            min_idx = min(min_idx, k)
            max_idx = max(max_idx, k)
    
    return min_idx, max_idx
