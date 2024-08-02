import torch
from torch import nn
import torch.nn.functional as F

# from .fast_transformer_decoder import FastTransformerDecoder
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from .transformer_helpers import (
  TokenEmbedding,
  PositionalEncoding,
  weights_init
)

def triangular_causal_mask(length, device):
  return torch.tril(torch.ones(length, length)).to(device)

class MusicGPT2(nn.Module):
  def __init__(self, n_token, n_layer, n_head, d_model, d_ff, d_embed,
    activation='relu', dropout=0.1, use_pe=True,
    use_segment_emb=False, n_segment_types=None,
    use_chord_mhot_emb=False
  ):
    super(MusicGPT2, self).__init__()
    self.n_token = n_token
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.dropout = dropout
    self.activation = activation

    self.token_emb = TokenEmbedding(n_token, d_embed, d_model)
    self.d_embed = d_embed

    self.pe = PositionalEncoding(d_embed)
    self.dec_out_proj = nn.Linear(d_model, n_token)

    # self.transformer_decoder = FastTransformerDecoder(
    #   n_layer, n_head, d_model, d_ff, dropout, activation, favor_feature_dims
    # )
    gpt_config = GPT2Config(
      n_layer=n_layer,
      n_head=n_head,
      n_embd=d_model,
      n_inner=d_ff,
      resid_pdrop=dropout,
      attn_pdrop=dropout,
      max_position_embeddings=4096,
    )
    self.transformer_decoder = nn.ModuleList([GPT2Block(gpt_config, layer_idx=i) for i in range(n_layer)])

    self.emb_dropout = nn.Dropout(self.dropout)
    self.use_pe = use_pe

    self.use_segment_emb = use_segment_emb
    if self.use_segment_emb:
      self.segemb = TokenEmbedding(n_segment_types, d_embed, d_model)
      self.n_segment_types = n_segment_types
    else:
      self.segemb = None

    self.use_chord_mhot_emb = use_chord_mhot_emb
    if use_chord_mhot_emb:
      self.chord_emb = nn.Linear(12, d_model)

    self.apply(weights_init)
    print ('[info] model init completed')

  def forward(self, x, seg_inp=None, chord_inp=None, keep_last_only=False, attn_kwargs=None):
    x_emb = self.token_emb(x)

    if seg_inp is not None and self.use_segment_emb:
      x_emb += self.segemb(seg_inp)

    if chord_inp is not None and self.use_chord_mhot_emb:
      x_emb += self.chord_emb(chord_inp)

    if self.use_pe:
      x_inp = self.emb_dropout(x_emb + self.pe(x.size(1)).permute(1, 0, 2))
    else:
      x_inp = self.emb_dropout(x_emb)

    dec_out = x_inp
    for i in range(self.n_layer):
      dec_out = self.transformer_decoder[i].forward(dec_out)[0]
    dec_logits = self.dec_out_proj(dec_out)

    if keep_last_only:
      dec_logits = dec_logits[:, -1, :]

    return dec_logits

  def compute_loss(self, dec_logits, dec_tgt, reduction='mean'):
    recons_loss = F.cross_entropy(
      dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), 
      ignore_index=self.n_token - 1, reduction=reduction
    ).float()

    return {
      'recons_loss': recons_loss,
      'total_loss': recons_loss
    }

if __name__ == "__main__":
  # mask = triangular_causal_mask(100, "cpu")
  # print(mask.size(), mask[:10, :10])

  bsize, seqlen = 2, 2048
  model = MusicGPT2(100, 12, 8, 512, 2048, 512).to("cuda")

  inp = torch.randint(0, 80, (bsize, seqlen)).to("cuda")
  out = model.forward(inp)
  print(out.size())