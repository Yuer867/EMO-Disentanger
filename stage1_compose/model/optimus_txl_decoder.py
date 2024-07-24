import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_helpers import WordEmbedding


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]



class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MultiHeadCrossAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False, **kwargs):
        super(MultiHeadCrossAttn, self).__init__()
    
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm

    def forward(self, h, c, attn_mask=None, h_pos_embed=None, c_pos_embed=None):
        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        if h_pos_embed is not None:
            h_ = h + self.drop(h_pos_embed)
        else:
            h_ = h
        if c_pos_embed is not None:
            c_ = c + self.drop(c_pos_embed)
        else:
            c_ = c

        head_q = self.q_net(h_)
        head_k, head_v = torch.chunk(self.kv_net(c_), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # print ('[cross inputs]', head_q.mean(), head_k.mean(), head_v.mean())

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        # print ('[attn score]', attn_score.mean(), attn_score.std())
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))
        # print (attn_score[0, :128, :, 0])
        # print ('[masked attn score]', attn_score.mean(), attn_score.std())

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_prob = attn_prob / (torch.sum(attn_prob, dim=1)[:, None, :, :] + 1e-8)
        # print (attn_prob[0, :128, :, 0])
        # print (torch.isnan(attn_prob).sum())
        # idx = torch.nonzero(torch.isnan(attn_prob))
        # for i in idx:
        #     print (i)
        # print ('[cross attn prob]', attn_prob.mean(), attn_prob.std())
        # exit()

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        # print ('[cross attn vec]', attn_vec.mean(), attn_vec.std())

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)
        # print ('[cross attn out]', attn_out.mean(), attn_out.std())

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        # attn_prob = attn_prob / (torch.sum(attn_prob, dim=1)[:, None, :, :] + 1e-8)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False, **kwargs):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None, return_avg_attn=False):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)
        # print ('[masked self attn score]', attn_score.mean(), attn_score.std())


        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        if return_avg_attn:
            avg_attn_prob = attn_prob.mean(dim=-1)
        attn_prob = self.dropatt(attn_prob)
        attn_prob = attn_prob / (torch.sum(attn_prob, dim=1)[:, None, :, :] + 1e-8)
        # print ('[self attn prob]', attn_prob.mean(), attn_prob.std())

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)
        
        if not return_avg_attn:
            return output
        else:
            return output, avg_attn_prob


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]                                   # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))                  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]                                              # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_prob = attn_prob / (torch.sum(attn_prob, dim=1)[:, None, :, :] + 1e-8)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout,
                                         **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

        if 'use_cross_attn' in kwargs and kwargs.get('use_cross_attn') is True:
            self.cross_attn = MultiHeadCrossAttn(n_head, d_model, d_head, dropout, **kwargs)
        else:
            self.cross_attn = None

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None, 
                cross_latent=None, dec_cross_pos_emb=None, latent_cross_pos_emb=None,
                cross_attn_mask=None, return_avg_attn=False):

        if not return_avg_attn:
            output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                                attn_mask=dec_attn_mask,
                                mems=mems, return_avg_attn=False)
        else:
            output, avg_attn_prob = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                                                  attn_mask=dec_attn_mask,
                                                  mems=mems, return_avg_attn=True)

        if self.cross_attn is not None and cross_latent is not None:
            if dec_cross_pos_emb is None:
                dec_cross_pos_emb = torch.zeros_like(dec_inp)
            if latent_cross_pos_emb is None:
                latent_cross_pos_emb = torch.zeros_like(cross_latent)  
            
            output = self.cross_attn.forward(
                                output, cross_latent,
                                attn_mask=cross_attn_mask,
                                h_pos_embed=dec_cross_pos_emb, 
                                c_pos_embed=latent_cross_pos_emb
                            )

        output = self.pos_ff(output)

        if not return_avg_attn:
            return output
        else:
            return output, avg_attn_prob

class SegmentEmbeddingProj(nn.Module):
    def __init__(self, d_in, d_out, n_layer=None, tie_seg_emb_projs=True, scale=1.):
        super(SegmentEmbeddingProj, self).__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.emb_proj = nn.ModuleList()
        self.tie_seg_emb_projs = tie_seg_emb_projs

        if tie_seg_emb_projs:
            self.emb_proj.append( nn.Linear(d_in, d_out, bias=False) )
        else:
            for l in range(n_layer):
                self.emb_proj.append( nn.Linear(d_in, d_out, bias=False) )
        
        self.scale = scale
        print ('[seg emb scale]', scale)

    def forward(self, inp, layer=None):
        if layer is None or self.tie_seg_emb_projs:
            emb_out = self.emb_proj[0](inp)
        else:
            emb_out = self.emb_proj[layer](inp)

        return emb_out.mul_(self.scale)


class OptimusTXLDecoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner, d_segment_emb,
                 dropout, dropatt, pre_lnorm=False, use_segment_emb=True,
                 tgt_len=None, ext_len=None, mem_len=None, 
                 same_length=False, attn_type=0, clamp_len=-1, 
                 tie_seg_emb_projs=True, in_attn_cond=True,
                 use_cross_attn=False, cross_len=192, seg_proj_scale=1.
        ):
        super(OptimusTXLDecoder, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.drop = nn.Dropout(dropout)
        self.n_layer = n_layer
        self.d_segment_emb = d_segment_emb

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len
        self.pre_lnorm = pre_lnorm
        self.use_segment_emb = use_segment_emb

        self.tie_seg_emb_projs = tie_seg_emb_projs
        self.in_attn_cond = in_attn_cond
        
        if self.use_segment_emb:
            self.seg_proj_scale = seg_proj_scale
            self.seg_emb_projs = SegmentEmbeddingProj(
                                    d_segment_emb, d_model, n_layer, tie_seg_emb_projs,
                                    scale=self.seg_proj_scale
                                )
        else:
            self.seg_emb_projs = None

        self.use_cross_attn = use_cross_attn
        if self.use_cross_attn:
            self.cross_len = cross_len
            self.cross_pos_emb = WordEmbedding(
                                    cross_len, d_model, d_model, emb_scale=0.2
                                )

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        if attn_type == 0: # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        use_cross_attn=use_cross_attn)
                )
        elif attn_type == 1: # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type in [2, 3]: # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0: # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1: # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2: # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3: # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self, batchsize=None):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                if batchsize is None:
                    empty = torch.empty(0, dtype=param.dtype, device=param.device)
                else:
                    empty = torch.empty(0, batchsize, self.d_model, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen, dec_seg_len=None):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []

            if dec_seg_len is None:
                end_idx = mlen + max(0, qlen - 0 - self.ext_len)
                beg_idx = max(0, end_idx - self.mem_len)
                for i in range(len(hids)):
                    cat = torch.cat([mems[i], hids[i]], dim=0)
                    new_mems.append(cat[beg_idx:end_idx].detach())

            else:      # different len for each sample in batch, `ext_len != 0` is not supported
                assert dec_seg_len.size(0) == hids[0].size(1)
                batchsize = hids[0].size(1)

                for i in range(len(hids)):
                    new_layer_mem = []
                    for samp_idx in range(batchsize):
                        samp_len = dec_seg_len[samp_idx]
                        old_samp_mem = mems[i][:, samp_idx, :]
                        new_samp_mem = hids[i][:samp_len, samp_idx, :]
                        cat = torch.cat([old_samp_mem, new_samp_mem], dim=0)
                        end_idx, beg_idx = cat.size(0), max(0, cat.size(0) - self.mem_len)
                        new_layer_mem.append(cat[beg_idx:end_idx].detach())

                    max_new_mlen = max([cat.size(0) for cat in new_layer_mem])
                    for samp_idx in range(batchsize):
                        samp_new_mlen = new_layer_mem[ samp_idx ].size(0)
                        if samp_new_mlen < max_new_mlen:
                            new_layer_mem[samp_idx] = torch.cat([
                                torch.zeros(max_new_mlen - samp_new_mlen, mems[i].size(-1), dtype=mems[i].dtype, device=mems[i].device).detach(),
                                new_layer_mem[samp_idx]
                            ], dim=0)
                    new_mems.append(torch.stack(new_layer_mem, dim=1).detach())
                
        return new_mems

    def _forward(self, dec_input, segment_emb, mems=None, dec_seg_len=None,
                 cross_latent=None, cross_attn_mask=None, 
                 dec_cross_pos_emb=None, latent_cross_pos_emb=None, return_avg_attn=False):
        qlen, bsz, _ = dec_input.size()
        # print ('[debug] reached inner _forward()')

        if isinstance(mems, tuple) and len(mems) == 1:
            mems = mems[0]
            assert len(mems) == self.n_layer + 1

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = dec_input.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).bool()[:, :, None] # -1
        else:
            dec_attn_mask = torch.triu(
                dec_input.new_ones(qlen, klen), diagonal=1+mlen).bool()[:,:,None]

        hids = []
        if return_avg_attn:
            all_layer_avg_attn_probs = []
        if self.use_segment_emb:
            layer_seg_emb = self.seg_emb_projs(segment_emb, layer=0)
        else:
            layer_seg_emb = torch.zeros_like(dec_input, device=dec_input.device)

        if self.use_cross_attn and cross_latent is not None:
            layer_cross_latent = self.drop(
                                    self.seg_emb_projs(cross_latent, layer=0)
                                 )
        else:
            layer_cross_latent = None
        # print ('[cross pos embs]', dec_cross_pos_emb.mean(), latent_cross_pos_emb.mean())

        if self.attn_type == 0: # default
            pos_seq = torch.arange(klen-1, -1, -1.0, device=dec_input.device, 
                                   dtype=dec_input.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(dec_input)
            # print ('[layer 0] inp: {:.3f} (+/- {:.3f}) | segemb: {:.3f} (+/- {:.3f})'.format(
            #     core_out.mean().item(), core_out.std().item(), 
            #     layer_seg_emb[ layer_seg_emb != 0. ].mean().item(), layer_seg_emb[ layer_seg_emb != 0. ].std().item()
            # ))
            core_out += self.drop(layer_seg_emb)
            pos_emb = self.drop(pos_emb)
            hids.append(core_out)

            for i, layer in enumerate(self.layers):
                # print ('[cross latent]', layer_cross_latent.mean())
                mems_i = None if mems is None else mems[i]
                if not return_avg_attn:
                    core_out = layer(
                                    core_out, pos_emb, self.r_w_bias,
                                    self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i,
                                    cross_latent=layer_cross_latent,
                                    dec_cross_pos_emb=dec_cross_pos_emb,
                                    latent_cross_pos_emb=latent_cross_pos_emb,
                                    cross_attn_mask=cross_attn_mask,
                                    return_avg_attn=False
                                )
                else:
                    core_out, layer_avg_attn_prob = layer(
                                    core_out, pos_emb, self.r_w_bias,
                                    self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i,
                                    cross_latent=layer_cross_latent,
                                    dec_cross_pos_emb=dec_cross_pos_emb,
                                    latent_cross_pos_emb=latent_cross_pos_emb,
                                    cross_attn_mask=cross_attn_mask,
                                    return_avg_attn=True
                                )
                    all_layer_avg_attn_probs.append(layer_avg_attn_prob)
                    # print ('[avg attn probs]', all_layer_avg_attn_probs[-1].size())

                if i != len(self.layers) - 1 and self.in_attn_cond and self.use_segment_emb:
                    layer_seg_emb = self.seg_emb_projs(segment_emb, layer=i+1)
                    core_out += self.drop(layer_seg_emb)
                    if self.use_cross_attn:
                        layer_cross_latent = self.drop(
                                        self.seg_emb_projs(cross_latent, layer=i+1)
                                    )

                hids.append(core_out)
                # print ('[layer {}] inp: {:.3f} (+/- {:.3f})'.format(
                #     i+1, core_out.mean().item(), core_out.std().item()
                # ))
                # print ('[layer {}] inp: {:.3f} (+/- {:.3f}) | segemb: {:.3f} (+/- {:.3f})'.format(
                #     i + 1,
                #     core_out.mean().item(), core_out.std().item(), 
                #     layer_seg_emb[ layer_seg_emb != 0. ].mean().item(), layer_seg_emb[ layer_seg_emb != 0. ].std().item()
                # ))

        elif self.attn_type == 1: # learnable
            core_out = self.drop(dec_input)
            core_out += self.drop(layer_seg_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len :]
                    r_bias = self.r_bias[i][-self.clamp_len :]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                        r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                if i != len(self.layers) - 1 and self.in_attn_cond and self.use_segment_emb:
                    layer_seg_emb = self.seg_emb_projs(segment_emb, layer=i+1)
                    core_out += self.drop(layer_seg_emb)
                hids.append(core_out)

        elif self.attn_type == 2: # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=dec_input.device,
                                   dtype=dec_input.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(dec_input + pos_emb[-qlen:])
            core_out += self.drop(layer_seg_emb)
            hids.append(core_out)

            for i, layer in enumerate(self.layers):
                layer_seg_emb = self.seg_emb_projs(segment_emb, layer=i)
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                if i != len(self.layers) - 1 and self.in_attn_cond and self.use_segment_emb:
                    print ('shouldn\'t be here !!!')
                    layer_seg_emb = self.seg_emb_projs(segment_emb, layer=i+1)
                    core_out += self.drop(layer_seg_emb)
                hids.append(core_out)
                
        elif self.attn_type == 3:
            core_out = self.drop(dec_input)
            core_out += self.drop(layer_seg_emb)
            hids.append(core_out)

            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                if i != len(self.layers) - 1 and self.in_attn_cond and self.use_segment_emb:
                    layer_seg_emb = self.seg_emb_projs(segment_emb, layer=i+1)
                    core_out += self.drop(layer_seg_emb)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen, dec_seg_len=dec_seg_len)

        if not return_avg_attn:
            return core_out, new_mems
        else:
            return core_out, new_mems, all_layer_avg_attn_probs

    def forward(self, dec_input, segment_emb, *mems, dec_seg_len=None, cross_latent=None,
                cross_attn_mask=None, dec_cross_pos_seq=None, latent_cross_pos_seq=None, return_avg_attn=False):
        if not mems: mems = self.init_mems(batchsize=dec_input.size(1) if dec_seg_len is not None else None)

        if self.use_cross_attn is True and dec_cross_pos_seq is not None and latent_cross_pos_seq is not None:
            dec_cross_pos_emb = self.cross_pos_emb(dec_cross_pos_seq)
            latent_cross_pos_emb = self.cross_pos_emb(latent_cross_pos_seq)
            # print ('[cross pos embs]', dec_cross_pos_emb.size(), latent_cross_pos_emb.size())
        else:
            dec_cross_pos_emb = latent_cross_pos_emb = None

        if not return_avg_attn:
            dec_out, new_mems = self._forward(
                                    dec_input, segment_emb, mems=mems, 
                                    dec_seg_len=dec_seg_len,
                                    cross_latent=cross_latent,
                                    cross_attn_mask=cross_attn_mask,
                                    dec_cross_pos_emb=dec_cross_pos_emb,
                                    latent_cross_pos_emb=latent_cross_pos_emb,
                                    return_avg_attn=False
                                )
        else:
            dec_out, new_mems, avg_attn_probs = self._forward(
                                                    dec_input, segment_emb, mems=mems, 
                                                    dec_seg_len=dec_seg_len,
                                                    cross_latent=cross_latent,
                                                    cross_attn_mask=cross_attn_mask,
                                                    dec_cross_pos_emb=dec_cross_pos_emb,
                                                    latent_cross_pos_emb=latent_cross_pos_emb,
                                                    return_avg_attn=True
                                                )

        if new_mems is None and not return_avg_attn:
            return [dec_out]
        elif new_mems is not None and not return_avg_attn:
            return [dec_out] + new_mems
        else:
            return [dec_out] + new_mems, avg_attn_probs

if __name__ == '__main__':
    device = 'cpu'

    tgt_len, mem_len, ext_len = 128, 600, 0

    model = OptimusTXLDecoder(n_layer=12, n_head=8, d_segment_emb=64,
                    d_model=512, d_head=64, d_inner=2048,
                    dropout=0.1, dropatt=0.1, pre_lnorm=True, tie_seg_emb_projs=False,
                    tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len).to(device)

    print(sum(p.numel() for p in model.parameters()))

    mems = tuple()
    for idx in range(10):
        # inp = torch.randint(0, 100, (tgt_len, 1))
        inp = torch.randn(128, 4, 512)
        segment_emb = torch.randn(128, 4, 64)
        print('batch {}'.format(idx))
        out = model(inp, segment_emb, *mems)
        mems = out[1:]
        print ('[dec out]', out[0].size())
        print ('[mem layer 0]', mems[0].size())
