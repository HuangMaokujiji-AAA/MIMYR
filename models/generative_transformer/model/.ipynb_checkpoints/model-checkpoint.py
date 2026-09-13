from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import inspect
import math
from loguru import logger
import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from transformers.generation.utils import SampleDecoderOnlyOutput


class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.config = config

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        y = self.resid_dropout(self.c_proj(y))
        return y

def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class MulanConfig:
    block_size: int = 1000
    vocab_size: int = 1011 
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True 
    train_mode: str = 'pretrian'
    expression_level: int = 10
    ele: int = 0
    bin_edges: np.ndarray = None


class scMulanModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.expression_level is not None
        assert config.ele == 1
        self.config = config
        if config.bin_edges is not None:
            self.bin_edges = torch.tensor(config.bin_edges)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wee = nn.Embedding(config.expression_level + 1, config.n_embd), # +1 for non gene tokens
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
            
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.epx_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # expr level

        self.epx_regressor = nn.Sequential(
            nn.Linear(config.vocab_size, config.n_embd),  # map vocab→hidden
            nn.ReLU(),
            nn.Linear(config.n_embd, 1)                   # hidden→scalar
        )

        if 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == '0':
            logger.info("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def to(self, device):
        model = super().to(device)
        if hasattr(model, "bin_edges"):
            model.bin_edges = model.bin_edges.to(device)
        return model
        
    def get_num_params(self): 
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def resize_token_embeddings(self, new_num_tokens: int):
        """
        Grow (or shrink) both the input token embeddings and the LM head
        to `new_num_tokens`, copying over old weights and randomly
        initializing any new rows.
        """
        old_emb = self.transformer.wte.weight.data
        old_num, emb_dim = old_emb.shape

        # 1) New embedding layer
        new_emb = nn.Embedding(new_num_tokens, emb_dim)
        with torch.no_grad():
            new_emb.weight[:old_num].copy_(old_emb)
            # rows [old_num:] are randomly init per nn.Embedding default
        self.transformer.wte = new_emb

        # 2) New LM head (tied weights in many HF models)
        old_head = self.lm_head.weight.data  # shape (old_num, emb_dim)
        new_head = nn.Linear(emb_dim, new_num_tokens, bias=False)
        with torch.no_grad():
            new_head.weight[:old_num].copy_(old_head)
        self.lm_head = new_head

        old_head2 = self.epx_head.weight.data  # shape (old_num, emb_dim)
        new_head2 = nn.Linear(emb_dim, new_num_tokens, bias=False)
        with torch.no_grad():
            new_head2.weight[:old_num].copy_(old_head2)
        self.epx_head = new_head2

        # 3) Update config
        self.config.vocab_size = new_num_tokens

        self.epx_regressor = nn.Sequential(
                nn.Linear(self.config.vocab_size, self.config.n_embd),  # map vocab→hidden
                nn.ReLU(),
                nn.Linear(self.config.n_embd, 1)                   # hidden→scalar
            )

    def forward(self, 
                idx=None,
                inputs_embeds=None,
                targets=None,     # token IDs for attr‐prediction (shape [B])
                xlen=None,
                x_prefix_len=None,
                x_expr=None,      # input expression for embedding head
                y_expr=None,      # true expression values for regression loss (shape [B])
                lambda_val: float = 1.0,
                return_hidden=False,
    ):
        
        if idx is not None:
            b, t = idx.size()
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        if inputs_embeds is not None:
            tok_emb = inputs_embeds

        expr_emb = self.transformer.wee(x_expr) # expression embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb + expr_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits_labels = self.lm_head(x) 
        logits_exp = self.epx_head(x).squeeze(-1)
        loss = None
        loss_cls = None
        loss_exp = None
        
       
        # 2) Classification loss over all tokens (causal shift)
        if targets is not None:
            B, T, V = logits_labels.shape
            # shift so that at step t we predict targets[:,t] from x[:,t-1]
            shift_logits = logits_labels[:, :-1, :].reshape(-1, V)   # (B*(T-1), V)
            shift_labels = targets[:, 1:].reshape(-1)             # (B*(T-1))
            loss_cls = F.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=-100
            )

        B, T, V = logits_exp.size()
        flat_exp = logits_exp.view(-1, V)             # (B*T, V)
        flat_reg = self.epx_regressor(flat_exp)       # (B*T, 1)
        logits_reg = flat_reg.view(B, T, 1)           # (B, T, 1)

        # 3) Expression loss over all tokens (aligned or shifted as needed)
        if y_expr is not None:
            # If you want causal‐style (predict y_expr[:,t] from x[:,t-1]):
            #   pred_vals = logits_exp[:, :-1].reshape(-1)
            #   true_vals = y_expr[:, 1:].reshape(-1)
            # Otherwise, to predict at each position:
            B, T, V = logits_exp.size()
            flat_exp = logits_exp.view(-1, V)             # (B*T, V)
            flat_reg = self.epx_regressor(flat_exp)       # (B*T, 1)
            logits_reg = flat_reg.view(B, T, 1)           # (B, T, 1)

            pred_vals = logits_reg.squeeze(-1)
            shift_pred = pred_vals[:,:-1].contiguous()
            # true_vals: ensure same shape
            true_vals = y_expr.squeeze(-1) if y_expr.dim()==3 else y_expr
            shift_true = true_vals[:,1:].contiguous()

            flat_pred = shift_pred.view(-1)
            flat_true = shift_true.view(-1)
            
            loss_exp = F.mse_loss(pred_vals, true_vals, reduction='mean')


        if (loss_cls is not None) and (loss_exp is not None):
            loss = loss_cls + lambda_val * loss_exp
        elif loss_cls is not None:
            loss = loss_cls
        elif loss_exp is not None:
            loss = lambda_val * loss_exp


        if return_hidden:
            return logits_labels, logits_reg, x
        else:
            return logits_labels, logits_reg, loss, loss_cls, loss_exp



    # @torch.no_grad()
    # def generate_cellGenesis(self, 
    #                   input_ids,
    #                   expression_level,
    #                   max_new_tokens, 
    #                   ignore_Idx = None, 
    #                   top_k = None, 
    #                   return_dict_in_generate = False, 
    #                   return_hidden = False,
    #                   gamma = 1):

    #     scores = ()
    #     hidden_states = ()
    #     while True:
    #         idx_cond = input_ids
    #         if return_hidden:
    #             logits_cls,logits_exp,hidden = self(idx = idx_cond, x_expr = expression_level, return_hidden = True)
    #             hidden_states += (hidden,)
    #         else:
    #             logits_cls,logits_exp,_,_,_ = self(idx = idx_cond, x_expr = expression_level)

    #         logits_cls = logits_cls[:,-1,:] # (B,C)
    #         logits_exp = logits_exp[:,-1,:] # (B,C)

    #         if ignore_Idx is not None:
    #             # return logits, ignore_Idx
    #             logits_cls[:,ignore_Idx] = float('-inf')
    #         logits_cls[:,input_ids[0,:-1]] = float('-inf')
    #         if top_k is not None:
    #             v, _ = torch.topk(logits_cls, min(top_k, logits_cls.size(-1)))
    #             logits_cls[logits_cls < v[:, [-1]]] = float('-inf')
            
    #         next_token_scores = logits_cls

    #         if return_dict_in_generate:
    #             scores += (next_token_scores,)

    #         probs = F.softmax(logits_cls, dim=-1) #(B,C)
    #         probs[:,0] = gamma*probs[:,0]
    #         next_tokens = torch.multinomial(probs,num_samples=1) #(B,1)
    #         next_token_ele = logits_exp[torch.arange(logits_exp.size(0)),next_tokens.squeeze(1)].unsqueeze(1) # (B,1)
    #         bin_ele_next_token = torch.clamp(torch.round(next_token_ele), 0, 10).int()
    #         input_ids = torch.cat((input_ids,next_tokens),dim=1)
    #         expression_level = torch.cat((expression_level,bin_ele_next_token),dim=1)
    #         # check break condition
    #         if next_tokens == 0 or len(input_ids[0]) >= max_new_tokens:
    #             break
            
    #     if return_dict_in_generate:
    #         return SampleDecoderOutput(
    #             sequences=input_ids,
    #             scores=scores,
    #             hidden_states=hidden_states,
    #             expression=expression_level,
    #             )
    #     elif return_hidden:
    #         return input_ids,expression_level, hidden_states
    #     else:return input_ids, expression_level

    @torch.no_grad()
    def generate_cellGenesis(
        self,
        input_ids: torch.LongTensor,
        expression_level: torch.LongTensor,
        max_new_tokens: int,
        ignore_Idx=None,
        top_k=None,
        return_dict_in_generate=False,
        return_hidden=False,
        gamma: float = 1.0,
    ):
        """
        Autoregressively generate gene tokens *and* real‐valued expression
        using your MLP regression head on top of the pretrained categorical head.
        """
        scores = ()           # if return_dict_in_generate
        hidden_states = ()    # if return_hidden
        real_expr = torch.tensor(expression_level.clone(), dtype=torch.float)
    
        # loop until EOS (token 0) or length
        while True:
            # 1) forward one step
            if return_hidden:
                logits_cls, logits_exp, hidden = self(
                    idx=input_ids,
                    x_expr=expression_level,
                    return_hidden=True
                )
                hidden_states += (hidden,)
            else:
                logits_cls, logits_exp, _, _, _ = self(
                    idx=input_ids,
                    x_expr=expression_level
                )
    
            # 2) grab the *last* time-step logits
            #    cls: (B, vocab_size), exp: (B,) real‐valued
            logits_cls = logits_cls[:, -1, :]           # (B, C)
            logits_exp = logits_exp[:, -1].float()      # (B, 1)
    
            # 3) mask out forbidden/seen tokens
            if ignore_Idx is not None:
                logits_cls[:, ignore_Idx] = float('-inf')
            seen = input_ids[0, :-1]
            logits_cls[:, seen] = float('-inf')
    
            # 4) top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits_cls, min(top_k, logits_cls.size(-1)), dim=-1)
                logits_cls[logits_cls < v[:, [-1]]] = float('-inf')
    
            if return_dict_in_generate:
                scores += (logits_cls,)
    
            # 5) sample next token
            probs       = F.softmax(logits_cls, dim=-1)
            probs[:, 0] = gamma * probs[:, 0]       # optional weighting for EOS
            next_token  = torch.multinomial(probs, num_samples=1)  # (B,1)
    
            # 6) take its real expression prediction
            #    here logits_exp is real, so pick the entry for the sampled token?
            #    **But** now that logits_exp is scalar, we just use it directly:
            next_expr_real = logits_exp  # (B,1)

            bin_next = torch.bucketize(next_expr_real, self.bin_edges)  # (B,)

            # optionally clamp to [0..n_bins]:
            bin_next = torch.clamp(bin_next, 0, self.bin_edges.numel()-1)  # (B,1)
    
            # print(expression_level.shape, bin_next.shape, next_expr_real.shape)    
            # 8) append to sequence
            input_ids        = torch.cat([input_ids, next_token], dim=1)
            expression_level = torch.cat([expression_level, bin_next], dim=1)
            real_expr = torch.cat([real_expr, next_expr_real], dim=1)
            # 9) stop if EOS or too long
            if (next_token == 0).all() or input_ids.size(1) >= max_new_tokens:
                break
    
        if return_dict_in_generate:
            return SampleDecoderOutput(
                sequences=input_ids,
                scores=scores,
                hidden_states=hidden_states,
                expression=expression_level,
            )
        elif return_hidden:
            return input_ids, expression_level, real_expr, hidden_states
        else:
            return input_ids, expression_level, real_expr

@dataclass
class SampleDecoderOutput(SampleDecoderOnlyOutput):

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    expression: Optional[torch.LongTensor] = None