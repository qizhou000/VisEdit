#%%
from typing import Dict, List, Tuple
from torch import nn
import torch
import numpy as np

class BaseResMapper(nn.Module):
    def __init__(self, in_dim, mid_dim, act_layer = nn.ReLU):
        super().__init__()
        self.l_in = nn.Linear(in_dim, mid_dim)
        self.act = act_layer()
        self.l_out = nn.Linear(mid_dim, in_dim)
        
    def reset_parameters(self):
        self.l_in.reset_parameters()
        self.l_out.reset_parameters()

    def forward(self, x):
        return self.l_out(self.act(self.l_in(x))) + x
    
class InfluenceMapper(nn.Module):
    def __init__(self, inpt_dim, mid_dim, att_head_n):
        super().__init__()
        self.ln_img_reps = nn.LayerNorm(inpt_dim) 
        self.ln_edit_reps = nn.LayerNorm(inpt_dim) 
        self.img_map = BaseResMapper(inpt_dim, mid_dim)
        self.prompt_token_map = BaseResMapper(inpt_dim, mid_dim)
        self.att_head_n = att_head_n
        self.scale = (inpt_dim // att_head_n) ** 0.5
    
    def reset_parameters(self):
        self.ln_img_reps.reset_parameters()
        self.ln_edit_reps.reset_parameters()
        self.img_map.reset_parameters()
        self.prompt_token_map.reset_parameters()

    def forward(self, img_reps, prompt_last_token_of_edit_reps):
        '''`img_reps`: [b, img_token_n, d], prompt_last_token_of_edit_reps: [b, d]'''
        b, tn, d = img_reps.shape
        img_reps = self.img_map(self.ln_img_reps(img_reps)) # [b, img_token_n, d]
        prompt_last_token_of_edit_reps = self.prompt_token_map(self.ln_edit_reps(prompt_last_token_of_edit_reps))
        img_reps = img_reps.reshape(b, tn, self.att_head_n, d//self.att_head_n) # [b,img_token_n,head,d//head]
        prompt_last_token_of_edit_reps = prompt_last_token_of_edit_reps.reshape(b, self.att_head_n, d//self.att_head_n)
        inf_map = torch.einsum('bihd,bhd->bih', img_reps, prompt_last_token_of_edit_reps) # [b, img_token_n, d//head]
        inf_map = inf_map.mean(2) / self.scale # [b, img_token_n]
        return inf_map

class VisionEditAdaptor(nn.Module):
    def __init__(self, hidden_size, mid_dim = 1024, cross_att_head_n = 8, 
                 img_tok_n = 576, add_it = False, infm_dim = 256) -> None:
        '''
        hidden_size: dimension of embeddings
        mid_dim: middle dimension of adaptor
        cross_att_head_n: head of cross attention
        '''
        super().__init__()
        if mid_dim % cross_att_head_n != 0: raise
        self.mid_dim = mid_dim
        self.cross_att_head_n = cross_att_head_n
        self.add_it = add_it
        self.img_tok_n = img_tok_n
        self.mlp_begin = nn.Linear(hidden_size, mid_dim)
        self.cross_att_q_mlp = nn.Linear(mid_dim, mid_dim)
        self.cross_att_k_mlp = nn.Linear(hidden_size, mid_dim)
        self.cross_att_v_mlp = nn.Linear(hidden_size, mid_dim)
        self.mlp_end = nn.Linear(mid_dim, hidden_size)
        self.ln_img_reps = nn.LayerNorm(hidden_size) 
        self.ln_edit_reps = nn.LayerNorm(hidden_size) 
        self.influence_mapper = InfluenceMapper(hidden_size, infm_dim, cross_att_head_n) 
        self.open_adaptor(False)
        self.set_edit_signal(None, None, None)

    def reset_parameters(self):
        self.mlp_begin.reset_parameters()
        self.cross_att_q_mlp.reset_parameters()
        self.cross_att_k_mlp.reset_parameters()
        self.cross_att_v_mlp.reset_parameters()
        self.mlp_end.reset_parameters()
        self.ln_img_reps.reset_parameters()
        self.ln_edit_reps.reset_parameters()
        self.influence_mapper.reset_parameters()

    def forward(self, layer_outpt):
        '''layer_outpt: [b, l, d]''' 
        if (not self.is_open 
            or layer_outpt.shape[1] == 1 # generate mode, which has attention cache 
            or not self.inpt_has_img # not has vision token in this input
            ): return layer_outpt
        if self.inpt_vt_begin == None or self.inpt_vt_end == None:
            raise BaseException('Have not set vision token range.')
        # get normed reps and determine legal
        img_reps = layer_outpt[:, self.inpt_vt_begin:self.inpt_vt_end].clone()
        b1, l1, _ = img_reps.shape # l1 = self.img_tok_n
        b2, l2, _ = self.edit_reps.shape 
        if l1 != self.img_tok_n: 
            raise BaseException('Number of selected vision token error.')
        if b1 != b2: 
            raise BaseException('Batch size of input and editing signal are not matched.')
        # introduce influence mapping
        if self.add_it:
            prompt_last_token_of_edit_reps = self.edit_reps[range(len(self.prompt_end)), self.prompt_end] # [b, d]
            inf_map = self.influence_mapper(img_reps, prompt_last_token_of_edit_reps) # [b, img_token_n]
            inf_map = torch.sigmoid(inf_map).unsqueeze(-1) # [b, img_token_n, 1]
        else:
            inf_map = 1
        # image representation transformation
        norm_img_reps = self.ln_img_reps(img_reps)  # [b,img_tok_n,d]
        norm_edit_reps = self.ln_edit_reps(self.edit_reps) # [b,l,d]
        x = self.mlp_begin(norm_img_reps)
        q = self.cross_att_q_mlp(x).reshape(b1, l1, self.cross_att_head_n, self.mid_dim//self.cross_att_head_n)
        k = self.cross_att_k_mlp(norm_edit_reps).reshape(b1, l2, self.cross_att_head_n, self.mid_dim//self.cross_att_head_n)
        v = self.cross_att_v_mlp(norm_edit_reps).reshape(b1, l2, self.cross_att_head_n, self.mid_dim//self.cross_att_head_n)
        s = torch.einsum('blhm,buhm->bhlu', q, k) # [batch_size, head_n, l1, l2]
        s = s / (self.mid_dim//self.cross_att_head_n)**0.5
        s = s + (self.edit_reps_att_mask.reshape(b1, 1, 1, l2) - 1)*9999999999
        s = torch.softmax(s, 3)
        x = torch.einsum('bhlu,buhm->blhm', s, v).reshape(b1, l1, self.mid_dim) # [batch_size, img_tok_n, mid_dim]
        x = self.mlp_end(x) * inf_map
        layer_outpt[:, self.inpt_vt_begin:self.inpt_vt_end] = img_reps + x
        return layer_outpt

    def open_adaptor(self, if_open:bool):
        self.is_open = if_open

    def set_edit_signal(self, edit_reps:torch.Tensor, edit_reps_att_mask:torch.Tensor, 
                           prompt_end:torch.Tensor):
        # edit_reps/edit_reps_att_mask: [b,l,d], prompt_end: [b]
        self.edit_reps = edit_reps
        self.edit_reps_att_mask = edit_reps_att_mask
        self.prompt_end = prompt_end
    
    def set_input_info(self, has_img = True, vt_begin:int = None, vt_end:int = None):
        '''Should be called every time input to the llm. '''
        self.inpt_has_img = has_img # whether has image in the input
        self.inpt_vt_begin = vt_begin # begin of vision token
        self.inpt_vt_end = vt_end # end of vision token






