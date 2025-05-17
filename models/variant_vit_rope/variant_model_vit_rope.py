"""
This code was originally obtained from:
https://github.com/facebookresearch/deit
and
https://github.com/meta-llama/codellama/blob/main/llama/model.py
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from einops import rearrange
import inspect

import torch
import torch.nn as nn
from functools import partial
from modules.layers_patch_embed import *
from modules.layers_ours import *
import torch.nn.functional as F

from timm.models.vision_transformer import    _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., isWithBias=True, activation = GELU()):
        super().__init__()
        print(f"inside MLP with isWithBias: {isWithBias} and activation {activation}")
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features, bias = isWithBias)
        self.act = activation
        self.fc2 = Linear(hidden_features, out_features, bias = isWithBias)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam


class RoPEAttention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)

        self.dim = dim
      
       
        v_weight = self.qkv.weight[dim*2:dim*3].view(dim, dim)
        self.v_proj = Linear(dim, dim, bias=qkv_bias)
        self.v_proj.weight.data = v_weight

 
        v_bias   = self.qkv.bias[dim*2:dim*3]
        self.v_proj.bias.data = v_bias

        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim, bias = True)
        self.proj_drop = Dropout(proj_drop)
        self.attn_activation = Softmax(dim=-1)


        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None
    
    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients
    
    def forward(self, x,freqs_cis):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
        q_new, k_new = q.clone(), k.clone()
        q_new[:, :, 1:], k_new[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
        q, k = q_new, k_new
        #done only for hook
        tmp = self.v_proj(x)
        #######


        self.save_v(v)

        dots = self.matmul1([q, k]) * self.scale
       
        attn = self.attn_activation(dots)
        attn = self.attn_drop(attn)

        self.save_attn(attn)
        attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def relprop(self, cam = None,cp_rule = False, **kwargs):

        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # attn = A*V
        (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
      
        cam1 = self.attn_activation.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        v_proj_map = cam_qkv[:,:,self.dim*2:]
        
        if cp_rule:
           
            return self.v_proj.relprop(v_proj_map, **kwargs) 
        else:
            return self.qkv.relprop(cam_qkv, **kwargs)
    

def safe_call(func, **kwargs):
    # Get the function's signature
    sig = inspect.signature(func)
    
    # Filter kwargs to only include parameters the function accepts
    filtered_kwargs = {
        k: v for k, v in kwargs.items() 
        if k in sig.parameters
    }
    
    # Call the function with only its compatible parameters
    return func(**filtered_kwargs)
    
class RoPE_Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=None, norm_layer=None,Attention_block = None,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = safe_call(partial(LayerNorm, eps=1e-6), normalized_shape= dim, bias = True ) 

        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
      
        self.norm2 = safe_call(partial(LayerNorm, eps=1e-6), normalized_shape= dim, bias = True ) 

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim )
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)))
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)))


        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x,freqs_cis):
        x1, x2 = self.clone1(x, 2)
     
        x = self.add1([x1, self.gamma_1 *  self.attn(self.norm1(x2), freqs_cis=freqs_cis)])
        x1, x2 = self.clone2(x, 2)
      
        x = self.add2([x1, self.gamma_2 *  self.mlp(self.norm2(x2))])
        return x
    
    def relprop(self, cam = None, cp_rule = False, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
       
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        gamma_rule = kwargs['gamma_rule']
        kwargs['gamma_rule'] = False
        cam2 = self.attn.relprop(cam2,cp_rule=cp_rule, **kwargs)
        kwargs['gamma_rule'] = gamma_rule

        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam



class rope_vit_models(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, global_pool=None,
                 block_layers = None, rope_theta=100.0, rope_mixed=False, use_ape=False,
                 Patch_layer=PatchEmbed,act_layer=GELU(),
                 Attention_block = None, Mlp_block=Mlp,
                dpr_constant=True,init_scale=1e-4,
                mlp_ratio_clstk = 4.0,**kwargs):
        super().__init__()
        
        self.dropout_rate = drop_rate

            
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, isWithBias=True)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)])
        

        
            
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = Linear(embed_dim, num_classes, bias = True)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.pool = IndexSelect()
        self.add = Add()

        self.inp_grad = None


        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        
        self.use_ape = use_ape
        if not self.use_ape:
            self.pos_embed = None            
        
        self.rope_mixed = rope_mixed
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.used_ADD = False
        
        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)
            
            freqs = []
            for i, _ in enumerate(self.blocks):
                freqs.append(
                    init_random_2d_freqs(dim=embed_dim // num_heads, num_heads=num_heads, theta=rope_theta)
                )
            freqs = torch.stack(freqs, dim=1).view(2, len(self.blocks), -1)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
            
            t_x, t_y = init_t_xy(end_x = img_size // patch_size, end_y = img_size // patch_size)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
        else:
            self.compute_cis = partial(compute_axial_cis, dim=embed_dim//num_heads, theta=rope_theta)
            
            freqs_cis = self.compute_cis(end_x = img_size // patch_size, end_y = img_size // patch_size)
            self.freqs_cis = freqs_cis



    def save_inp_grad(self,grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = Linear(self.embed_dim, num_classes) 

   
       

    def forward(self, x):

        B, C, H, W = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.use_ape:
            pos_embed = self.pos_embed
            if pos_embed.shape[-2] != x.shape[-2]:
                img_size = self.patch_embed.img_size
                patch_size = self.patch_embed.patch_size
                pos_embed = pos_embed.view(
                    1, (img_size[1] // patch_size[1]), (img_size[0] // patch_size[0]), self.embed_dim
                ).permute(0, 3, 1, 2)
                pos_embed = F.interpolate(
                    pos_embed, size=(H // patch_size[1], W // patch_size[0]), mode='bicubic', align_corners=False
                )
                pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            self.used_ADD=True
            print(x.shape)
            print(pos_embed.shape)

            x = self.add([x, pos_embed])
        
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.rope_mixed:
          
            if self.freqs_t_x.shape[0] != x.shape[1] - 1:
                t_x, t_y = init_t_xy(end_x = W // self.patch_size, end_y = H // self.patch_size)
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            else:
                t_x, t_y = self.freqs_t_x, self.freqs_t_y
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
            
            for i , blk in enumerate(self.blocks):
               
                x = blk(x, freqs_cis=freqs_cis[i])
        else:
            if self.freqs_cis.shape[0] != x.shape[1] - 1:
                freqs_cis = self.compute_cis(end_x = W // self.patch_size, end_y = H // self.patch_size)
            else:
                freqs_cis = self.freqs_cis
            freqs_cis = freqs_cis.to(x.device)
            
            for i , blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis)
                
        x = self.norm(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))

        
    
        
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = x.squeeze(1)
        
        x = self.head(x)
        
        return x


    
    def relprop(self, cam=None,method="transformer_attribution", cp_rule = False, conv_prop_rule = None, is_ablation=False, start_layer=0, **kwargs):
        # print(kwargs)
        # print("conservation 1", cam.sum())
        cam = self.head.relprop(cam, **kwargs)
        #print(cam.shape)

        cam = cam.unsqueeze(1)
   
        cam = self.pool.relprop(cam, **kwargs)
     
        cam = self.norm.relprop(cam, **kwargs)
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam,cp_rule = cp_rule, **kwargs)

        # print("conservation 2", cam.sum())
        # print("min", cam.min())

        if "custom_lrp" in method:
            if self.used_ADD ==True:
                if cam.shape[1] == 197:
                    cam = cam[:, 1:, :]

                (sem_cam, pos_cam) = self.add.relprop(cam, **kwargs)

            ''' if "PE_ONLY" not in method and "SEMANTIC_ONLY" not in method:
                #print("SDFSDF")
                #exit(1)
                cam = pos_cam

                
                #sem_cam = sem_cam[:, 1:, :]
                #pos_cam = pos_cam[:, 1:, :]
#
                #norms1 = torch.norm(sem_cam, p=2, dim=-1)  # Shape: [196]
                #norms2 = torch.norm(pos_cam, p=2, dim=-1)  # Shape: [196]
                #
                #norms1 = (norms1 - norms1.min()) / (norms1.max() - norms1.min())
                #norms2 = (norms2 - norms2.min()) / (norms2.max() - norms2.min())
                #return norms1+norms2'''
            
            if "PE_ONLY" in method:
                cam = pos_cam
            if "SEMANTIC_ONLY" in method:
                #print("semantic")
                cam = sem_cam
            if cam.shape[1] == 197:
                cam = cam[:, 1:, :]
            #FIXME: slight tradeoff between noise and intensity of important features
           
            
            norms = torch.norm(cam, p=2, dim=-1)  # Shape: [196]
            return norms

        elif "full" in method:
          
            (cam, pos_cam) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:, :]
            #dont forget to change cp and change normalization layers
            #cam = cam.clamp(min=0)

            cam = self.patch_embed.relprop(cam, conv_prop_rule, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)

            if 'POS_GRAD_ENC' in method:
               
                grad_input_pos = torch.abs(torch.matmul(torch.abs(self.pos_embed) , pos_cam.transpose(-1, -2)) ) 
            
                grad_input_pos = grad_input_pos[:, 1:, 1:]
                grad_input_pos /=(16*16)
                
                grad_input_pos = torch.diagonal(grad_input_pos,offset=0, dim1=1, dim2=2).reshape(grad_input_pos.shape[0],
                     (224 // 16), (224 // 16))
                
                
               
                pe_att = torch.zeros(grad_input_pos.shape[0], 224, 224).to(grad_input_pos.device)
                for i in range(grad_input_pos.shape[-2]):
                    for j in range(grad_input_pos.shape[-1]):
                        value = grad_input_pos[:, i, j]
                        start_i = i * 16
                        start_j = j * 16
                        value = value.view(-1, 1, 1)  # shape: [32, 1, 1]

                        pe_att[:, start_i:start_i+16, start_j:start_j+16] = value 
                


                return (pe_att+cam).clamp(min=0)
                
                
            
            if 'POS_ENC' in method:
                
                pos_cam = pos_cam[:, 1:, :]
                pos_cam = pos_cam.sum(dim=2, keepdim=True)
                pos_cam = pos_cam.transpose(1,2)
                pos_cam = pos_cam.reshape(pos_cam.shape[0],
                                          
                     (224 // 16), (224 // 16))
                
                pos_cam /= (16*16)
                pe_att = torch.zeros(pos_cam.shape[0], 224, 224).to(pos_cam.device)
                for i in range(pos_cam.shape[-2]):
                    for j in range(pos_cam.shape[-1]):
                        value = pos_cam[:, i, j]
                        start_i = i * 16
                        start_j = j * 16
                        value = value.view(-1, 1, 1)  # shape: [32, 1, 1]

                        pe_att[:, start_i:start_i+16, start_j:start_j+16] = value 
                if "PE_ONLY" in method:
                    
                    return (pe_att).clamp(min=0)
                return (pe_att+cam).clamp(min=0)

            cam = cam.clamp(min=0)
           
            return cam

        

def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    depth = freqs.shape[1]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
                    
    return freqs_cis


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)
'''
class rope_vit_models(vit_models):
    def __init__(self, rope_theta=100.0, rope_mixed=False, use_ape=False,
                 **kwargs):
        super().__init__(**kwargs)
        
        img_size = kwargs['img_size'] if 'img_size' in kwargs else 224
        patch_size = kwargs['patch_size'] if 'patch_size' in kwargs else 16
        num_heads = kwargs['num_heads'] if 'num_heads' in kwargs else 12
        embed_dim = kwargs['embed_dim'] if 'embed_dim' in kwargs else 768
        mlp_ratio = kwargs['mlp_ratio'] if 'mlp_ratio' in kwargs else 4.
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        
        self.use_ape = use_ape
        if not self.use_ape:
            self.pos_embed = None            
        
        self.rope_mixed = rope_mixed
        self.num_heads = num_heads
        self.patch_size = patch_size
        
        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)
            
            freqs = []
            for i, _ in enumerate(self.blocks):
                freqs.append(
                    init_random_2d_freqs(dim=embed_dim // num_heads, num_heads=num_heads, theta=rope_theta)
                )
            freqs = torch.stack(freqs, dim=1).view(2, len(self.blocks), -1)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
            
            t_x, t_y = init_t_xy(end_x = img_size // patch_size, end_y = img_size // patch_size)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
        else:
            self.compute_cis = partial(compute_axial_cis, dim=embed_dim//num_heads, theta=rope_theta)
            
            freqs_cis = self.compute_cis(end_x = img_size // patch_size, end_y = img_size // patch_size)
            self.freqs_cis = freqs_cis
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'freqs'}
        
    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.use_ape:
            pos_embed = self.pos_embed
            if pos_embed.shape[-2] != x.shape[-2]:
                img_size = self.patch_embed.img_size
                patch_size = self.patch_embed.patch_size
                pos_embed = pos_embed.view(
                    1, (img_size[1] // patch_size[1]), (img_size[0] // patch_size[0]), self.embed_dim
                ).permute(0, 3, 1, 2)
                pos_embed = F.interpolate(
                    pos_embed, size=(H // patch_size[1], W // patch_size[0]), mode='bicubic', align_corners=False
                )
                pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            x = x + pos_embed
        
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.rope_mixed:
            if self.freqs_t_x.shape[0] != x.shape[1] - 1:
                t_x, t_y = init_t_xy(end_x = W // self.patch_size, end_y = H // self.patch_size)
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            else:
                t_x, t_y = self.freqs_t_x, self.freqs_t_y
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
            
            for i , blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis[i])
        else:
            if self.freqs_cis.shape[0] != x.shape[1] - 1:
                freqs_cis = self.compute_cis(end_x = W // self.patch_size, end_y = H // self.patch_size)
            else:
                freqs_cis = self.freqs_cis
            freqs_cis = freqs_cis.to(x.device)
            
            for i , blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis)
                
        x = self.norm(x)
        x = x[:, 0]
        
        return x
'''
def hf_checkpoint_load(model_name):
  
    try:
        from huggingface_hub import hf_hub_download
       

        ckpt_path = hf_hub_download(
            repo_id="naver-ai/" + model_name, filename= "pytorch_model.bin"
        )

        checkpoint = torch.load(ckpt_path, map_location='cpu')
    except:
        _HF_URL = "https://huggingface.co/naver-ai/" + model_name + "/resolve/main/pytorch_model.bin"
        checkpoint = torch.hub.load_state_dict_from_url(_HF_URL, map_location='cpu')
     
    state_dict = checkpoint['model']
    for k in ['freqs_t_x', 'freqs_t_y']:
        if k in state_dict:
            print(f"Removing key {k} from pretrained checkpoint")
            del state_dict[k]
            
    return state_dict

def adjust_pos_embed_size(model, state_dict):
    
    # interpolate position embedding
    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        state_dict['pos_embed'] = new_pos_embed
        
    return state_dict

# RoPE-Axial
@register_model
def rope_axial_deit_small_patch16_LS(pretrained=False, img_size=224,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6), block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=100.0, rope_mixed=False, **kwargs)
    model.default_cfg = _cfg()
    
    if pretrained:
        state_dict = hf_checkpoint_load("rope_axial_deit_small_patch16_LS")
        model.load_state_dict(state_dict, strict=False)
        
    return model

@register_model
def rope_axial_deit_base_patch16_LS(pretrained=False, img_size=224,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=100.0, rope_mixed=False, **kwargs)
    
    if pretrained:
        state_dict = hf_checkpoint_load("rope_axial_deit_base_patch16_LS")
        model.load_state_dict(state_dict, strict=False)
        
    return model

@register_model
def rope_axial_deit_large_patch16_LS(pretrained=False, img_size=224,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=100.0, rope_mixed=False, **kwargs)
    
    if pretrained:
        state_dict = hf_checkpoint_load("rope_axial_deit_large_patch16_LS")
        model.load_state_dict(state_dict, strict=False)
        
    return model

# RoPE-Mixed
@register_model
def rope_mixed_deit_small_patch16_LS(pretrained=False, img_size=224,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6), block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=10.0, rope_mixed=True, **kwargs)
    model.default_cfg = _cfg()
    
    if pretrained:
        state_dict = hf_checkpoint_load("rope_mixed_deit_small_patch16_LS")
        model.load_state_dict(state_dict, strict=False)
        print("STOPPP")
    
    return model

@register_model
def rope_mixed_deit_base_patch16_LS(pretrained=False, img_size=224,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=10.0, rope_mixed=True, **kwargs)
    
    if pretrained:
        state_dict = hf_checkpoint_load("rope_mixed_deit_base_patch16_LS")
        model.load_state_dict(state_dict, strict=False)
        
    return model

@register_model
def rope_mixed_deit_large_patch16_LS(pretrained=False, img_size=224,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=10.0, rope_mixed=True, **kwargs)
    
    if pretrained:
        state_dict = hf_checkpoint_load("rope_mixed_deit_large_patch16_LS")
        model.load_state_dict(state_dict, strict=False)
        
    return model


# RoPE-Axial + APE
@register_model
def rope_axial_ape_deit_small_patch16_LS(pretrained=False, img_size=224,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6), block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=100.0, rope_mixed=False, use_ape=True, **kwargs)
    model.default_cfg = _cfg()
    
    if pretrained:
        state_dict = hf_checkpoint_load("rope_axial_ape_deit_small_patch16_LS")
        state_dict = adjust_pos_embed_size(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
        
    return model

@register_model
def rope_axial_ape_deit_base_patch16_LS(pretrained=False, img_size=224,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=100.0, rope_mixed=False, use_ape=True, **kwargs)
    
    if pretrained:
        state_dict = hf_checkpoint_load("rope_axial_ape_deit_base_patch16_LS")
        state_dict = adjust_pos_embed_size(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
        
    return model

@register_model
def rope_axial_ape_deit_large_patch16_LS(pretrained=False, img_size=224,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=100.0, rope_mixed=False, use_ape=True, **kwargs)
    
    if pretrained:
        state_dict = hf_checkpoint_load("rope_axial_ape_deit_large_patch16_LS")
        state_dict = adjust_pos_embed_size(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
        
    return model

# RoPE-Mixed + APE
@register_model
def rope_mixed_ape_deit_small_patch16_LS(pretrained=False, img_size=224,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6), block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=10.0, rope_mixed=True, use_ape=True, **kwargs)
    model.default_cfg = _cfg()
    
    if pretrained:
        state_dict = hf_checkpoint_load("rope_mixed_ape_deit_small_patch16_LS")
        state_dict = adjust_pos_embed_size(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
        
    return model

@register_model
def rope_mixed_ape_deit_base_patch16_LS(pretrained=False, img_size=224,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=10.0, rope_mixed=True, use_ape=True, **kwargs)
    
    if pretrained:
        state_dict = hf_checkpoint_load("rope_mixed_ape_deit_base_patch16_LS")
        state_dict = adjust_pos_embed_size(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
        
    return model

@register_model
def rope_mixed_ape_deit_large_patch16_LS(pretrained=False, img_size=224,  **kwargs):
    model = rope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6),block_layers=RoPE_Layer_scale_init_Block, Attention_block=RoPEAttention,
        rope_theta=10.0, rope_mixed=True, use_ape=True, **kwargs)
    
    if pretrained:
        state_dict = hf_checkpoint_load("rope_mixed_ape_deit_large_patch16_LS")
        state_dict = adjust_pos_embed_size(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
        
    return model