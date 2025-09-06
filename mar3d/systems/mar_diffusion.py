from dataclasses import dataclass, field
import os
import numpy as np
import json
import copy
import torch
import torch.nn.functional as F
from skimage import measure
from einops import repeat
from tqdm import tqdm
from PIL import Image

from transformers import CLIPImageProcessor, CLIPTokenizer
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
    KarrasVeScheduler,
    DPMSolverMultistepScheduler
)
import scipy.stats as stats

from plyfile import PlyData, PlyElement
import mar3d
from mar3d.systems.base import BaseSystem
from mar3d.utils.ops import generate_dense_grid_points
from mar3d.utils.typing import *
import torchvision
import pandas as pd
import math
from timm.models.vision_transformer import Block
from mar3d.systems.diffloss import DiffLoss
import torch.nn as nn


class MAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, latent_size=256, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=64,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=256,
                 diffloss_d=6,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim
        # self.vae_stride = vae_stride
        self.patch_size = patch_size
        # self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len =latent_size// patch_size
        self.token_embed_dim = vae_embed_dim * patch_size
        self.grad_checkpointing = grad_checkpointing

    
        # --------------------------------------------------------------------------
        # Class Embedding
        # self.num_classes = class_num
        # self.class_emb = nn.Embedding(1000, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(buffer_size, encoder_embed_dim))

    
        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer( encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        # parameters
        # torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)

        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding):


        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # random drop class embedding during training
        if self.training:

            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask =drop_latent_mask .to(x.device)
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).to(x.dtype)
      
            class_embedding = drop_latent_mask[:,:,None] * self.fake_latent[None] + (1 - drop_latent_mask) [:,:,None]* class_embedding
      

        x[:, :self.buffer_size] = class_embedding

        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # dropping
        x = x[ (1-mask_with_buffer).nonzero(as_tuple=True)  ].reshape(bsz, -1, embed_dim)
        
        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask):

        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def forward(self, x, labels,low_res=None):

    
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        # mae encoder
        x = self.forward_mae_encoder(x, mask,labels)
    
        # mae decoder
        z = self.forward_mae_decoder(x, mask)
        # print(z.shape)
        # diffloss
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=True):

        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()
            if not cfg == 1.0:
                # ipdb.set_trace()
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([labels, self.fake_latent[None].repeat(bsz, 1,1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)
            # mae encoder
            x = self.forward_mae_encoder(tokens, mask, class_embedding)
            
            # mae decoder
            z = self.forward_mae_decoder(x, mask)
            # ipdb.set_trace()
            
            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            # ipdb.set_trace()
            # print(z.shape)   
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        return  tokens


@mar3d.register("mar-diffusion-system")
class ShapeDiffusionSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        data_type: str = None
        val_samples_json: str = None
        # shape vae model
        shape_model_type: str = None
        # condition model
        shape_model: dict = field(default_factory=dict)
        condition_model_type: str = None
        condition_model: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        super().configure()
        self.shape_model = mar3d.find(self.cfg.shape_model_type)(self.cfg.shape_model)
        self.shape_model.eval()
        self.sigma_min = 0.0
        self.condition = mar3d.find(self.cfg.condition_model_type)(self.cfg.condition_model)
        self.condition.requires_grad_(False) 
        self.mar=MAR(patch_size=1,latent_size=256,buffer_size=257) 
   
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:

        shape_embeds, kl_embed, posterior = self.shape_model.encode(
            batch["surface"][..., :3 + self.cfg.shape_model.point_feats], 
            sample_posterior=True)

        latents=kl_embed
        cond_latents = self.condition(batch).to(latents.device)

        loss=self.mar(latents,cond_latents)
        return {
            "loss_diffusion": loss,
            "latents": latents,
            }

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = 0.
        for name, value in out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            if name.startswith("lambda_"):
                self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        oct_num=9
        with torch.no_grad():

            cond = self.condition.encode_image(batch['image'])

            
            latents = self.mar.sample_tokens(bsz=cond.shape[0], num_iter=64, cfg=3.0 ,
                                                                        cfg_schedule="linear", labels=cond, temperature=1.0)
            output=self.shape_model.decode(latents )
      
            mesh_v_f, has_surface = self.shape_model.extract_geometry(output, octree_depth=oct_num)
            for j in range(len(mesh_v_f)):
                self.save_mesh(
                                f"it{self.true_global_step}/{batch['uid'][0]}_cfg{3.0}_oct{oct_num}.obj",
                                mesh_v_f[j][0], mesh_v_f[j][1]
                            )
                        
        self.save_image(f"it{self.true_global_step}/{batch['uid'][0]}_gt.jpg",  (batch['image']*255).int() )
       
        torch.cuda.empty_cache()