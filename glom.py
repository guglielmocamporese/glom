import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torchmetrics.functional import accuracy


def patches2image_batch(x, output_size):
    B, C, N_PATCH, H_PATCH, W_PATCH = x.shape
    x = x.reshape(B, C, N_PATCH, -1).transpose(2, 3).reshape(B, -1, N_PATCH) # [B, C, prod(PATCH_SIZE)]
    x = F.fold(x, output_size, kernel_size=(H_PATCH, W_PATCH), stride=(H_PATCH, W_PATCH)) # [B, C, H, W]
    return x # [B, C, H, W]

def attention(x):
    """
    Inputs:
        x: tensor of shape [B, N, D]
    Outputs:
        x: tensor of shape [B, N, D]
    """
    att = torch.bmm(x, x.transpose(1, 2))
    att = F.softmax(att, 1)
    x = torch.bmm(att, x)
    return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop) if isinstance(drop, float) else drop

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class GlomLayer(nn.Module):
    def __init__(self, n_levels=5, feat_dim=256, num_heads=8, **kwargs):
        super().__init__()
        self.bottom_up_net = Block(feat_dim, num_heads=num_heads, **kwargs)
        self.top_down_net = Block(feat_dim, num_heads=num_heads, **kwargs)
        self.current_net = attention

    def forward(self, x, pos_lev_emb):
        """
        Inputs:
            x: tensor of shape [B, N, NL, DL]
        Outputs:
            z: tensor of shape [B, N, NL, DL]
        """
        B, N, NL, DL = x.shape
        z = []
        for l in range(NL):
            x_lev_cur = x[:, :, l]
            x_lev_prev = x[:, :, l - 1] if l > 0 else torch.zeros_like(x_lev_cur) # [B, N, DL]
            x_lev_next = x[:, :, l + 1] if l < (NL - 1) else torch.zeros_like(x_lev_cur) # [B, N, DL]

            z_lev_prev = self.bottom_up_net(x_lev_prev) # [B, N, DL]
            z_lev_next = self.top_down_net(x_lev_next) + pos_lev_emb[l].unsqueeze(0).unsqueeze(0).repeat(B, N, 1) # [B, N, DL]
            z_lev_cur = self.current_net(x_lev_cur) + z_lev_prev + z_lev_next # [B, N, DL]
            z += [z_lev_cur]
        z = torch.stack(z, 2) # [B, N, NL, DL]
        return z

class Glom(nn.Module):
    """
    Glom Backbone.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, n_layers=3, n_levels=5, feat_dim=1280, use_pos_emb=False):
        super().__init__()
        assert feat_dim % n_levels == 0, 'Features dimension must be divisible by number of levels!'
        self.n_levels = n_levels
        self.feat_dim = feat_dim
        self.use_pos_emb = use_pos_emb
        self.feat_dim_per_level = self.feat_dim // self.n_levels
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=feat_dim)
        self.pos_lev_emb = nn.Parameter(torch.randn(n_levels, self.feat_dim_per_level))
        self.layers = nn.ModuleList([GlomLayer(n_levels=n_levels, feat_dim=self.feat_dim_per_level) for _ in range(n_layers)])
        if self.use_pos_emb:
            self.clf_token = nn.Parameter(torch.randn(1, self.feat_dim))

    def forward(self, x):
        """
        Inputs:
            x: tensor of shape [B, C, H, W]
        Outputs:
            x: tensor of shape [B, N, NL, DL]
        """
        _, C, H, W = x.shape
        x = self.patch_embed(x) # [B, N, D]
        B, N, D = x.shape
        if self.use_pos_emb:
            x = torch.cat([self.clf_token.unsqueeze(0).repeat(B, 1, 1), x], 1) # [B, N + 1, D]
        x = x.reshape(B, N, self.n_levels, self.feat_dim_per_level) # [B, N, NL, DL]
        for layer in self.layers:
            x = layer(x, self.pos_lev_emb)
        return x # [B, N, NL, DL]


class GlomReconstruction(pl.LightningModule):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, n_layers=3, n_levels=5, feat_dim=1280):
        super().__init__()
        assert feat_dim % n_levels == 0, 'Features dimension must be divisible by number of levels!'
        self.backbone = Glom(img_size=img_size, patch_size=patch_size, in_chans=in_chans, n_layers=n_layers, 
                             n_levels=n_levels, feat_dim=feat_dim, use_pos_emb=False)
        self.rec_head = nn.Linear(self.feat_dim, in_chans * (patch_size ** 2))

    def forward(self, x):
        """
        Inputs:
            x: tensor of shape [B, C, H, W]
        Outputs:
            x: tensor of shape [B, N, NL, DL]
        """
        _, C, H, W = x.shape
        PS = self.backbone.patch_embed.patch_size # patch_size
        x = self.backbone(x) # [B, N, NL, DL]
        x = x.reshape(B, N, self.backbone.feat_dim) # [B, N, D]
        B, N, D = x.shape
        x = self.rec_head(x) # [B, N, C * PS * PS] 
        x = x.reshape(B, N, C, PS[0], PS[0]).transpose(1, 2)
        x = patches2image_batch(x, (H, W))
        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=3e-4)

    def training_step(self, batch, idx, mode='train'):
        x, y = batch
        x_ = self(x)

        loss = F.mse_loss(x_, x)
        self.log(f'{mode}_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, idx):
        return self.training_step(batch, idx, mode='val')


class GlomClassification(pl.LightningModule):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, n_layers=3, n_levels=5, feat_dim=1280, num_classes=10):
        super().__init__()
        self.backbone = Glom(img_size=img_size, patch_size=patch_size, in_chans=in_chans, n_layers=n_layers, 
                             n_levels=n_levels, feat_dim=feat_dim, use_pos_emb=True)
        self.clf_head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        """
        Inputs:
            x: tensor of shape [B, C, H, W]
        Outputs:
            x: tensor of shape [B, N, NL, DL]
        """
        B, C, H, W = x.shape
        x = self.backbone(x) # [B, N + 1, NL, DL]
        N = x.shape[1] - 1
        x = x.reshape(B, N + 1, self.feat_dim) # [B, N + 1, D]
        x = self.clf_head(x[:, 0]) # [B, N_CL]
        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=3e-4)

    def training_step(self, batch, idx, mode='train'):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)
        self.log(f'{mode}_loss', loss, prog_bar=True)
        self.log(f'{mode}_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, idx):
        return self.training_step(batch, idx, mode='val')
