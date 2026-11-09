import gzip
import math
import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
IMG = 64
PATTERNS = ["stripes", "checks", "dots", "zigzag", "gradient", "blobs"]
N_PAT = len(PATTERNS)
# style vector = one-hot pattern (6) + colour1 RGB (3) + colour2 RGB (3) + angle (1) + scale (1)
STYLE_DIM = N_PAT + 3 + 3 + 2

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
GARMENT_CLASSES = [0, 1, 2, 3, 4, 6]   # clothes only (no shoes / bags)

FASHION_MIRRORS = [
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
]
FASHION_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


# ----------------------------------------------------------------------------
# Data download
# ----------------------------------------------------------------------------
def _download(fname, root):
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, fname)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    last_err = None
    for base in FASHION_MIRRORS:
        try:
            print("downloading", base + fname)
            tmp = path + ".part"
            urllib.request.urlretrieve(base + fname, tmp)
            os.replace(tmp, path)
            return path
        except Exception as e:  # try next mirror
            last_err = e
    raise RuntimeError("Could not download %s from any mirror: %s" % (fname, last_err))


def load_fashion_mnist(root="data"):
    """Downloads Fashion-MNIST (if needed) and returns uint8 arrays."""
    def read_images(p):
        with gzip.open(p, "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    def read_labels(p):
        with gzip.open(p, "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    xtr = read_images(_download(FASHION_FILES["train_images"], root))
    ytr = read_labels(_download(FASHION_FILES["train_labels"], root))
    xte = read_images(_download(FASHION_FILES["test_images"], root))
    yte = read_labels(_download(FASHION_FILES["test_labels"], root))
    return xtr, ytr, xte, yte


def to_gray_tensor(images_uint8, size=IMG):
    """uint8 (N,28,28) -> float (N,1,size,size) in [0,1]."""
    x = torch.from_numpy(np.ascontiguousarray(images_uint8)).float().div(255.0).unsqueeze(1)
    out = []
    for i in range(0, len(x), 4096):
        out.append(F.interpolate(x[i:i + 4096], size=(size, size), mode="bilinear", align_corners=False))
    return torch.cat(out)


# ----------------------------------------------------------------------------
# Masks, patterns and the styled-clothing renderer
# ----------------------------------------------------------------------------
def clean_mask(m, k=5):
    """Morphological closing to fill small holes. m: (B,1,H,W) float 0/1."""
    p = k // 2
    m = F.max_pool2d(m, k, 1, p)
    m = -F.max_pool2d(-m, k, 1, p)
    return m


def mask_from_gray(gray, thr=0.06):
    return clean_mask((gray > thr).float())


def sample_style(B, device="cpu", pattern=None, colors=None):
    """Random style codes. pattern: int index or None; colors: ((r,g,b),(r,g,b)) in 0..1 or None."""
    if pattern is None:
        kind = torch.randint(0, N_PAT, (B,), device=device)
    else:
        kind = torch.full((B,), int(pattern), device=device, dtype=torch.long)
    onehot = F.one_hot(kind, N_PAT).float()
    if colors is None:
        c1 = torch.rand(B, 3, device=device)
        c2 = (c1 + 0.3 + 0.4 * torch.rand(B, 3, device=device)) % 1.0
    else:
        c1 = torch.tensor(colors[0], device=device, dtype=torch.float32).expand(B, 3)
        c2 = torch.tensor(colors[1], device=device, dtype=torch.float32).expand(B, 3)
    angle = torch.rand(B, 1, device=device)
    scale = torch.rand(B, 1, device=device)
    return torch.cat([onehot, c1, c2, angle, scale], 1)


def pattern_mix(s, H, W):
    """Pattern field in [0,1], shape (B,1,H,W), fully determined by the style code."""
    B = s.shape[0]
    kind = s[:, :N_PAT].argmax(1)
    theta = s[:, 12].view(B, 1, 1) * math.pi
    freq = 3.0 + s[:, 13].view(B, 1, 1) * 6.0
    ys, xs = torch.meshgrid(
        torch.linspace(-0.5, 0.5, H, device=s.device),
        torch.linspace(-0.5, 0.5, W, device=s.device),
        indexing="ij",
    )
    xs, ys = xs[None], ys[None]
    u = xs * torch.cos(theta) + ys * torch.sin(theta)
    v = -xs * torch.sin(theta) + ys * torch.cos(theta)
    tp = 2 * math.pi

    su = torch.sin(tp * freq * u)
    sv = torch.sin(tp * freq * v)
    stripes = (su > 0).float()
    checks = ((su > 0) ^ (sv > 0)).float()
    dots = ((su * sv) > 0.45).float()
    tri = freq * v
    tri = 2 * torch.abs(tri - torch.floor(tri) - 0.5)
    zigzag = (torch.sin(tp * freq * (u + 0.12 * tri)) > 0).float()
    gradient = ((u + 0.7) / 1.4).clamp(0, 1)
    blobs = ((torch.cos(tp * freq * 0.6 * u + 1.0)
              + torch.cos(tp * freq * 0.5 * v + 2.0)
              + torch.cos(tp * freq * 0.4 * (u + v))) > 0.3).float()

    stack = torch.stack([stripes, checks, dots, zigzag, gradient, blobs], 1)  # (B,6,H,W)
    onehot = F.one_hot(kind, N_PAT).float().view(B, N_PAT, 1, 1)
    return (stack * onehot).sum(1, keepdim=True)


def render_styled(gray, mask, s):
    """Paint a garment (shading from `gray`, silhouette from `mask`) with style `s`. Output in [-1,1]."""
    B, _, H, W = gray.shape
    mix = pattern_mix(s, H, W)
    c1 = s[:, 6:9].view(B, 3, 1, 1)
    c2 = s[:, 9:12].view(B, 3, 1, 1)
    color = c1 * (1 - mix) + c2 * mix
    g = gray / (gray.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
    shade = (0.55 + 0.6 * g).clamp(0, 1)
    img = color * shade * mask + (1 - mask)       # white background
    return img * 2 - 1


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def _down(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True))


def _up(i, o):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


def _tile(s, hw):
    return s[:, :, None, None].expand(-1, -1, hw[0], hw[1])


class Generator(nn.Module):
    """U-Net: (source clothing image, silhouette mask, style code) -> restyled clothing image."""

    def __init__(self, style_dim=STYLE_DIM, nf=64):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(3 + 1 + style_dim, nf, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.e2 = _down(nf, nf * 2)
        self.e3 = _down(nf * 2, nf * 4)
        self.e4 = _down(nf * 4, nf * 4)
        self.bott = nn.Sequential(nn.Conv2d(nf * 4 + style_dim, nf * 4, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(nf * 4), nn.ReLU(True))
        self.u4 = _up(nf * 8, nf * 4)
        self.u3 = _up(nf * 8, nf * 2)
        self.u2 = _up(nf * 4, nf)
        self.u1 = nn.ConvTranspose2d(nf * 2, 3, 4, 2, 1)

    def forward(self, src, mask, s):
        x = torch.cat([src, mask, _tile(s, src.shape[-2:])], 1)
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        b = self.bott(torch.cat([e4, _tile(s, e4.shape[-2:])], 1))
        d4 = self.u4(torch.cat([b, e4], 1))
        d3 = self.u3(torch.cat([d4, e3], 1))
        d2 = self.u2(torch.cat([d3, e2], 1))
        out = torch.tanh(self.u1(torch.cat([d2, e1], 1)))
        return out * mask + (1 - mask)          # keep the silhouette, white background


class Discriminator(nn.Module):
    """Image -> one logit. sigmoid(logit) is the 'realness' score (1 = real, 0 = generated)."""

    def __init__(self, nf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, nf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(nf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(nf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(nf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, 1, 4, 1, 0),
        )

    def forward(self, x):
        return self.net(x).view(-1)

    @torch.no_grad()
    def score(self, x):
        return torch.sigmoid(self.forward(x))


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# ----------------------------------------------------------------------------
# Inference helpers (used by the notebook and by the web app)
# ----------------------------------------------------------------------------
def prepare_input(pil_img, size=IMG):
    """PIL image -> (source [-1,1] (3,H,W), mask (1,H,W)). Background is estimated from the image border."""
    img = pil_img.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]], 0)
    bg = np.median(border, 0)
    dist = np.abs(arr - bg).max(-1)
    m = torch.from_numpy((dist > 0.15).astype(np.float32))[None, None]
    m = clean_mask(m, 5)
    if m.mean() < 0.02:                      # nothing detected -> treat whole image as garment
        m = torch.ones_like(m)
    x = torch.from_numpy(arr).permute(2, 0, 1)[None]
    x = x * m + (1 - m)
    return (x * 2 - 1)[0], m[0]


@torch.no_grad()
def generate_variants(G, D, pil_img, n=6, pattern=None, colors=None, device="cpu"):
    G.eval()
    D.eval()
    src, mask = prepare_input(pil_img)
    src = src[None].to(device)
    mask = mask[None].to(device)
    s = sample_style(n, device, pattern=pattern, colors=colors)
    out = G(src.expand(n, -1, -1, -1), mask.expand(n, -1, -1, -1), s)
    scores = D.score(out).cpu().tolist()
    src_score = float(D.score(src).item())
    return src[0].cpu(), out.cpu(), s.cpu(), scores, src_score


def describe_style(s_row):
    name = PATTERNS[int(s_row[:N_PAT].argmax())]
    def hx(c):
        return "#%02x%02x%02x" % tuple(int(round(float(v) * 255)) for v in c)
    return name, hx(s_row[6:9]), hx(s_row[9:12])


def to_np(t):
    """(3,H,W) in [-1,1] -> (H,W,3) float in [0,1]."""
    return ((t.detach().cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()


def to_pil(t, upscale=256):
    arr = (to_np(t) * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((upscale, upscale), Image.BICUBIC)


def best_threshold(real, fake):
    """Threshold on the realness score that maximises balanced accuracy."""
    real, fake = np.asarray(real), np.asarray(fake)
    best_t, best_acc = 0.5, 0.0
    for t in np.unique(np.concatenate([real, fake])):
        acc = 0.5 * ((real >= t).mean() + (fake < t).mean())
        if acc > best_acc:
            best_t, best_acc = float(t), float(acc)
    return best_t, best_acc


def verdict(score, threshold):
    return "REAL" if score >= threshold else "GENERATED"
