#!/usr/bin/env python3
"""Lyric video "You Rock My World" para Catalina, mi estrellita.

Uso:
  python3 render.py                    # video completo -> out/catalina_you_rock_my_world.mp4
  python3 render.py --stills 5,30,70   # PNGs de prueba en out/stills/
  python3 render.py --start 60 --dur 10 --fps 30   # clip corto de prueba
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

ROOT = os.path.dirname(os.path.abspath(__file__))
A = lambda *p: os.path.join(ROOT, "assets", *p)
OUT = os.path.join(ROOT, "out")

W, H = 1920, 1080
FPS = 30
AUDIO = A("audio", "you_rock_my_world.mp3")
SONG_END = 266.29
VIDEO_END = 268.5

ORANGE = (255, 128, 24)
ORANGE_L = (255, 170, 80)
CREAM = (255, 244, 230)
INK = (58, 38, 28)

# ---------------------------------------------------------------- timing
LY = json.load(open(os.path.join(ROOT, "lyrics.json"), encoding="utf-8"))
BPM = LY["bpm"]
BEAT = 60.0 / BPM
OFFSET = LY["offset"]
BAR = 4 * BEAT


def bar_t(b):
    return OFFSET + b * BAR


def beat_pulse(t):
    """1 justo en el golpe, decae rapido. Mas fuerte en el primer tiempo del compas."""
    if t < OFFSET:
        return 0.0
    k = (t - OFFSET) / BEAT
    i = math.floor(k)
    dt = (k - i) * BEAT
    amp = 1.0 if i % 4 == 0 else (0.7 if i % 2 == 0 else 0.5)
    return amp * math.exp(-dt / 0.16)


# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in(x):
    x = clamp(x)
    return x ** 3


def ease_io(x):
    x = clamp(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def ease_back(x, s=1.6):
    x = clamp(x) - 1
    return 1 + x * x * ((s + 1) * x + s)


def window(t, a, b, fi, fo):
    """0..1 envolvente: sube en [a, a+fi], baja en [b-fo, b]."""
    if t <= a or t >= b:
        return 0.0
    v = 1.0
    if fi > 0:
        v = min(v, ease_io((t - a) / fi))
    if fo > 0:
        v = min(v, ease_io((b - t) / fo))
    return v


# ---------------------------------------------------------------- fonts
def font(name, size, wght=None):
    f = ImageFont.truetype(A("fonts", name), int(size))
    if wght is not None:
        try:
            f.set_variation_by_axes([wght])
        except Exception:
            pass
    return f


SCRIPT = "GreatVibes-Regular.ttf"
SERIF_I = "CormorantGaramond-Italic.ttf"
SANS = "Montserrat.ttf"

# ---------------------------------------------------------------- helpers


def comp(dst, src, x, y):
    if src is None:
        return
    x, y = int(round(x)), int(round(y))
    w, h = src.size
    if x >= W or y >= H or x + w <= 0 or y + h <= 0:
        return
    sx0, sy0 = max(0, -x), max(0, -y)
    sx1, sy1 = min(w, W - x), min(h, H - y)
    if (sx0, sy0, sx1, sy1) != (0, 0, w, h):
        src = src.crop((sx0, sy0, sx1, sy1))
    dst.alpha_composite(src, (x + sx0, y + sy0))


def comp_c(dst, src, cx, cy):
    if src is not None:
        comp(dst, src, cx - src.width / 2, cy - src.height / 2)


_LUTS = {}


def fade(img, a):
    if img is None or a <= 0.004:
        return None
    if a >= 0.996:
        return img
    q = int(a * 255)
    lut = _LUTS.get(q)
    if lut is None:
        lut = _LUTS[q] = [v * q // 255 for v in range(256)]
    out = img.copy()
    out.putalpha(img.getchannel("A").point(lut))
    return out


def solid(size, color, alpha_mask):
    im = Image.new("RGBA", size, color + (255,))
    im.putalpha(alpha_mask)
    return im


def glow_text_sprite(text, f, fill=CREAM, glow=None, glow_r=16, glow_a=0.5,
                     shadow_a=0.75, tracking=0, gradient=None, pad=None):
    """Sprite RGBA de texto con sombra suave y halo. Devuelve (sprite, pad, ascent)."""
    asc, desc = f.getmetrics()
    if tracking:
        widths = [f.getlength(c) + tracking for c in text]
        tw = sum(widths) - tracking
    else:
        tw = f.getlength(text)
    pad = pad or int(max(30, glow_r * 2.5))
    w, h = int(tw + 2 * pad), int(asc + desc + 2 * pad)
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    if tracking:
        x = pad
        for c, cw in zip(text, widths):
            d.text((x, pad + asc), c, font=f, fill=255, anchor="ls")
            x += cw
    else:
        d.text((pad, pad + asc), text, font=f, fill=255, anchor="ls")
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if shadow_a > 0:
        sh = mask.filter(ImageFilter.GaussianBlur(max(4, glow_r * 0.5)))
        sh = sh.point(lambda v: int(v * shadow_a))
        out.alpha_composite(solid((w, h), (8, 4, 2), sh), (0, 0))
        out = ImageChops.offset(out, 0, 4)
    if glow:
        g = mask.filter(ImageFilter.GaussianBlur(glow_r)).point(lambda v: min(255, int(v * glow_a * 1.6)))
        out.alpha_composite(solid((w, h), glow, g))
    if gradient:
        c0, c1 = gradient
        gx = np.linspace(0, 1, w)[None, :, None]
        gy = np.linspace(0, 1, h)[:, None, None]
        tt = np.clip(0.75 * gx + 0.25 * gy, 0, 1)
        arr = (np.array(c0)[None, None] * (1 - tt) + np.array(c1)[None, None] * tt).astype(np.uint8)
        arr = np.broadcast_to(arr, (h, w, 3))
        fillim = Image.fromarray(np.ascontiguousarray(arr), "RGB").convert("RGBA")
        fillim.putalpha(mask)
    else:
        fillim = solid((w, h), fill, mask)
    out.alpha_composite(fillim)
    return out, pad, asc


# ---------------------------------------------------------------- hearts
def heart_poly(n=400):
    t = np.linspace(0, 2 * np.pi, n)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    return x, y


def heart_master(kind, S=512):
    x, y = heart_poly()
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    m = 0.06
    pts = list(zip(m * S + x * S * (1 - 2 * m), m * S + y * S * (1 - 2 * m) * 0.95))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    gy = np.linspace(0, 1, S)[:, None, None]
    if kind == "orange":
        c0, c1 = np.array([255, 170, 70]), np.array([235, 85, 0])
    else:
        c0, c1 = np.array([52, 46, 44]), np.array([3, 3, 3])
    arr = (c0 * (1 - gy) + c1 * gy).astype(np.uint8)
    arr = np.ascontiguousarray(np.broadcast_to(arr, (S, S, 3)))
    im = Image.fromarray(arr, "RGB").convert("RGBA")
    im.putalpha(mask)
    # brillo suave arriba a la izquierda
    hl = Image.new("L", (S, S), 0)
    ImageDraw.Draw(hl).ellipse((S * 0.2, S * 0.17, S * 0.42, S * 0.33), fill=150 if kind == "orange" else 70)
    hl = ImageChops.multiply(hl.filter(ImageFilter.GaussianBlur(S * 0.04)), mask)
    im.alpha_composite(solid((S, S), (255, 245, 230), hl))
    return im, mask


HEART_M = {}
_HCACHE = {}


def heart_sprite(kind, size, angle=0, soft=0):
    size = max(6, int(round(size / 2) * 2))
    angle = int(round(angle / 3) * 3)
    key = (kind, size, angle, soft)
    sp = _HCACHE.get(key)
    if sp is not None:
        return sp
    if kind not in HEART_M:
        HEART_M[kind] = heart_master(kind)
    im, mask = HEART_M[kind]
    pad = int(size * 0.45) + 4
    S = size + 2 * pad
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    small = im.resize((size, size), Image.LANCZOS)
    m = small.getchannel("A")
    gm = Image.new("L", (S, S), 0)
    gm.paste(m, (pad, pad))
    if kind == "orange":
        g = gm.filter(ImageFilter.GaussianBlur(size * 0.22)).point(lambda v: int(v * 0.85))
        out.alpha_composite(solid((S, S), (255, 120, 20), g))
    else:
        g = gm.filter(ImageFilter.GaussianBlur(size * 0.2)).point(lambda v: int(v * 0.45))
        out.alpha_composite(solid((S, S), (255, 228, 200), g))
    out.alpha_composite(small, (pad, pad))
    if soft:
        out = out.filter(ImageFilter.GaussianBlur(soft))
    if angle:
        out = out.rotate(angle, Image.BICUBIC, expand=True)
    if len(_HCACHE) > 6000:
        _HCACHE.clear()
    _HCACHE[key] = out
    return out


# ---------------------------------------------------------------- stars (mi estrellita)
_SCACHE = {}


def sparkle(size, color=CREAM):
    size = max(4, int(size))
    key = (size, color)
    if key in _SCACHE:
        return _SCACHE[key]
    S = size * 4
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    c = S / 2
    r, w = size * 1.6, size * 0.16
    d.polygon([(c, c - r), (c + w, c - w), (c + r, c), (c + w, c + w), (c, c + r), (c - w, c + w), (c - r, c), (c - w, c - w)], fill=255)
    core = m.filter(ImageFilter.GaussianBlur(0.8))
    g = m.filter(ImageFilter.GaussianBlur(size * 0.5)).point(lambda v: min(255, v * 2))
    out = solid((S, S), ORANGE_L, g)
    out.alpha_composite(solid((S, S), color, core))
    _SCACHE[key] = out
    return out


# ---------------------------------------------------------------- photos
PHOTOS = {
    # id: (archivo, rotar, blanco&negro, foco_y, caption, sticker)
    1: ("1.jpg", 0, False, 0.45, "mi beso favorito", "black"),
    2: ("2.jpg", 0, True, 0.45, "nosotros", "orange"),
    3: ("3.jpg", 0, False, 0.42, "nuestro lugar", "orange"),
    4: ("4.jpg", 0, False, 0.42, "mi estrellita", "black"),
    5: ("5.jpg", 90, True, 0.5, "siempre tú", "orange"),
    6: ("6.jpg", 0, False, 0.36, "tú y yo", "black"),
    7: ("7.jpg", 0, False, 0.36, "esa mirada", "orange"),
    8: ("8.jpg", 0, False, 0.38, "sushi date", "black"),
    9: ("9.jpg", 0, False, 0.40, "juntitos", "orange"),
}


GRADE_OVERRIDE = {4: dict(wb=0.25, sat=0.9)}


def grade(img, bw, wb=0.55, sat=1.12):
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    lum_w = np.array([0.299, 0.587, 0.114], np.float32)
    if not bw:
        # balance de blancos parcial (gray world)
        mean = a.reshape(-1, 3).mean(0)
        gain = (mean.mean() / np.maximum(mean, 1e-3)) ** wb
        a = a * gain
    lum = a @ lum_w
    lo, hi = np.percentile(lum, [0.8, 99.6])
    a = np.clip((a - lo) / max(hi - lo, 1e-3), 0, 1)
    # curva S suave
    a = a * 0.62 + (a * a * (3 - 2 * a)) * 0.38
    lum = a @ lum_w
    if bw:
        l = lum[..., None]
        dark = np.array([0.07, 0.045, 0.035], np.float32)
        light = np.array([1.0, 0.95, 0.88], np.float32)
        a = dark + (light - dark) * l
    else:
        gray = lum[..., None]
        a = gray + (a - gray) * sat
        sh = (1 - lum[..., None]) ** 2
        hl = lum[..., None] ** 2
        a = a + sh * np.array([0.035, 0.012, -0.02]) + hl * np.array([0.03, 0.012, -0.025])
        a = a * 0.97 + 0.03 * np.array([0.12, 0.06, 0.02])  # negros levantados, calidos
    a = np.clip(a, 0, 1)
    out = Image.fromarray((a * 255).astype(np.uint8), "RGB")
    return out.filter(ImageFilter.UnsharpMask(radius=1.6, percent=60, threshold=2))


def load_photo(pid):
    fn, rot, bw, fy, cap, st = PHOTOS[pid]
    im = ImageOps.exif_transpose(Image.open(A("photos", fn)))
    if rot:
        im = im.rotate(rot, expand=True)
    return grade(im, bw, **GRADE_OVERRIDE.get(pid, {}))


def crop_box(img, aspect, fy, fx=0.5):
    w, h = img.size
    if w / h > aspect:
        cw, ch = h * aspect, h
    else:
        cw, ch = w, w / aspect
    x0 = clamp(fx * w - cw / 2, 0, w - cw)
    y0 = clamp(fy * h - ch / 2, 0, h - ch)
    return (x0, y0, cw, ch)


def rounded_mask(size, r):
    S = 3
    m = Image.new("L", (size[0] * S, size[1] * S), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size[0] * S - 1, size[1] * S - 1), radius=r * S, fill=255)
    return m.resize(size, Image.LANCZOS)


class Card:
    """Polaroid con cinta, sticker de corazon, sombra y foto con movimiento Ken Burns."""

    def __init__(self, pid, inner_w, seed=0, small=False):
        self.pid = pid
        fn, rot, bw, fy, cap, st = PHOTOS[pid]
        self.photo = PHOTO_IMG[pid]
        pw, ph = self.photo.size
        landscape = pw > ph
        aspect = 16 / 10 if landscape else 4 / 5
        iw = int(inner_w * (1.45 if landscape else 1.0))
        ih = int(iw / aspect)
        self.iw, self.ih = iw, ih
        self.box = crop_box(self.photo, aspect, fy)
        sc = iw / 640
        bd = int(26 * sc)
        bot = int(112 * sc)
        cw, ch = iw + 2 * bd, ih + bd + bot
        pad = int(80 * sc) + 10
        self.pad = pad
        Wc, Hc = cw + 2 * pad, ch + 2 * pad
        base = Image.new("RGBA", (Wc, Hc), (0, 0, 0, 0))
        sh = Image.new("L", (Wc, Hc), 0)
        ImageDraw.Draw(sh).rounded_rectangle((pad + 10 * sc, pad + 26 * sc, pad + cw - 10 * sc, pad + ch + 16 * sc), radius=int(14 * sc), fill=170)
        sh = sh.filter(ImageFilter.GaussianBlur(30 * sc))
        base.alpha_composite(solid((Wc, Hc), (0, 0, 0), sh))
        rng = np.random.default_rng(seed + pid)
        paper = np.ones((ch, cw, 3), np.float32) * np.array([250, 244, 234], np.float32)
        paper += rng.normal(0, 3.0, (ch, cw, 1))
        paper += np.linspace(4, -6, ch)[:, None, None]
        paper_im = Image.fromarray(np.clip(paper, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
        paper_im.putalpha(rounded_mask((cw, ch), int(10 * sc)))
        base.alpha_composite(paper_im, (pad, pad))
        # caption
        f = font(SCRIPT, 54 * sc)
        d = ImageDraw.Draw(base)
        d.text((pad + cw / 2, pad + bd + ih + bot * 0.56), cap, font=f, fill=INK + (235,), anchor="mm")
        # marco interno de la foto (sombra interior suave)
        self.inner_xy = (pad + bd, pad + bd)
        self.inner_mask = rounded_mask((iw, ih), int(4 * sc))
        # overlay: cinta + sticker
        ov = Image.new("RGBA", (Wc, Hc), (0, 0, 0, 0))
        tw, th = int(170 * sc), int(46 * sc)
        tape = Image.new("RGBA", (tw, th), (255, 248, 236, 150))
        tnoise = Image.fromarray(np.clip(rng.normal(150, 18, (th, tw)), 0, 255).astype(np.uint8), "L")
        tape.putalpha(tnoise)
        tape = tape.rotate(float(rng.uniform(-8, 8)), Image.BICUBIC, expand=True)
        comp_tmp = Image.new("RGBA", (Wc, Hc), (0, 0, 0, 0))
        comp_tmp.alpha_composite(tape, (int(pad + cw / 2 - tape.width / 2), int(pad - th * 0.45)))
        ov.alpha_composite(comp_tmp)
        hs = heart_sprite(st, 58 * sc, angle=float(rng.uniform(-18, 18)))
        ov.alpha_composite(hs, (int(pad + cw - bd - hs.width * 0.62), int(pad + bd + ih - hs.height * 0.5)))
        self.base, self.overlay = base, ov
        self.static = None
        if small:
            self.static = self.render(0.5, 1.0)

    def render(self, p, zoom_dir=1):
        """p 0..1 progreso de Ken Burns."""
        x0, y0, cw, ch = self.box
        z = 1.0 + 0.09 * (p if zoom_dir > 0 else 1 - p)
        w, h = cw / z, ch / z
        cx = x0 + cw / 2 + (p - 0.5) * (cw - w) * 0.6
        cy = y0 + ch / 2 + (0.5 - p) * (ch - h) * 0.3
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        inner = self.photo.resize((self.iw, self.ih), Image.BICUBIC, box=box)
        c = self.base.copy()
        c.paste(inner, self.inner_xy, self.inner_mask)
        c.alpha_composite(self.overlay)
        return c


def transform(img, angle=0.0, scale=1.0, alpha=1.0):
    if abs(scale - 1) > 0.003:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BICUBIC)
    if abs(angle) > 0.05:
        img = img.rotate(angle, Image.BICUBIC, expand=True)
    return fade(img, alpha)


# ---------------------------------------------------------------- backgrounds
def make_base_bg():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W * 0.5) / W) ** 2 + ((yy - H * 0.62) / H) ** 2 * 1.4)
    t = np.clip(r / 0.85, 0, 1)[..., None]
    c0 = np.array([52, 22, 8], np.float32)
    c1 = np.array([8, 5, 4], np.float32)
    arr = c0 * (1 - t) ** 1.6 + c1 * (1 - (1 - t) ** 1.6)
    return Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")


def make_bokeh(seed, n, rmin, rmax, blur, alpha):
    rng = random.Random(seed)
    BW, BH = W + 500, H + 300
    im = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
    cols = [(255, 140, 40), (255, 170, 90), (255, 110, 20), (255, 210, 160), (220, 90, 10)]
    for _ in range(n):
        r = rng.uniform(rmin, rmax)
        x, y = rng.uniform(0, BW), rng.uniform(0, BH)
        lay = Image.new("L", (BW, BH), 0)
        a = int(alpha * rng.uniform(0.35, 1.0) * 255)
        ImageDraw.Draw(lay).ellipse((x - r, y - r, x + r, y + r), fill=a)
        lay = lay.crop((int(x - r - blur * 3), int(y - r - blur * 3), int(x + r + blur * 3), int(y + r + blur * 3)))
        lay = lay.filter(ImageFilter.GaussianBlur(blur))
        im.alpha_composite(solid(lay.size, rng.choice(cols), lay), (int(x - r - blur * 3), int(y - r - blur * 3)))
    return im


def make_photo_bg(pid):
    img = PHOTO_IMG[pid]
    BW, BH = W + 120, H + 70
    img = ImageOps.fit(img, (BW // 4, BH // 4), Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(9)).resize((BW, BH), Image.BICUBIC)
    a = np.asarray(img).astype(np.float32)
    a = a * 0.42 + np.array([14, 6, 0])
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


def make_vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    a = np.clip((r - 0.62) / 0.75, 0, 1) ** 1.4 * 215
    v = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    v.putalpha(Image.fromarray(a.astype(np.uint8), "L"))
    return v


def make_grain(n=6):
    rng = np.random.default_rng(7)
    out = []
    for _ in range(n):
        g = rng.normal(128, 55, (H // 2, W // 2)).clip(0, 255).astype(np.uint8)
        gi = Image.fromarray(g, "L").resize((W, H), Image.BILINEAR)
        im = Image.merge("RGBA", (gi, gi, gi, Image.new("L", (W, H), 14)))
        out.append(im)
    return out


def make_leak():
    S = 1500
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    r = np.sqrt(((xx - S / 2) / (S / 2)) ** 2 + ((yy - S / 2) / (S / 2)) ** 2)
    a = np.clip(1 - r, 0, 1) ** 2.2 * 200
    im = Image.new("RGBA", (S, S), (255, 135, 35, 0))
    im.putalpha(Image.fromarray(a.astype(np.uint8), "L"))
    return im


# ---------------------------------------------------------------- scenes
SCENES = [
    # (inicio, fin, tipo, foto, lado)  lado: 'L' tarjeta a la izquierda
    (bar_t(10), bar_t(18), "photo", 3, "L"),
    (bar_t(18), bar_t(25), "photo", 6, "R"),
    (bar_t(25), bar_t(33), "photo", 1, "L"),
    (bar_t(33), bar_t(40), "photo", 7, "R"),
    (bar_t(40), bar_t(48), "photo", 8, "L"),
    (bar_t(48), bar_t(57), "photo", 4, "R"),
    (bar_t(57), bar_t(65), "photo", 9, "L"),
    (bar_t(65), bar_t(73), "photo", 5, "R"),
    (bar_t(73), bar_t(81), "collage", None, None),
    (bar_t(81), bar_t(90), "photo", 2, "L"),
    (bar_t(90), bar_t(99), "gallery", None, None),
]
INTRO_END = bar_t(10)
OUTRO_T = bar_t(99)


def text_column(scene):
    """(centro_x, centro_y, ancho) de la columna de letra segun la escena."""
    if scene is None:
        return (W / 2, H / 2, 1300)
    s, e, kind, pid, side = scene
    if kind == "collage":
        return (W / 2, 905, 1500)
    if kind == "gallery":
        return (W / 2, 535, 900)
    card = CARDS[pid]
    landscape = card.iw > card.ih
    if landscape:
        return (1475, 520, 720) if side == "L" else (445, 520, 720)
    return (1335, 500, 840) if side == "L" else (585, 500, 840)


def card_center(scene):
    s, e, kind, pid, side = scene
    card = CARDS[pid]
    landscape = card.iw > card.ih
    if landscape:
        return (575, 515) if side == "L" else (1345, 515)
    return (500, 520) if side == "L" else (1420, 520)


def scene_at(t):
    for sc in SCENES:
        if sc[0] <= t < sc[1]:
            return sc
    return None


# ---------------------------------------------------------------- lyrics layout
LINE_CACHE = {}


def build_line(idx, col):
    key = (idx, col)
    if key in LINE_CACHE:
        return LINE_CACHE[key]
    L = LY["lines"][idx]
    cx, cy, cw = col
    style = L.get("style", "line")
    items = []
    if style == "hook":
        size = 150
        f = font(SCRIPT, size)
        while f.getlength(L["text"]) > cw - 20 and size > 70:
            size -= 6
            f = font(SCRIPT, size)
        sp, pad, asc = glow_text_sprite(L["text"], f, glow=(255, 110, 10), glow_r=26, glow_a=0.55,
                                        gradient=((255, 196, 120), (255, 105, 10)), shadow_a=0.8)
        glow_only = None
        items.append(("hook", sp, cx - sp.width / 2, cy - 30 - sp.height / 2))
        if L.get("sub"):
            fs = font(SANS, 25, 500)
            ss, sp2, _ = glow_text_sprite(L["sub"].upper(), fs, fill=CREAM, tracking=7, shadow_a=0.9, glow_r=10)
            items.append(("sub", ss, cx - ss.width / 2, cy + size * 0.42 - ss.height / 2 + 6))
    else:
        size = 70
        f = font(SERIF_I, size, 560)
        words = L["text"].split()
        space = f.getlength(" ")
        while True:
            lines, cur, curw = [], [], 0
            for wd in words:
                wl = f.getlength(wd)
                if cur and curw + space + wl > cw:
                    lines.append((cur, curw))
                    cur, curw = [], 0
                cur.append((wd, wl))
                curw = curw + (space if len(cur) > 1 else 0) + wl
            if cur:
                lines.append((cur, curw))
            if len(lines) <= 2 or size <= 50:
                break
            size -= 4
            f = font(SERIF_I, size, 560)
            space = f.getlength(" ")
        lh = size * 1.18
        top = cy - lh * len(lines) / 2
        k = 0
        for li, (ws, lw) in enumerate(lines):
            x = cx - lw / 2
            base_y = top + li * lh + size * 0.85
            for wd, wl in ws:
                sp, pad, asc = glow_text_sprite(wd, f, fill=CREAM, glow=(255, 120, 30), glow_r=18, glow_a=0.22, shadow_a=0.85, pad=36)
                items.append(("word", sp, x - pad, base_y - asc - pad, k))
                x += wl + space
                k += 1
    LINE_CACHE[key] = items
    return items


def draw_lyrics(fr, t, col_fn):
    for idx, L in enumerate(LY["lines"]):
        t0 = bar_t(L["bar"]) - 0.05
        t1 = bar_t(L["bar"] + L.get("bars", 2)) - 0.05
        if not (t0 <= t < t1 + 0.1):
            continue
        sc = scene_at(bar_t(L["bar"]) + 0.01)
        col = col_fn(sc)
        items = build_line(idx, col)
        out_p = ease_in((t - (t1 - 0.55)) / 0.55)
        float_y = 4 * math.sin(t * 1.3)
        for it in items:
            kind, sp = it[0], it[1]
            x, y = it[2], it[3]
            if kind == "word":
                k = it[4]
                p = ease_out((t - t0 - k * 0.11) / 0.7)
                a = p * (1 - out_p)
                dy = (1 - p) * 26 - out_p * 22 + float_y
                comp(fr, fade(sp, a), x, y + dy)
            elif kind == "hook":
                p = clamp((t - t0) / 1.1)
                if p <= 0:
                    continue
                img = sp
                if p < 1:
                    w = sp.width
                    F = 140
                    xs = np.arange(w, dtype=np.float32)
                    m = np.clip((ease_out(p) * (w + F) - xs) / F, 0, 1) * 255
                    mask = Image.fromarray(np.broadcast_to(m.astype(np.uint8), (sp.height, w)).copy(), "L")
                    img = sp.copy()
                    img.putalpha(ImageChops.multiply(sp.getchannel("A"), mask))
                pulse = beat_pulse(t)
                sc_ = 1 + 0.025 * pulse
                if sc_ > 1.003:
                    img2 = img.resize((int(img.width * sc_), int(img.height * sc_)), Image.BICUBIC)
                else:
                    img2 = img
                a = (1 - out_p)
                comp(fr, fade(img2, a), x + img.width / 2 - img2.width / 2, y + img.height / 2 - img2.height / 2 - out_p * 20 + float_y)
            elif kind == "sub":
                p = ease_out((t - t0 - 0.6) / 0.8)
                a = p * (1 - out_p)
                comp(fr, fade(sp, a * 0.9), x, y + (1 - p) * 12 - out_p * 16 + float_y)


# ---------------------------------------------------------------- particles
def build_hearts():
    rng = random.Random(42)
    hearts = []
    t = 1.0
    while t < VIDEO_END:
        if t < INTRO_END:
            rate = 0.9
        elif t > OUTRO_T + 3:
            rate = 0.6
        else:
            # mas corazones en los coros
            chorus = any(bar_t(a) <= t < bar_t(b) for a, b in [(25, 33), (48, 81), (90, 99)])
            rate = 2.3 if chorus else 1.4
        t += rng.expovariate(rate)
        z = rng.random() ** 0.8
        speed = 55 + 110 * z
        size = 16 + 46 * z
        life = (H + 200) / speed
        hearts.append(dict(t0=t, life=life, x=rng.uniform(40, W - 40), z=z, speed=speed, size=size,
                           kind="orange" if rng.random() < 0.5 else "black",
                           sw=rng.uniform(15, 45), sf=rng.uniform(0.18, 0.45), ph=rng.uniform(0, 6.28),
                           rot=rng.uniform(8, 22), rf=rng.uniform(0.2, 0.5)))
    return hearts


def draw_hearts(fr, t, layer, outro_fade=1.0):
    pulse = beat_pulse(t)
    for h in HEARTS:
        if h["t0"] > t:
            break
        age = t - h["t0"]
        if age > h["life"]:
            continue
        z = h["z"]
        front = z > 0.86
        if (layer == "front") != front:
            continue
        y = H + 70 - age * h["speed"]
        x = h["x"] + h["sw"] * math.sin(h["sf"] * age * 6.283 + h["ph"])
        ang = h["rot"] * math.sin(h["rf"] * age * 6.283 + h["ph"])
        size = h["size"] * (1 + 0.14 * pulse)
        soft = 2 if z < 0.3 else 0
        a = (0.45 + 0.55 * z) * clamp(age / 1.2) * clamp((h["life"] - age) / 1.5) * outro_fade
        sp = heart_sprite(h["kind"], size, ang, soft)
        comp_c(fr, fade(sp, a), x, y)


# ---------------------------------------------------------------- pieces
def draw_background(fr, t):
    # bokeh en dos capas con parallax
    bx = 250 + 220 * math.sin(t * 0.045)
    by = 150 + 120 * math.sin(t * 0.06 + 1)
    fr.alpha_composite(BOKEH_FAR.crop((int(bx), int(by), int(bx) + W, int(by) + H)))
    bx2 = 250 + 240 * math.sin(t * 0.07 + 2)
    by2 = 150 + 140 * math.cos(t * 0.05)
    fr.alpha_composite(BOKEH_NEAR.crop((int(bx2), int(by2), int(bx2) + W, int(by2) + H)))
    # fondo desenfocado de la foto de la escena
    for sc in SCENES:
        s, e, kind, pid, side = sc
        if pid is None:
            continue
        a = window(t, s - 0.6, e + 0.6, 1.2, 1.2)
        if a <= 0:
            continue
        bg = PHOTO_BG[pid]
        ox = 60 + 55 * math.sin(t * 0.11)
        oy = 35 + 30 * math.cos(t * 0.09)
        crop = bg.crop((int(ox), int(oy), int(ox) + W, int(oy) + H))
        comp(fr, fade(crop, a * 0.85), 0, 0)


def draw_leaks(fr, t):
    for sc in SCENES[1:] + [(OUTRO_T, 0, "", None, None)]:
        s = sc[0]
        d = t - (s - 0.7)
        if 0 <= d <= 1.6:
            p = d / 1.6
            a = math.sin(math.pi * p) * 0.75
            x = -700 + p * (W + 1400)
            comp_c(fr, fade(LEAK, a), x, 300 + 250 * p)


def draw_card_scene(fr, t, sc):
    s, e, kind, pid, side = sc
    if not (s - 0.25 <= t < e + 0.25):
        return
    card = CARDS[pid]
    sign = -1 if side == "L" else 1
    p_in = ease_out((t - (s - 0.25)) / 1.25)
    p_out = ease_in((t - (e - 0.75)) / 1.0)
    prog = clamp((t - s) / (e - s))
    img = card.render(prog, 1 if pid % 2 else -1)
    base_ang = 2.2 * (1 if side == "L" else -1)
    ang = base_ang + (1 - p_in) * 9 * sign + 1.1 * math.sin(t * 0.8 + pid) + p_out * 7 * sign
    scale = (0.9 + 0.1 * p_in) * (1 - 0.06 * p_out)
    alpha = p_in * (1 - p_out)
    cx, cy = card_center(sc)
    cy += (1 - p_in) * 140 + 9 * math.sin(t * 1.05 + pid) - p_out * 40
    cx += p_out * 160 * sign
    comp_c(fr, transform(img, ang, scale, alpha), cx, cy)


def draw_collage(fr, t, sc):
    s, e = sc[0], sc[1]
    a = window(t, s - 0.3, e + 0.2, 1.0, 1.0)
    if a <= 0:
        return
    v = 150
    L_ = sum(c.width + 30 for c in SMALL_CARDS)
    acc = 0
    for i, c in enumerate(SMALL_CARDS):
        x = ((acc - v * (t - s) + W + 200) % L_) - c.width / 2 - 150
        acc += c.width + 30
        y = 420 + 26 * math.sin(t * 0.9 + i * 1.3) + (18 if i % 2 else -18)
        comp(fr, fade(c, a), x, y - c.height / 2)
    # velo inferior para la letra
    comp(fr, fade(SCRIM_BOTTOM, a), 0, H - SCRIM_BOTTOM.height)


GALLERY_POS = [(250, 230), (640, 150), (1280, 150), (1670, 235), (215, 790), (600, 905), (1320, 905), (1700, 800), (960, 905)]


def draw_gallery(fr, t, sc):
    s, e = sc[0], sc[1]
    out = ease_in((t - (OUTRO_T - 0.4)) / 1.8)
    if out >= 1:
        return
    for i, (gx, gy) in enumerate(GALLERY_POS):
        ti = s + i * BEAT * 2 - 0.1
        if t < ti:
            continue
        p = clamp((t - ti) / 0.9)
        sc_ = 0.55 + 0.45 * ease_back(p)
        c = GALLERY_CARDS[i]
        dx, dy = gx - W / 2, gy - H / 2
        x = gx + dx * out * 0.8 + 6 * math.sin(t * 0.8 + i)
        y = gy + dy * out * 0.8 + 8 * math.sin(t * 1.1 + i * 2)
        ang = 2.5 * math.sin(t * 0.6 + i)
        comp_c(fr, transform(c, ang, sc_ * (1 - 0.2 * out), clamp(p * 2) * (1 - out)), x, y)
    a = window(t, s, OUTRO_T + 1, 1.0, 1.2)
    comp_c(fr, fade(SCRIM_CENTER, a), W / 2, H / 2)


def draw_hud(fr, t):
    a = window(t, INTRO_END - 1, bar_t(90) + 0.5, 2.0, 1.5) * 0.9
    if a <= 0:
        return
    comp(fr, fade(HUD_TEXT, a * 0.75), 70 - 30, 990 - 30)
    x0, x1, y = 80, W - 80, 1040
    p = clamp(t / SONG_END)
    bar = Image.new("RGBA", (x1 - x0, 3), CREAM + (40,))
    fill = Image.new("RGBA", (max(1, int((x1 - x0) * p)), 3), ORANGE + (230,))
    bar.alpha_composite(fill)
    comp(fr, fade(bar, a), x0, y)
    hs = heart_sprite("orange", 18 * (1 + 0.2 * beat_pulse(t)))
    comp_c(fr, fade(hs, a), x0 + (x1 - x0) * p, y + 1)


def draw_intro(fr, t):
    if t > INTRO_END + 1:
        return
    # estrellitas
    for i, (sx, sy, ss, sp_) in enumerate(STARS):
        a = window(t, 0.5 + i * 0.05, INTRO_END - 0.5, 1.5, 2.0) * (0.45 + 0.55 * (0.5 + 0.5 * math.sin(t * sp_ + i)))
        comp_c(fr, fade(sparkle(ss), a), sx, sy)
    # titulo
    a = window(t, 0.8, 9.6, 0.1, 1.2)
    if a > 0:
        p = clamp((t - 0.8) / 2.2)
        sp = TITLE
        w = sp.width
        F = 180
        xs = np.arange(w, dtype=np.float32)
        m = np.clip((ease_io(p) * (w + F) - xs) / F, 0, 1) * 255
        mask = Image.fromarray(np.broadcast_to(m.astype(np.uint8), (sp.height, w)).copy(), "L")
        img = sp.copy()
        img.putalpha(ImageChops.multiply(sp.getchannel("A"), mask))
        up = ease_in((t - 8.4) / 1.2) * 30
        comp_c(fr, fade(img, a), W / 2, 470 - up)
        a2 = ease_out((t - 2.6) / 1.2) * a
        comp_c(fr, fade(ARTIST, a2), W / 2, 610 - up + (1 - ease_out((t - 2.6) / 1.2)) * 14)
        # linea con corazones
        a3 = ease_out((t - 3.4) / 1.0) * a
        comp_c(fr, fade(DIVIDER, a3), W / 2, 680 - up)
    # dedicatoria
    a = window(t, 10.2, INTRO_END + 0.3, 0.1, 1.4)
    if a > 0:
        comp_c(fr, fade(PARA, ease_out((t - 10.2) / 1.2) * a), W / 2, 330)
        p = clamp((t - 10.9) / 2.4)
        sp = NAME
        w = sp.width
        F = 200
        xs = np.arange(w, dtype=np.float32)
        m = np.clip((ease_io(p) * (w + F) - xs) / F, 0, 1) * 255
        mask = Image.fromarray(np.broadcast_to(m.astype(np.uint8), (sp.height, w)).copy(), "L")
        img = sp.copy()
        img.putalpha(ImageChops.multiply(sp.getchannel("A"), mask))
        comp_c(fr, fade(img, a), W / 2, 500)
        p2 = ease_out((t - 13.2) / 1.4)
        comp_c(fr, fade(STAR_SUB, p2 * a), W / 2, 680 + (1 - p2) * 16)
        # estrella fugaz
        d = t - 12.0
        if 0 <= d <= 1.6:
            q = ease_io(d / 1.6)
            sx, sy = 300 + q * 1400, 120 + q * 220
            comp_c(fr, fade(SHOOT, math.sin(math.pi * d / 1.6)), sx - 120, sy - 19)
            comp_c(fr, sparkle(16), sx, sy)


def heart_outline_points(n):
    x, y = heart_poly(2000)
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0], np.cumsum(seg)])
    targets = np.linspace(0, s[-1], n, endpoint=False)
    px = np.interp(targets, s, x)
    py = np.interp(targets, s, y)
    return px, py


def draw_outro(fr, t):
    if t < OUTRO_T - 0.5:
        return
    pulse = beat_pulse(t) if t < SONG_END else 0
    p = ease_io((t - OUTRO_T) / 4.2)
    hb = 1 + 0.045 * pulse
    cx, cy, k = W / 2, 430, 22
    for i in range(len(OUT_PX)):
        tx = cx + OUT_PX[i] * k * hb
        ty = cy + OUT_PY[i] * k * hb
        sx, sy = OUT_START[i]
        # trayectoria curva
        q = clamp(p * 1.25 - OUT_DELAY[i])
        q = ease_io(q)
        mx = (sx + tx) / 2 + OUT_CURVE[i][0]
        my = (sy + ty) / 2 + OUT_CURVE[i][1]
        x = (1 - q) ** 2 * sx + 2 * (1 - q) * q * mx + q * q * tx
        y = (1 - q) ** 2 * sy + 2 * (1 - q) * q * my + q * q * ty
        ang = (1 - q) * OUT_ROT[i] + 6 * math.sin(t * 1.5 + i)
        sp = heart_sprite("orange" if i % 2 else "black", 30 * (1 + 0.12 * pulse), ang)
        comp_c(fr, fade(sp, clamp((t - OUTRO_T + 0.5) / 0.8)), x, y)
    # textos
    a = ease_out((t - OUTRO_T - 3.6) / 1.6)
    if a > 0:
        p2 = clamp((t - OUTRO_T - 3.6) / 2.4)
        sp = NAME_SMALL
        w = sp.width
        F = 160
        xs = np.arange(w, dtype=np.float32)
        m = np.clip((ease_io(p2) * (w + F) - xs) / F, 0, 1) * 255
        mask = Image.fromarray(np.broadcast_to(m.astype(np.uint8), (sp.height, w)).copy(), "L")
        img = sp.copy()
        img.putalpha(ImageChops.multiply(sp.getchannel("A"), mask))
        comp_c(fr, img, cx, cy - 40)
        comp_c(fr, fade(STAR_SUB_SMALL, ease_out((t - OUTRO_T - 5.5) / 1.4)), cx, cy + 85)
        a3 = ease_out((t - OUTRO_T - 7.0) / 1.6)
        comp_c(fr, fade(END_LINE, a3), cx, 900 + (1 - a3) * 14)
        for i, (sx, sy, ss, sp_) in enumerate(STARS[:14]):
            aa = a3 * (0.4 + 0.6 * (0.5 + 0.5 * math.sin(t * sp_ + i)))
            comp_c(fr, fade(sparkle(ss), aa), sx, sy * 0.7 + 20)


# ---------------------------------------------------------------- frame
def render_frame(t):
    fr = BASE_BG.copy()
    draw_background(fr, t)
    draw_leaks(fr, t)
    outro_fade = 1 - 0.6 * clamp((t - OUTRO_T) / 3)
    draw_hearts(fr, t, "back", outro_fade)
    draw_intro(fr, t)
    for sc in SCENES:
        if sc[2] == "photo":
            draw_card_scene(fr, t, sc)
        elif sc[2] == "collage" and sc[0] - 1 <= t < sc[1] + 1:
            draw_collage(fr, t, sc)
        elif sc[2] == "gallery" and sc[0] - 1 <= t:
            draw_gallery(fr, t, sc)
    draw_hearts(fr, t, "front", outro_fade)
    draw_outro(fr, t)
    draw_lyrics(fr, t, text_column)
    draw_hud(fr, t)
    fr.alpha_composite(VIGNETTE)
    fr.alpha_composite(GRAIN[int(t * FPS) % len(GRAIN)])
    # fundidos de entrada/salida
    k = clamp(t / 0.9) * clamp((VIDEO_END - t) / 2.2)
    if k < 1:
        black = Image.new("RGBA", (W, H), (0, 0, 0, int((1 - k) * 255)))
        fr.alpha_composite(black)
    return fr.convert("RGB")


# ---------------------------------------------------------------- setup (antes del fork)
def setup():
    global PHOTO_IMG, PHOTO_BG, CARDS, SMALL_CARDS, GALLERY_CARDS, BASE_BG, BOKEH_FAR, BOKEH_NEAR
    global VIGNETTE, GRAIN, LEAK, HEARTS, STARS, TITLE, ARTIST, DIVIDER, PARA, NAME, STAR_SUB
    global SHOOT, NAME_SMALL, STAR_SUB_SMALL, END_LINE, HUD_TEXT, SCRIM_BOTTOM, SCRIM_CENTER
    global OUT_PX, OUT_PY, OUT_START, OUT_DELAY, OUT_CURVE, OUT_ROT
    print("preparando fotos...", flush=True)
    PHOTO_IMG = {pid: load_photo(pid) for pid in PHOTOS}
    PHOTO_BG = {pid: make_photo_bg(pid) for pid in PHOTOS}
    CARDS = {pid: Card(pid, 590) for pid in PHOTOS}
    order = [3, 6, 1, 7, 8, 4, 9, 5, 2]
    SMALL_CARDS = []
    for i, pid in enumerate(order):
        c = Card(pid, 300, seed=10 + i, small=True).static
        SMALL_CARDS.append(c.rotate((-4, 3, -2, 5, -3, 2, -5, 4, -1)[i], Image.BICUBIC, expand=True))
    GALLERY_CARDS = [Card(pid, 215, seed=20 + i, small=True).static for i, pid in enumerate([1, 6, 9, 4, 3, 7, 8, 5, 2])]
    print("preparando fondos...", flush=True)
    BASE_BG = make_base_bg()
    BOKEH_FAR = make_bokeh(1, 26, 30, 90, 22, 0.20)
    BOKEH_NEAR = make_bokeh(2, 12, 70, 170, 40, 0.16)
    VIGNETTE = make_vignette()
    GRAIN = make_grain()
    LEAK = make_leak()
    HEARTS = build_hearts()
    rng = random.Random(5)
    STARS = [(rng.uniform(80, W - 80), rng.uniform(60, H - 120), rng.uniform(5, 13), rng.uniform(1.5, 3.5)) for _ in range(34)]
    STARS = [s for s in STARS if not (450 < s[0] < 1470 and 250 < s[1] < 760)]
    # textos fijos
    TITLE, _, _ = glow_text_sprite("You Rock My World", font(SCRIPT, 190), glow=(255, 110, 10), glow_r=30, glow_a=0.6,
                                   gradient=((255, 200, 125), (255, 100, 5)))
    ARTIST, _, _ = glow_text_sprite("MICHAEL JACKSON", font(SANS, 30, 500), fill=CREAM, tracking=16, glow_r=10)
    DIVIDER = Image.new("RGBA", (520, 60), (0, 0, 0, 0))
    ImageDraw.Draw(DIVIDER).line((20, 30, 225, 30), fill=CREAM + (120,), width=2)
    ImageDraw.Draw(DIVIDER).line((295, 30, 500, 30), fill=CREAM + (120,), width=2)
    hb, ho = heart_sprite("black", 26), heart_sprite("orange", 26)
    DIVIDER.alpha_composite(hb, (int(243 - hb.width / 2), int(30 - hb.height / 2)))
    DIVIDER.alpha_composite(ho, (int(277 - ho.width / 2), int(30 - ho.height / 2)))
    PARA, _, _ = glow_text_sprite("PARA", font(SANS, 30, 500), fill=CREAM, tracking=22, glow_r=10)
    NAME, _, _ = glow_text_sprite("Catalina", font(SCRIPT, 250), fill=CREAM, glow=(255, 120, 20), glow_r=34, glow_a=0.45)
    STAR_SUB = star_line("mi estrellita", 76)
    STAR_SUB_SMALL = star_line("mi estrellita", 58)
    NAME_SMALL, _, _ = glow_text_sprite("Catalina", font(SCRIPT, 165), fill=CREAM, glow=(255, 120, 20), glow_r=26, glow_a=0.5)
    END_LINE, _, _ = glow_text_sprite("YOU ROCK MY WORLD", font(SANS, 28, 500), fill=CREAM, tracking=18, glow=(255, 120, 20), glow_r=12, glow_a=0.4)
    HUD_TEXT, _, _ = glow_text_sprite("MICHAEL JACKSON  ·  YOU ROCK MY WORLD", font(SANS, 17, 500), fill=CREAM, tracking=6, glow_r=6, pad=30)
    # estela de estrella fugaz
    sh = Image.new("RGBA", (260, 40), (0, 0, 0, 0))
    m = Image.new("L", (260, 40), 0)
    md = ImageDraw.Draw(m)
    for i in range(240):
        md.line((i, 20 - 2 * i / 240, i + 1, 20 + 2 * i / 240), fill=int(255 * (i / 240) ** 2), width=3)
    sh.alpha_composite(solid((260, 40), (255, 220, 170), m.filter(ImageFilter.GaussianBlur(1.5))))
    SHOOT = sh.rotate(-9, Image.BICUBIC, expand=True)
    # velos para que la letra se lea
    yy = np.linspace(0, 1, 360)[:, None]
    SCRIM_BOTTOM = Image.new("RGBA", (W, 360), (6, 3, 2, 0))
    SCRIM_BOTTOM.putalpha(Image.fromarray(np.broadcast_to((yy ** 1.3 * 200).astype(np.uint8), (360, W)).copy(), "L"))
    yy, xx = np.mgrid[0:520, 0:1400].astype(np.float32)
    r = np.sqrt(((xx - 700) / 700) ** 2 + ((yy - 260) / 260) ** 2)
    SCRIM_CENTER = Image.new("RGBA", (1400, 520), (6, 3, 2, 0))
    SCRIM_CENTER.putalpha(Image.fromarray((np.clip(1 - r, 0, 1) ** 0.8 * 175).astype(np.uint8), "L"))
    # corazones del final
    OUT_PX, OUT_PY = heart_outline_points(112)
    OUT_PY = OUT_PY - 1.5
    r2 = random.Random(9)
    OUT_START, OUT_DELAY, OUT_CURVE, OUT_ROT = [], [], [], []
    for i in range(len(OUT_PX)):
        side = r2.choice(["l", "r", "t", "b"])
        if side == "l":
            st = (r2.uniform(-150, 100), r2.uniform(0, H))
        elif side == "r":
            st = (r2.uniform(W - 100, W + 150), r2.uniform(0, H))
        elif side == "t":
            st = (r2.uniform(0, W), r2.uniform(-150, 60))
        else:
            st = (r2.uniform(0, W), r2.uniform(H - 60, H + 150))
        OUT_START.append(st)
        OUT_DELAY.append(r2.uniform(0, 0.25))
        OUT_CURVE.append((r2.uniform(-260, 260), r2.uniform(-260, 260)))
        OUT_ROT.append(r2.uniform(-90, 90))


def star_line(text, size):
    t_sp, _, _ = glow_text_sprite(text, font(SERIF_I, size, 520), fill=ORANGE_L, glow=(255, 110, 10), glow_r=16, glow_a=0.45)
    s = sparkle(12)
    out = Image.new("RGBA", (t_sp.width + 2 * s.width + 10, max(t_sp.height, s.height)), (0, 0, 0, 0))
    out.alpha_composite(t_sp, (s.width + 5, (out.height - t_sp.height) // 2))
    out.alpha_composite(s, (s.width // 2 + 30, out.height // 2 - s.height // 2))
    out.alpha_composite(s, (out.width - s.width - s.width // 2 - 30, out.height // 2 - s.height // 2))
    return out


# ---------------------------------------------------------------- output
def init_worker(fps):
    """En Windows/macOS los procesos arrancan desde cero (spawn): hay que preparar todo de nuevo."""
    global FPS
    FPS = fps
    if "BASE_BG" not in globals():
        setup()


def worker(args):
    i, f0, f1, path, fps = args
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        p.stdin.write(render_frame(fi / fps).tobytes())
        if i == 0 and (fi - f0) % 150 == 0:
            print(f"  worker0 {fi - f0}/{f1 - f0}", flush=True)
    p.stdin.close()
    p.wait()
    return path


def main():
    global FPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", default="")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default=os.path.join(OUT, "catalina_you_rock_my_world.mp4"))
    a = ap.parse_args()
    FPS = a.fps
    os.makedirs(OUT, exist_ok=True)
    setup()
    if a.stills:
        os.makedirs(os.path.join(OUT, "stills"), exist_ok=True)
        for s in a.stills.split(","):
            t = float(s)
            render_frame(t).save(os.path.join(OUT, "stills", f"t{t:07.2f}.jpg"), quality=88)
            print("still", t, flush=True)
        return
    start = a.start
    end = VIDEO_END if a.dur is None else min(VIDEO_END, start + a.dur)
    f0, f1 = int(start * FPS), int(end * FPS)
    n = f1 - f0
    jobs = max(1, a.jobs)
    tmp = os.path.join(OUT, "tmp")
    os.makedirs(tmp, exist_ok=True)
    chunks = []
    step = math.ceil(n / jobs)
    for i in range(jobs):
        c0, c1 = f0 + i * step, min(f1, f0 + (i + 1) * step)
        if c0 < c1:
            chunks.append((i, c0, c1, os.path.join(tmp, f"seg{i:02d}.mp4"), FPS))
    print(f"renderizando {n} frames en {len(chunks)} procesos...", flush=True)
    with Pool(len(chunks), initializer=init_worker, initargs=(FPS,)) as pool:
        paths = pool.map(worker, chunks)
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as fh:
        for p in paths:
            fh.write("file '%s'\n" % os.path.abspath(p).replace("\\", "/"))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-ss", str(start), "-t", str(end - start), "-i", AUDIO,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-af", f"afade=t=out:st={max(0, SONG_END - start - 0.3)}:d=0.3,apad",
                    "-shortest", "-movflags", "+faststart", a.out], check=True)
    print("listo:", a.out, flush=True)


if __name__ == "__main__":
    main()
