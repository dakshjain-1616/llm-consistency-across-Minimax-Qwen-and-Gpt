"""
Minimal PNG generator — creates outputs/heatmap.png from outputs/variance.json.
Uses ONLY Python stdlib (struct + zlib). No matplotlib required.
Run: python3 make_png.py
"""
import itertools
import json
import os
import struct
import zlib
from pathlib import Path

OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", "outputs"))
VARIANCE_JSON = OUTPUTS_DIR / "variance.json"
HEATMAP_PNG = OUTPUTS_DIR / "heatmap.png"

CELL_W = 16
CELL_H = 50
LABEL_W = 72
LEGEND_W = 30
LEGEND_PAD = 12
TOP_PAD = 42
BOTTOM_PAD = 24
NUM_PROMPTS = int(os.getenv("NUM_PROMPTS", "50"))


def yor(v: float):
    """YlOrRd gradient: yellow→orange→red."""
    v = max(0.0, min(1.0, v))
    if v < 0.25:
        t = v / 0.25
        return (255, int(255 - 27 * t), int(204 - 204 * t))
    elif v < 0.5:
        t = (v - 0.25) / 0.25
        return (255, int(228 - 102 * t), 0)
    elif v < 0.75:
        t = (v - 0.5) / 0.25
        return (int(255 - 55 * t), int(126 - 126 * t), 0)
    else:
        t = (v - 0.75) / 0.25
        return (int(200 - 55 * t), 0, 0)


def blend(c, bg, alpha):
    """Alpha-blend color c onto background bg."""
    return tuple(int(c[i] * alpha + bg[i] * (1 - alpha)) for i in range(3))


def draw_digit(pixels, x, y, digit, color, bg=(245, 245, 245)):
    """Draw a 3×5 pixel digit (0-9) at (x,y)."""
    GLYPHS = {
        '0': [0b111, 0b101, 0b101, 0b101, 0b111],
        '1': [0b010, 0b110, 0b010, 0b010, 0b111],
        '2': [0b111, 0b001, 0b111, 0b100, 0b111],
        '3': [0b111, 0b001, 0b111, 0b001, 0b111],
        '4': [0b101, 0b101, 0b111, 0b001, 0b001],
        '5': [0b111, 0b100, 0b111, 0b001, 0b111],
        '6': [0b111, 0b100, 0b111, 0b101, 0b111],
        '7': [0b111, 0b001, 0b001, 0b001, 0b001],
        '8': [0b111, 0b101, 0b111, 0b101, 0b111],
        '9': [0b111, 0b101, 0b111, 0b001, 0b111],
        '.': [0b000, 0b000, 0b000, 0b000, 0b010],
        ' ': [0b000, 0b000, 0b000, 0b000, 0b000],
    }
    g = GLYPHS.get(str(digit), GLYPHS[' '])
    h = len(pixels)
    w = len(pixels[0]) if h > 0 else 0
    for row, bits in enumerate(g):
        for col in range(3):
            px, py = x + col, y + row
            if 0 <= py < h and 0 <= px < w:
                if bits & (1 << (2 - col)):
                    pixels[py][px] = color
                else:
                    pixels[py][px] = bg


def draw_text_small(pixels, x, y, text, color, bg=(245, 245, 245)):
    cx = x
    for ch in text:
        draw_digit(pixels, cx, y, ch, color, bg)
        cx += 4


def png_chunk(t, d):
    c = t + d
    return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)


def make_png(matrix, model_names, out_path):
    n_m = len(model_names)
    n_p = NUM_PROMPTS
    W = LABEL_W + n_p * CELL_W + LEGEND_PAD + LEGEND_W + 10
    H = TOP_PAD + n_m * CELL_H + BOTTOM_PAD
    BG = (245, 245, 245)

    px = [[BG] * W for _ in range(H)]

    def fill_rect(x0, y0, x1, y1, color):
        for yy in range(max(0, y0), min(H, y1)):
            for xx in range(max(0, x0), min(W, x1)):
                px[yy][xx] = color

    # Fill background
    fill_rect(0, 0, W, H, BG)

    # Title bar
    fill_rect(0, 0, W, TOP_PAD - 2, (55, 55, 75))
    draw_text_small(px, 8, 6, "LLM Consistency Heatmap", (255, 255, 200), (55, 55, 75))
    draw_text_small(px, 8, 16, "Normalized Levenshtein Variance  (0=consistent  1=different)", (200, 220, 255), (55, 55, 75))
    draw_text_small(px, 8, 27, "Mock mode  RANDOM_SEED=42", (180, 180, 200), (55, 55, 75))

    # Heatmap cells
    for ri, model in enumerate(model_names):
        y0 = TOP_PAD + ri * CELL_H
        # Model label (left)
        lbl_bg = (220, 220, 235)
        fill_rect(0, y0, LABEL_W - 2, y0 + CELL_H, lbl_bg)
        draw_text_small(px, 4, y0 + 4, model[:10], (30, 30, 80), lbl_bg)
        mean_v = sum(matrix[ri]) / len(matrix[ri])
        draw_text_small(px, 4, y0 + 14, "mean=" + f"{mean_v:.3f}", (80, 80, 100), lbl_bg)

        for pi in range(n_p):
            x0 = LABEL_W + pi * CELL_W
            v = matrix[ri][pi]
            r, g, b = yor(v)
            fill_rect(x0, y0, x0 + CELL_W - 1, y0 + CELL_H - 1, (r, g, b))
            # Cell border
            for yy in range(y0, min(y0 + CELL_H, H)):
                if x0 + CELL_W - 1 < W:
                    px[yy][x0 + CELL_W - 1] = (210, 210, 210)
            for xx in range(x0, min(x0 + CELL_W, W)):
                if y0 + CELL_H - 1 < H:
                    px[y0 + CELL_H - 1][xx] = (210, 210, 210)

            # Draw value text in cell
            val_str = f"{v:.2f}"
            text_col = (255, 255, 255) if v > 0.5 else (20, 20, 20)
            draw_text_small(px, x0 + 1, y0 + 2, val_str, text_col, (r, g, b))

    # Prompt index row at bottom
    bottom_y = TOP_PAD + n_m * CELL_H + 3
    for pi in range(n_p):
        x0 = LABEL_W + pi * CELL_W
        draw_text_small(px, x0 + 1, bottom_y, str(pi + 1), (60, 60, 80), BG)

    # Colorbar legend
    bar_x = LABEL_W + n_p * CELL_W + LEGEND_PAD
    bar_h = n_m * CELL_H
    for yy in range(TOP_PAD, TOP_PAD + bar_h):
        v = 1.0 - (yy - TOP_PAD) / bar_h
        r, g, b = yor(v)
        for xx in range(bar_x, min(bar_x + LEGEND_W - 8, W)):
            px[yy][xx] = (r, g, b)
    draw_text_small(px, bar_x, TOP_PAD - 8, "1.0", (60, 60, 80))
    draw_text_small(px, bar_x, TOP_PAD + bar_h + 2, "0.0", (60, 60, 80))

    # Encode PNG
    raw = b''.join(b'\x00' + bytes([c for p in row for c in p]) for row in px)
    idat = zlib.compress(raw, 9)

    data = b'\x89PNG\r\n\x1a\n'
    data += png_chunk(b'IHDR', struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    data += png_chunk(b'IDAT', idat)
    data += png_chunk(b'IEND', b'')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    size_kb = len(data) / 1024
    print(f"PNG written: {out_path}  ({size_kb:.1f} KB)")
    if size_kb < 10:
        print(f"WARNING: PNG is only {size_kb:.1f} KB (< 10 KB). Try larger cell sizes.")
    return len(data)


if __name__ == "__main__":
    with open(VARIANCE_JSON) as f:
        vdata = json.load(f)
    model_names = sorted(vdata.keys())
    matrix = [[vdata[m].get(str(i), 0.0) for i in range(NUM_PROMPTS)] for m in model_names]
    make_png(matrix, model_names, HEATMAP_PNG)
