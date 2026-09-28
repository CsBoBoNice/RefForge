"""网格 / 宫格拼贴（Pillow 实现，默认不叠加文字）。"""

import cv2
import numpy as np
from PIL import Image


def _bgr_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _pil_to_bgr(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def make_grid(images, cols, rows, max_side=4096, background=(0, 0, 0),
              gap=0, margin=0):
    """把 images 排成 rows×cols 网格；支持单元格间距 gap 与画布留白 margin。"""
    images = [im for im in images if im is not None]
    if not images or cols <= 0 or rows <= 0:
        return None
    images = images[:cols * rows]

    cell_w = max(im.shape[1] for im in images)
    cell_h = max(im.shape[0] for im in images)
    total_w = cols * cell_w + (cols - 1) * gap + 2 * margin
    total_h = rows * cell_h + (rows - 1) * gap + 2 * margin
    scale = min(1.0, float(max_side) / float(max(total_w, total_h)))

    cell_w = max(1, int(round(cell_w * scale)))
    cell_h = max(1, int(round(cell_h * scale)))
    gap = max(0, int(round(gap * scale)))
    margin = max(0, int(round(margin * scale)))
    canvas_w = cols * cell_w + (cols - 1) * gap + 2 * margin
    canvas_h = rows * cell_h + (rows - 1) * gap + 2 * margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), background)
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        pil = _bgr_to_pil(image)
        ratio = min(cell_w / float(pil.width), cell_h / float(pil.height))
        size = (max(1, int(pil.width * ratio)), max(1, int(pil.height * ratio)))
        pil = pil.resize(size, Image.LANCZOS)
        x = margin + col * (cell_w + gap) + (cell_w - size[0]) // 2
        y = margin + row * (cell_h + gap) + (cell_h - size[1]) // 2
        canvas.paste(pil, (x, y))
    return _pil_to_bgr(canvas)


def make_triptych(images, horizontal=True, max_side=4096, gap=0, margin=0):
    if horizontal:
        return make_grid(images, 3, 1, max_side, gap=gap, margin=margin)
    return make_grid(images, 1, 3, max_side, gap=gap, margin=margin)


def make_storyboard(images, max_side=4096, gap=0, margin=0):
    return make_grid(images, 3, 3, max_side, gap=gap, margin=margin)
