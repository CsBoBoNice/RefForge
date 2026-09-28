"""生成合成测试视频：C1-C11 镜头素材与 S1-S4 分割验收素材。

所有视频由一张高纹理"世界图"经仿射/单应变换渲染。C1-C11 输出到 input/ 目录，
S1-S4 由 make_segmentation_videos 输出到分割验收的临时输入目录。
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "app"))

from io_utils import package_root, ensure_dir  # noqa: E402


WIDTH, HEIGHT = 960, 540
FPS = 25
WORLD_W, WORLD_H = 2600, 1600


def build_world(seed=7):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (WORLD_H, WORLD_W, 3), dtype=np.uint8)
    world = cv2.GaussianBlur(noise, (0, 0), 1.2)
    for _ in range(260):
        color = tuple(int(v) for v in rng.integers(0, 256, 3))
        p1 = (int(rng.integers(0, WORLD_W)), int(rng.integers(0, WORLD_H)))
        p2 = (int(rng.integers(0, WORLD_W)), int(rng.integers(0, WORLD_H)))
        thickness = int(rng.integers(1, 6))
        if rng.random() < 0.5:
            cv2.line(world, p1, p2, color, thickness)
        else:
            radius = int(rng.integers(10, 70))
            cv2.circle(world, p1, radius, color, -1 if rng.random() < 0.5 else 2)
    for _ in range(120):
        p = (int(rng.integers(0, WORLD_W)), int(rng.integers(0, WORLD_H)))
        size = int(rng.integers(20, 90))
        color = tuple(int(v) for v in rng.integers(0, 256, 3))
        cv2.rectangle(world, p, (p[0] + size, p[1] + size), color, -1)
    world = cv2.GaussianBlur(world, (0, 0), 0.6)
    return world


def affine_view(center_x, center_y, zoom=1.0, roll_deg=0.0):
    angle = np.deg2rad(roll_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    matrix = np.array([
        [zoom * cos_a, -zoom * sin_a, 0.0],
        [zoom * sin_a, zoom * cos_a, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    matrix[0, 2] = WIDTH / 2.0 - (matrix[0, 0] * center_x + matrix[0, 1] * center_y)
    matrix[1, 2] = HEIGHT / 2.0 - (matrix[1, 0] * center_x + matrix[1, 1] * center_y)
    return matrix


def yaw_homography(angle_deg):
    focal = float(WIDTH)
    k = np.array([[focal, 0.0, WIDTH / 2.0],
                  [0.0, focal, HEIGHT / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    angle = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(angle), 0.0, np.sin(angle)],
                         [0.0, 1.0, 0.0],
                         [-np.sin(angle), 0.0, np.cos(angle)]], dtype=np.float64)
    return k @ rotation @ np.linalg.inv(k)


def roll_homography(angle_deg):
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    to_center = np.array([[1.0, 0.0, WIDTH / 2.0],
                          [0.0, 1.0, HEIGHT / 2.0],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
    rotate = np.array([[cos_a, -sin_a, 0.0],
                       [sin_a, cos_a, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
    from_center = np.array([[1.0, 0.0, -WIDTH / 2.0],
                            [0.0, 1.0, -HEIGHT / 2.0],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
    return to_center @ rotate @ from_center


def render(world, matrix, blur=0):
    frame = cv2.warpPerspective(world, matrix, (WIDTH, HEIGHT),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
    if blur > 0:
        size = int(blur) * 2 + 1
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, :] = 1.0 / size
        frame = cv2.filter2D(frame, -1, kernel)
    return frame


def _scale_brightness(frame, factor):
    if factor == 1.0:
        return frame
    return np.clip(frame.astype(np.float32) * float(factor), 0, 255).astype(np.uint8)


def _tint(frame, factors):
    if factors is None or tuple(factors) == (1.0, 1.0, 1.0):
        return frame
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    out = np.stack([
        np.clip(b.astype(np.float32) * factors[0], 0, 255),
        np.clip(g.astype(np.float32) * factors[1], 0, 255),
        np.clip(r.astype(np.float32) * factors[2], 0, 255),
    ], axis=2)
    return out.astype(np.uint8)


def write_video(path, frames):
    ensure_dir(os.path.dirname(path))
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter open failed for %s" % path)
    for frame in frames:
        writer.write(frame)
    writer.release()


def make_fixed(world):
    frames = []
    rng = np.random.default_rng(1)
    for _ in range(int(2.0 * FPS)):
        jitter_x = 1.5 * rng.standard_normal()
        jitter_y = 1.5 * rng.standard_normal()
        matrix = affine_view(WORLD_W / 2 + jitter_x, WORLD_H / 2 + jitter_y)
        frames.append(render(world, matrix))
    return frames


def make_zoom(world, zoom_from, zoom_to):
    frames = []
    count = int(2.0 * FPS)
    for i in range(count):
        t = i / float(count - 1)
        zoom = zoom_from + (zoom_to - zoom_from) * t
        matrix = affine_view(WORLD_W / 2, WORLD_H / 2, zoom=zoom)
        frames.append(render(world, matrix))
    return frames


def make_translate(world, dx_total, dy_total, duration=2.0):
    frames = []
    count = int(duration * FPS)
    for i in range(count):
        t = i / float(count - 1)
        matrix = affine_view(WORLD_W / 2 + dx_total * t,
                             WORLD_H / 2 + dy_total * t)
        frames.append(render(world, matrix))
    return frames


def make_pan(world, angle_total, duration=2.0):
    frames = []
    count = int(duration * FPS)
    for i in range(count):
        t = i / float(count - 1)
        angle = -angle_total / 2.0 + angle_total * t
        roll = -4.0 + 8.0 * t
        matrix = roll_homography(roll) @ yaw_homography(angle)
        frames.append(render(world, matrix))
    return frames


def make_orbit(world, radius, duration=3.0):
    frames = []
    count = int(duration * FPS)
    for i in range(count):
        t = i / float(count - 1)
        angle = np.pi * t
        cx = WORLD_W / 2 + radius * np.sin(angle)
        cy = WORLD_H / 2 + radius * (1.0 - np.cos(angle)) * 0.5
        matrix = affine_view(cx, cy, roll_deg=12.0 * t)
        frames.append(render(world, matrix))
    return frames


def make_whip(world, duration=0.4):
    frames = []
    count = max(4, int(duration * FPS))
    for i in range(count):
        t = i / float(count - 1)
        matrix = affine_view(WORLD_W / 2 + 10000 * t, WORLD_H / 2)
        frames.append(render(world, matrix, blur=12))
    return frames


def make_follow(world, subject, duration=2.5):
    frames = []
    count = int(duration * FPS)
    sub_h, sub_w = int(HEIGHT * 0.72), int(WIDTH * 0.68)
    subject_patch = cv2.resize(subject, (sub_w, sub_h))
    for i in range(count):
        t = i / float(count - 1)
        background = render(world, affine_view(
            WORLD_W / 2 + 420 * t, WORLD_H / 2))
        x = (WIDTH - sub_w) // 2
        y = (HEIGHT - sub_h) // 2
        background[y:y + sub_h, x:x + sub_w] = subject_patch
        frames.append(background)
    return frames


def make_black(duration=1.0):
    count = int(duration * FPS)
    return [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(count)]


def make_trim(world):
    black_head = [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                  for _ in range(int(0.4 * FPS))]
    black_tail = [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                  for _ in range(int(0.4 * FPS))]
    return black_head + make_translate(world, 700, 0, 1.6) + black_tail


def make_combined_c1_c11(world, subject):
    """把 C1-C11 场景按顺序拼成一条完整视频（硬切衔接）。

    每个镜头 >=2s，以便通过镜头分割的时长约束；C7 甩镜头取 2s。
    C10 全黑与 C11 的黑场头相邻，会合并为一段 invalid；
    C11 的移动部分（含黑场尾）成为一段 move。
    """
    def fixed(seconds, seed=101):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(int(seconds * FPS)):
            out.append(render(world, affine_view(
                WORLD_W / 2 + 1.5 * rng.standard_normal(),
                WORLD_H / 2 + 1.5 * rng.standard_normal())))
        return out

    def zoom(z0, z1, seconds):
        out = []
        count = int(seconds * FPS)
        for i in range(count):
            t = i / float(count - 1)
            out.append(render(world, affine_view(
                WORLD_W / 2, WORLD_H / 2, zoom=z0 + (z1 - z0) * t)))
        return out

    def translate(dx, dy, seconds):
        out = []
        count = int(seconds * FPS)
        for i in range(count):
            t = i / float(count - 1)
            out.append(render(world, affine_view(
                WORLD_W / 2 + dx * t, WORLD_H / 2 + dy * t)))
        return out

    def pan(angle, seconds):
        out = []
        count = int(seconds * FPS)
        for i in range(count):
            t = i / float(count - 1)
            matrix = roll_homography(-4.0 + 8.0 * t) @ yaw_homography(
                -angle / 2.0 + angle * t)
            out.append(render(world, matrix))
        return out

    def orbit(radius, seconds):
        out = []
        count = int(seconds * FPS)
        for i in range(count):
            t = i / float(count - 1)
            angle = np.pi * t
            out.append(render(world, affine_view(
                WORLD_W / 2 + radius * np.sin(angle),
                WORLD_H / 2 + radius * (1.0 - np.cos(angle)) * 0.5,
                roll_deg=12.0 * t)))
        return out

    def whip(seconds):
        out = []
        count = max(4, int(seconds * FPS))
        for i in range(count):
            t = i / float(count - 1)
            out.append(render(world, affine_view(
                WORLD_W / 2 + 20000 * t, WORLD_H / 2), blur=12))
        return out

    def follow(seconds):
        out = []
        count = int(seconds * FPS)
        sub_h, sub_w = int(HEIGHT * 0.72), int(WIDTH * 0.68)
        patch = cv2.resize(subject, (sub_w, sub_h))
        for i in range(count):
            t = i / float(count - 1)
            background = render(world, affine_view(
                WORLD_W / 2 + 420 * t, WORLD_H / 2))
            x = (WIDTH - sub_w) // 2
            y = (HEIGHT - sub_h) // 2
            background[y:y + sub_h, x:x + sub_w] = patch
            out.append(background)
        return out

    def black(seconds):
        return [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                for _ in range(int(seconds * FPS))]

    shots = [
        fixed(3.0),                              # C1 fixed
        zoom(1.0, 1.35, 3.0),                    # C2 push
        zoom(1.35, 1.0, 3.0),                    # C3 pull
        pan(24.0, 3.0),                          # C4 pan
        translate(700, 0, 3.0),                  # C5 move
        translate(0, 500, 3.0),                  # C6 crane
        whip(2.0),                               # C7 whip
        follow(3.0),                             # C8 follow
        orbit(520, 3.0),                         # C9 orbit
        black(2.5),                              # C10 invalid
        black(0.4) + translate(700, 0, 2.0) + black(0.4),  # C11 trim -> move
    ]
    # 每个镜头给不同色调，使硬切在 PySceneDetect 下可检出（真实视频无需此步）。
    tints = [
        (1.0, 1.0, 1.0), (0.4, 0.6, 1.6), (1.6, 0.6, 0.4), (0.4, 1.6, 0.6),
        (1.5, 1.5, 0.4), (0.4, 1.2, 1.6), (1.6, 0.4, 1.6), (0.6, 0.6, 0.6),
        (1.6, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0),
    ]
    frames = []
    for shot, tint in zip(shots, tints):
        frames.extend(_tint(frame, tint) for frame in shot)
    return frames


def main(input_dir=None):
    world = build_world(7)
    subject = build_world(99)
    if input_dir is None:
        input_dir = os.path.join(package_root(), "input")
    ensure_dir(input_dir)

    videos = {
        "C1_fixed.mp4": make_fixed(world),
        "C2_push.mp4": make_zoom(world, 1.0, 1.35),
        "C3_pull.mp4": make_zoom(world, 1.35, 1.0),
        "C4_pan.mp4": make_pan(world, 24.0),
        "C5_move.mp4": make_translate(world, 700, 0),
        "C6_crane.mp4": make_translate(world, 0, 500),
        "C7_whip.mp4": make_whip(world),
        "C8_follow.mp4": make_follow(world, subject),
        "C9_orbit.mp4": make_orbit(world, 520),
        "C10_black.mp4": make_black(),
        "C11_trim.mp4": make_trim(world),
    }

    for name, frames in videos.items():
        path = os.path.join(input_dir, name)
        write_video(path, frames)
        print("wrote %s (%d frames)" % (path, len(frames)))


def make_segmentation_videos(input_dir):
    """生成用于镜头分割验收的多镜头 / 长视频。

    S1: 3 个镜头（3s 固定 + 4s 平移 + 3s 推） -> 期望 3 段
    S2: 单个长镜头 38s -> 期望被切成 ceil(38/15)=3 段，每段 <=15s
    S3: 4 个 1s 短镜头 -> 期望合并为 >=2s 的片段
    S4: 单个 1.5s 视频 -> 低于最短时长，below_min
    """
    world = build_world(7)
    ensure_dir(input_dir)

    def fixed_shot(cx, cy, seconds, seed, brightness=1.0, tint=None):
        rng = np.random.default_rng(seed)
        frames = []
        for _ in range(int(seconds * FPS)):
            jitter_x = 1.0 * rng.standard_normal()
            jitter_y = 1.0 * rng.standard_normal()
            frame = render(world, affine_view(cx + jitter_x, cy + jitter_y))
            frames.append(_tint(_scale_brightness(frame, brightness), tint))
        return frames

    def translate_shot(cx0, cy0, cx1, cy1, seconds, brightness=1.0, tint=None):
        frames = []
        count = int(seconds * FPS)
        for i in range(count):
            t = i / float(count - 1)
            frame = render(world, affine_view(
                cx0 + (cx1 - cx0) * t, cy0 + (cy1 - cy0) * t))
            frames.append(_tint(_scale_brightness(frame, brightness), tint))
        return frames

    def zoom_shot(cx, cy, z0, z1, seconds, brightness=1.0, tint=None):
        frames = []
        count = int(seconds * FPS)
        for i in range(count):
            t = i / float(count - 1)
            frame = render(world, affine_view(
                cx, cy, zoom=z0 + (z1 - z0) * t))
            frames.append(_tint(_scale_brightness(frame, brightness), tint))
        return frames

    s1 = (fixed_shot(500, 500, 3.0, 11, tint=(1.0, 1.0, 1.0))
          + translate_shot(700, 800, 1900, 800, 4.0, tint=(0.3, 0.5, 1.6))
          + zoom_shot(1500, 1000, 1.0, 1.35, 3.0, tint=(1.7, 0.5, 0.3)))
    s2 = translate_shot(500, 800, 1900, 800, 38.0)
    s3 = (fixed_shot(400, 400, 1.0, 21, tint=(1.0, 1.0, 1.0))
          + fixed_shot(1500, 400, 1.0, 22, tint=(1.7, 0.4, 0.4))
          + fixed_shot(2200, 500, 1.0, 23, tint=(0.3, 0.4, 1.7))
          + fixed_shot(500, 1200, 1.0, 24, tint=(0.3, 1.7, 0.4)))
    s4 = fixed_shot(1300, 800, 1.5, 31)

    videos = {
        "S1_multi_shot.mp4": s1,
        "S2_long_take.mp4": s2,
        "S3_short_shots.mp4": s3,
        "S4_tiny.mp4": s4,
    }
    for name, frames in videos.items():
        path = os.path.join(input_dir, name)
        write_video(path, frames)
        print("wrote %s (%d frames)" % (path, len(frames)))


if __name__ == "__main__":
    main()
