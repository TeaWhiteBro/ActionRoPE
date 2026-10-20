"""Image / video input and output."""
from __future__ import annotations

import numpy as np

from .geometry import FPS, HEIGHT, WIDTH


def read_frames(path: str, n: int | None = None) -> np.ndarray:
    """Decode a video into uint8 RGB frames [F, H, W, 3] (PyAV)."""
    import av
    out = []
    with av.open(path) as container:
        for frame in container.decode(video=0):
            out.append(frame.to_ndarray(format="rgb24"))
            if n is not None and len(out) >= n:
                break
    return np.stack(out, 0)


def load_first_frame(path: str) -> np.ndarray:
    """Image (or the first frame of a video) -> uint8 [480, 832, 3].

    Inputs of another size are centre-cropped to the 26:15 aspect ratio and resized (bicubic), so the pixel scale of
    the scene - and therefore the walking speed in pixels - is not distorted."""
    from PIL import Image
    if path.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".avi")):
        img = read_frames(path, 1)[0]
    else:
        img = np.array(Image.open(path).convert("RGB"))
    if img.shape[:2] != (HEIGHT, WIDTH):
        h, w = img.shape[:2]
        tw, th = (w, round(w * HEIGHT / WIDTH)) if w * HEIGHT <= h * WIDTH else (round(h * WIDTH / HEIGHT), h)
        x0, y0 = (w - tw) // 2, (h - th) // 2
        img = np.array(Image.fromarray(img[y0:y0 + th, x0:x0 + tw]).resize((WIDTH, HEIGHT), Image.BICUBIC))
    return img


def write_video(path: str, frames, fps: float = FPS, crf: int | None = 16, quality: float | None = None) -> None:
    """Write uint8 RGB frames with libx264. ``quality`` selects imageio's quality scale instead of a CRF (the Wan2.2
    evaluation videos of the paper were written with quality=9, the other backbones with crf=16)."""
    import imageio
    frames = [np.asarray(f) for f in frames]
    if quality is not None:
        writer = imageio.get_writer(path, fps=fps, quality=quality)
        for f in frames:
            writer.append_data(f)
        writer.close()
    else:
        import imageio.v3 as iio
        iio.imwrite(path, np.stack(frames, 0), fps=fps, codec="libx264", output_params=["-crf", str(int(crf))])
