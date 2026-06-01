"""
PRISM - Visual processing utilities.
Handles CLIP-based visual embedding extraction from video frames.
"""
import numpy as np
import torch
import cv2
from PIL import Image

from config import NUM_SAMPLE_FRAMES, EMBED_DIM


def sample_frames(video_path, num_frames: int = NUM_SAMPLE_FRAMES) -> list:
    """
    Uniformly sample frames from a video file.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to sample

    Returns:
        List of PIL Image objects
    """
    cap = cv2.VideoCapture(str(video_path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if length == 0:
        cap.release()
        return []

    idxs = set(np.linspace(0, length - 1, num_frames).astype(int))
    frames = []

    for i in range(length):
        ret, frame = cap.read()
        if not ret:
            break
        if i in idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()
    return frames


def extract_visual_embedding(
    video_path,
    clip_model,
    clip_preprocess,
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract a CLIP visual embedding from a video by averaging frame embeddings.

    Args:
        video_path: Path to the video file
        clip_model: Loaded CLIP model
        clip_preprocess: CLIP preprocessing transform
        device: Device string

    Returns:
        Normalized embedding array of shape (512,)
    """
    frames = sample_frames(video_path)

    if not frames:
        return np.zeros(EMBED_DIM)

    with torch.no_grad():
        imgs = torch.stack([clip_preprocess(f) for f in frames]).to(device)
        emb = clip_model.encode_image(imgs)
        emb = emb.mean(dim=0)
        emb = emb / emb.norm()

    return emb.cpu().numpy()
