"""Workload profiling utilities with multi-modal support (text, image, video)."""
import os
import csv
import gc
import json
import time
import torch
import tempfile
import subprocess
import traceback
import re
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoConfig
from PIL import Image
from .model import Model, EngineHangError, DEFAULT_MAX_NUM_SEQS
from .gpu_guard import assert_gpus_exclusive, GPUContentionError
from .utils_encoding import NumpyEncoder


# Configuration
DEBUG = True
# Persistent (not node-local /tmp): holds the Video-MME clips, <videoID>.mp4
IMAGE_CACHE_DIR = os.environ.get("IMAGE_CACHE_DIR", "models/image_cache")
VIDEO_CACHE_DIR = os.environ.get("VIDEO_CACHE_DIR", "models/video_cache")

def debug_log(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")


class Modality(Enum):
    TEXT_ONLY = 1
    TEXT_IMAGE = 2
    VIDEO_ONLY = 3
    TEXT_VIDEO = 4


def detect_modality(item: Dict[str, Any]) -> Modality:
    """Auto-detect modality from dataset item fields."""
    has_video = "video" in item and item["video"] is not None
    has_image = any(k in item for k in ["image", "image_url", "coco_url", "images"])
    has_text = any(k in item for k in [
        "text", "sentences", "captions", "caption", "question", 
        "prompt", "instruction", "query", "scene", "script", "labels"
    ])
    
    if has_video:
        return Modality.TEXT_VIDEO if has_text else Modality.VIDEO_ONLY
    elif has_image:
        return Modality.TEXT_IMAGE
    else:
        return Modality.TEXT_ONLY


# ════════════════════════════════════════════════════════════════════════════
#  VIDEO EXTRACTION (for HuggingFace datasets with embedded bytes)
# ════════════════════════════════════════════════════════════════════════════

def _extract_youtube_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    if not url:
        return None
    
    # Standard youtube.com/watch?v=ID format
    match = re.search(r'youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)
    
    # Short youtu.be/ID format
    match = re.search(r'youtu\.be/([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)
    
    # Embedded youtube.com/embed/ID format
    match = re.search(r'youtube\.com/embed/([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)
    
    return None


def _download_youtube_video(
    youtube_id: str,
    cache_dir: str = VIDEO_CACHE_DIR,
    max_height: int = 720,
) -> Optional[str]:
    """
    Download YouTube video using yt-dlp.
    Returns path to downloaded video file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    output_path = os.path.join(cache_dir, f"{youtube_id}.mp4")
    
    # Check cache first
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        debug_log(f"Using cached video: {output_path}")
        return output_path
    
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    
    cmd = [
        "yt-dlp",
        "-f", f"best[height<={max_height}]/best",
        "--no-playlist",
        "-o", output_path,
        url,
    ]
    
    debug_log(f"Downloading YouTube video: {youtube_id}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        if result.returncode == 0 and os.path.exists(output_path):
            debug_log(f"Download successful: {output_path}")
            return output_path
        else:
            debug_log(f"yt-dlp failed: {result.stderr[:200]}")
            return None
            
    except subprocess.TimeoutExpired:
        debug_log("Download timed out")
        return None
    except FileNotFoundError:
        debug_log("yt-dlp not found. Install: pip install yt-dlp")
        return None
    except Exception as e:
        debug_log(f"Download error: {e}")
        return None


def _get_video_bytes(item: Dict[str, Any]) -> Optional[bytes]:
    """
    Extract video bytes from HuggingFace dataset item.
    
    Handles multiple formats:
    - Embedded bytes: video['bytes']
    - File path: video['path']  
    - HuggingFace Video feature: video (torchcodec decoder or dict)
    - YouTube URL: url field + videoID field (Video-MME)
    """
    # ─── Try 'video' field first ───
    if "video" in item and item["video"] is not None:
        video = item["video"]
        
        # Direct bytes
        if isinstance(video, bytes):
            return video
        
        # Dict with bytes or path
        if isinstance(video, dict):
            if "bytes" in video and video["bytes"]:
                return video["bytes"]
            if "path" in video and video["path"]:
                path = video["path"]
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        return f.read()
        
        # HuggingFace Video feature (torchcodec VideoDecoder or similar)
        # Check if it has a path attribute
        if hasattr(video, 'path') and video.path:
            if os.path.exists(video.path):
                with open(video.path, "rb") as f:
                    return f.read()
        
        # Try to get path from string representation
        if isinstance(video, str) and os.path.exists(video):
            with open(video, "rb") as f:
                return f.read()
    
    # ─── Try YouTube download (Video-MME format) ───
    youtube_id = None
    
    # Check videoID field (Video-MME)
    if "videoID" in item and item["videoID"]:
        youtube_id = item["videoID"]
    
    # Check url field
    if not youtube_id and "url" in item and item["url"]:
        youtube_id = _extract_youtube_id(item["url"])
    
    if youtube_id:
        video_path = _download_youtube_video(youtube_id)
        if video_path and os.path.exists(video_path):
            with open(video_path, "rb") as f:
                return f.read()
    
    return None


def _get_video_path_from_item(item: Dict[str, Any]) -> Optional[str]:
    """
    Get video file path from dataset item.
    For datasets where video is stored as a path reference.
    """
    if "video" in item and item["video"] is not None:
        video = item["video"]
        
        # Dict with path
        if isinstance(video, dict) and "path" in video:
            return video["path"]
        
        # Direct path string
        if isinstance(video, str):
            return video
        
        # HuggingFace Video feature with path attribute
        if hasattr(video, 'path'):
            return video.path
    
    return None


def _get_video_frames_from_bytes(
    video_bytes: bytes,
    num_frames: int = 16,
    adaptive: bool = True,
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """
    Extract frames from video bytes using decord.
    
    Args:
        video_bytes: Raw video bytes
        num_frames: Target number of frames to extract
        adaptive: If True, adjust frame count based on video duration
    
    Returns:
        Tuple of (list of PIL Images, metadata dict)
    """
    try:
        import decord
        from decord import VideoReader, cpu
        decord.bridge.set_bridge("native")
    except ImportError:
        raise ImportError("decord required. Install: pip install decord")
    
    metadata = {}
    
    # Write bytes to temp file (decord needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        temp_path = tmp.name
    
    try:
        vr = VideoReader(temp_path, ctx=cpu(0))
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
        duration_sec = total_frames / video_fps if video_fps > 0 else 0
        
        debug_log(f"Video: {total_frames} frames @ {video_fps:.2f} fps = {duration_sec:.1f}s")
        
        metadata["total_frames"] = total_frames
        metadata["video_fps"] = video_fps
        metadata["duration_sec"] = duration_sec
        
        # Adaptive frame count based on duration (creates variation!)
        if adaptive:
            if duration_sec < 5:
                actual_frames = min(num_frames, 8)
            elif duration_sec < 10:
                actual_frames = min(num_frames, 12)
            elif duration_sec < 20:
                actual_frames = num_frames
            else:
                actual_frames = min(num_frames + 8, 32)
            debug_log(f"Adaptive frames: {actual_frames} for {duration_sec:.1f}s video")
        else:
            actual_frames = num_frames
        
        # Uniform sampling
        if total_frames >= actual_frames:
            indices = [int(i * total_frames / actual_frames) for i in range(actual_frames)]
        else:
            indices = list(range(total_frames))
            actual_frames = total_frames
        
        indices = [min(max(i, 0), total_frames - 1) for i in indices]
        metadata["num_frames"] = len(indices)
        
        # Extract frames
        frames = vr.get_batch(indices).asnumpy()
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        if pil_frames:
            metadata["frame_width"] = pil_frames[0].width
            metadata["frame_height"] = pil_frames[0].height
            debug_log(f"Frames: {len(pil_frames)} @ {metadata['frame_width']}x{metadata['frame_height']}")
        
        return pil_frames, metadata
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _get_video_frames_from_path(
    video_path: str,
    num_frames: int = 16,
    adaptive: bool = True,
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """
    Extract frames from video file path using decord.
    """
    try:
        import decord
        from decord import VideoReader, cpu
        decord.bridge.set_bridge("native")
    except ImportError:
        raise ImportError("decord required. Install: pip install decord")
    
    metadata = {}
    
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    duration_sec = total_frames / video_fps if video_fps > 0 else 0
    
    debug_log(f"Video: {total_frames} frames @ {video_fps:.2f} fps = {duration_sec:.1f}s")
    
    metadata["total_frames"] = total_frames
    metadata["video_fps"] = video_fps
    metadata["duration_sec"] = duration_sec
    
    # Adaptive frame count
    if adaptive:
        if duration_sec < 5:
            actual_frames = min(num_frames, 8)
        elif duration_sec < 10:
            actual_frames = min(num_frames, 12)
        elif duration_sec < 20:
            actual_frames = num_frames
        else:
            actual_frames = min(num_frames + 8, 32)
    else:
        actual_frames = num_frames
    
    # Uniform sampling
    if total_frames >= actual_frames:
        indices = [int(i * total_frames / actual_frames) for i in range(actual_frames)]
    else:
        indices = list(range(total_frames))
    
    indices = [min(max(i, 0), total_frames - 1) for i in indices]
    metadata["num_frames"] = len(indices)
    
    frames = vr.get_batch(indices).asnumpy()
    pil_frames = [Image.fromarray(frame) for frame in frames]
    
    if pil_frames:
        metadata["frame_width"] = pil_frames[0].width
        metadata["frame_height"] = pil_frames[0].height
    
    return pil_frames, metadata


def _get_video_frames_from_hf_video(
    video,
    num_frames: int = 16,
    adaptive: bool = True,
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """
    Extract frames from HuggingFace Video feature (torchcodec VideoDecoder).
    """
    metadata = {}
    
    try:
        # Get video metadata
        # torchcodec VideoDecoder has methods like get_frame_at, get_frames_in_range
        
        # Try to get total frames and fps
        if hasattr(video, 'metadata'):
            total_frames = video.metadata.get('num_frames', 100)
            video_fps = video.metadata.get('fps', 30)
        else:
            # Estimate
            total_frames = 100
            video_fps = 30
        
        duration_sec = total_frames / video_fps if video_fps > 0 else 0
        
        metadata["total_frames"] = total_frames
        metadata["video_fps"] = video_fps
        metadata["duration_sec"] = duration_sec
        
        # Adaptive frame count
        if adaptive:
            if duration_sec < 5:
                actual_frames = min(num_frames, 8)
            elif duration_sec < 10:
                actual_frames = min(num_frames, 12)
            elif duration_sec < 20:
                actual_frames = num_frames
            else:
                actual_frames = min(num_frames + 8, 32)
        else:
            actual_frames = num_frames
        
        # Get frames using torchcodec API
        pil_frames = []
        
        if hasattr(video, 'get_frames_in_range'):
            # Calculate step for uniform sampling
            step = max(1, total_frames // actual_frames)
            frames = video.get_frames_in_range(0, min(actual_frames * step, total_frames), step)
            
            # Convert to PIL
            for i in range(min(actual_frames, frames.data.shape[0])):
                frame_data = frames.data[i].permute(1, 2, 0).numpy()
                pil_frames.append(Image.fromarray(frame_data.astype('uint8')))
        
        elif hasattr(video, 'get_frame_at'):
            # Sample uniformly
            indices = [int(i * total_frames / actual_frames) for i in range(actual_frames)]
            for idx in indices:
                frame = video.get_frame_at(min(idx, total_frames - 1))
                frame_data = frame.data.permute(1, 2, 0).numpy()
                pil_frames.append(Image.fromarray(frame_data.astype('uint8')))
        
        metadata["num_frames"] = len(pil_frames)
        
        if pil_frames:
            metadata["frame_width"] = pil_frames[0].width
            metadata["frame_height"] = pil_frames[0].height
        
        return pil_frames, metadata
        
    except Exception as e:
        debug_log(f"Error extracting frames from HF Video: {e}")
        # Fallback: try to get path and use decord
        if hasattr(video, 'path') and video.path:
            return _get_video_frames_from_path(video.path, num_frames, adaptive)
        raise


# ════════════════════════════════════════════════════════════════════════════
#  IMAGE EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def _get_image(item: Dict[str, Any]) -> Optional[Image.Image]:
    """Extract PIL image from dataset item."""
    if "image" in item:
        img = item["image"]
        if isinstance(img, dict):
            if "bytes" in img and img["bytes"]:
                from io import BytesIO
                return Image.open(BytesIO(img["bytes"])).convert("RGB")
            elif "path" in img:
                return Image.open(img["path"]).convert("RGB")
        if img is not None and isinstance(img, Image.Image):
            return img
    
    for url_key in ["image_url", "coco_url"]:
        if url_key in item and item[url_key]:
            import urllib.request
            from io import BytesIO
            # Compute nodes are offline: use the copy download_images.py cached.
            local = os.path.join(IMAGE_CACHE_DIR, os.path.basename(item[url_key]))
            if os.path.exists(local):
                return Image.open(local).convert("RGB")
            with urllib.request.urlopen(item[url_key]) as r:
                return Image.open(BytesIO(r.read())).convert("RGB")
    
    if "images" in item and item["images"]:
        imgs = item["images"]
        if isinstance(imgs, list) and len(imgs) > 0:
            return _get_image({"image": imgs[0]})
    
    return None


# ════════════════════════════════════════════════════════════════════════════
#  TEXT/CAPTION EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def _get_caption(item: Dict[str, Any]) -> str:
    """Extract caption/text from dataset item."""
    # Priority order for text fields
    text_fields = [
        "caption", "question", "text", "prompt", "instruction",
        "query", "sentence", "description", "script",
    ]
    
    for field in text_fields:
        if field in item and item[field]:
            val = item[field]
            if isinstance(val, str):
                return val.strip()
            elif isinstance(val, list) and len(val) > 0:
                return str(val[0]).strip()
    
    # Handle Charades 'scene' field
    if "scene" in item and item["scene"]:
        return f"Describe what happens in this scene: {item['scene']}"
    
    # Handle Charades 'labels' field (action classes)
    if "labels" in item and item["labels"]:
        labels = item["labels"]
        if isinstance(labels, list):
            label_str = ", ".join(str(l) for l in labels[:3])
            return f"What actions are shown in this video? Activities include: {label_str}"
        return f"What actions are shown in this video?"
    
    # Handle nested structures
    if "sentences" in item:
        sentences = item["sentences"]
        if isinstance(sentences, dict) and "raw" in sentences:
            return sentences["raw"].strip()
        elif isinstance(sentences, list) and len(sentences) > 0:
            s = sentences[0]
            if isinstance(s, dict):
                return s.get("raw", s.get("text", "")).strip()
            return str(s).strip()
    
    if "captions" in item:
        caps = item["captions"]
        if isinstance(caps, list) and len(caps) > 0:
            c = caps[0]
            if isinstance(c, str):
                return c.strip()
            elif isinstance(c, dict):
                return c.get("raw", c.get("caption", c.get("text", ""))).strip()
        elif isinstance(caps, str):
            return caps.strip()
    
    # Default prompt for videos
    return "Describe what happens in this video."


# ════════════════════════════════════════════════════════════════════════════
#  TOKEN COMPUTATION
# ════════════════════════════════════════════════════════════════════════════

def compute_image_tokens(
    image: Image.Image,
    model_config,
    model_type: Optional[str] = None,
) -> int:
    """Compute ACTUAL image token count based on image dimensions."""
    if model_type is None:
        model_type = getattr(model_config, "model_type", "").lower()
    
    width, height = image.size
    
    if "qwen2_vl" in model_type or "qwen2-vl" in model_type:
        vision_config = getattr(model_config, "vision_config", None)
        patch_size = 14
        if vision_config:
            patch_size = getattr(vision_config, "patch_size", 14)
        
        min_pixels = 28 * 28
        max_pixels = 1280 * 28 * 28
        
        pixels = width * height
        if pixels < min_pixels:
            scale = (min_pixels / pixels) ** 0.5
        elif pixels > max_pixels:
            scale = (max_pixels / pixels) ** 0.5
        else:
            scale = 1.0
        
        new_w = max(28, (int(width * scale) // 28) * 28)
        new_h = max(28, (int(height * scale) // 28) * 28)
        
        n_patches_w = new_w // patch_size
        n_patches_h = new_h // patch_size
        n_tokens = (n_patches_w * n_patches_h) // 4  # 2x2 merge
        
        return max(n_tokens, 4)
    
    elif "llava" in model_type:
        return 576
    elif "phi3" in model_type or "phi-3" in model_type:
        return 768
    elif "internvl" in model_type:
        return 256
    else:
        vision_config = getattr(model_config, "vision_config", None)
        if vision_config:
            image_size = getattr(vision_config, "image_size", 336)
            patch_size = getattr(vision_config, "patch_size", 14)
            return (image_size // patch_size) ** 2
        return 576


def compute_video_tokens(
    num_frames: int,
    frame_width: int,
    frame_height: int,
    model_config,
    model_type: Optional[str] = None,
) -> int:
    """Compute ACTUAL video token count based on frame dimensions and count."""
    if model_type is None:
        model_type = getattr(model_config, "model_type", "").lower()
    
    if "qwen2_vl" in model_type or "qwen2-vl" in model_type:
        # Qwen2-VL: per-frame tokens with temporal merge
        dummy_image = Image.new('RGB', (frame_width, frame_height))
        per_frame_tokens = compute_image_tokens(dummy_image, model_config, model_type)
        temporal_merge = 2  # Qwen2-VL merges 2 frames temporally
        total_tokens = (per_frame_tokens * num_frames) // temporal_merge
        return max(total_tokens, num_frames)
    else:
        # Other models: simple per-frame tokens
        dummy_image = Image.new('RGB', (frame_width, frame_height))
        per_frame_tokens = compute_image_tokens(dummy_image, model_config, model_type)
        return per_frame_tokens * num_frames


def _estimate_image_tokens_static(model_config) -> int:
    """Static estimate for batching (doesn't require actual image)."""
    vision_config = getattr(model_config, "vision_config", None)
    if vision_config is not None:
        image_size = getattr(vision_config, "image_size", None)
        patch_size = getattr(vision_config, "patch_size", None)
        if image_size and patch_size:
            return (image_size // patch_size) ** 2

    model_type = getattr(model_config, "model_type", "").lower()
    if "llava" in model_type:
        return 576
    elif "qwen2_vl" in model_type:
        return 1280
    return 576


def _estimate_video_tokens_static(model_config, num_frames: int = 16) -> int:
    """Static estimate for batching."""
    model_type = getattr(model_config, "model_type", "").lower()
    if "qwen2_vl" in model_type:
        return 256 * num_frames
    else:
        return _estimate_image_tokens_static(model_config) * num_frames


# ════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def _estimate_params_billion(arch_cfg, num_layers, hidden_size,
                             num_attention_heads, num_kv_heads,
                             intermediate_size, vocab_size):
    """Estimate total and active parameters (in billions).

    Handles dense and Mixture-of-Experts FFNs. For MoE, *total* counts every
    expert while *active* counts only the experts routed per token (plus any
    always-on shared experts). Assumes a gated (SwiGLU-style) FFN with 3 weight
    matrices, which covers Llama, Mistral, Mixtral, Qwen, etc.

    Returns (total_billion, active_billion, num_experts, experts_per_tok).
    """
    head_dim = hidden_size // max(num_attention_heads, 1)
    # Attention: q_proj + o_proj (hidden×hidden) + k_proj + v_proj (hidden×kv_dim)
    kv_dim = num_kv_heads * head_dim
    attn_per_layer = 2 * hidden_size * hidden_size + 2 * hidden_size * kv_dim

    # MoE detection across common config field names
    num_experts = (getattr(arch_cfg, "num_local_experts", None)
                   or getattr(arch_cfg, "num_experts", None)
                   or getattr(arch_cfg, "n_routed_experts", None)
                   or 1)
    experts_per_tok = (getattr(arch_cfg, "num_experts_per_tok", None)
                       or getattr(arch_cfg, "num_experts_per_token", None)
                       or getattr(arch_cfg, "moe_top_k", None)
                       or (1 if num_experts == 1 else 2))
    n_shared = getattr(arch_cfg, "n_shared_experts", 0) or 0

    ffn_one_expert = 3 * hidden_size * intermediate_size  # gated SwiGLU: gate/up/down
    if num_experts > 1:
        total_ffn = (num_experts + n_shared) * ffn_one_expert
        active_ffn = (experts_per_tok + n_shared) * ffn_one_expert
        router = hidden_size * num_experts
    else:
        total_ffn = active_ffn = ffn_one_expert
        router = 0

    embed = vocab_size * hidden_size
    if not getattr(arch_cfg, "tie_word_embeddings", True):
        embed *= 2  # separate lm_head when embeddings are untied

    total = (num_layers * (attn_per_layer + total_ffn + router) + embed) / 1e9
    active = (num_layers * (attn_per_layer + active_ffn + router) + embed) / 1e9
    return total, active, int(num_experts), int(experts_per_tok)


def _accel():
    """torch.cuda on NVIDIA, torch.xpu on Intel GPUs."""
    from .energy_intel import xpu_available
    return torch.xpu if xpu_available() else torch.cuda


def get_free_vram_gb():
    acc = _accel()
    acc.synchronize()
    try:
        free, total = acc.mem_get_info()
    except Exception:   # older torch.xpu without mem_get_info
        total = acc.get_device_properties(0).total_memory
        free = total - acc.memory_reserved(0)
    return free / (1024 ** 3), total / (1024 ** 3)


def estimate_kv_cache_gb(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    max_model_len: int,
    max_num_seqs: int,
    dtype_bytes: int,
) -> float:
    bytes_per_token = 2 * num_layers * num_heads * head_dim * dtype_bytes
    total_bytes = bytes_per_token * max_model_len * max_num_seqs
    return total_bytes / (1024 ** 3)


# ════════════════════════════════════════════════════════════════════════════
#  WORKLOAD & DATA EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def build_workload(
    prompt: str,
    heavy_cfg: Dict[str, Any],
    light_cfg: Dict[str, Any],
    modality: Modality,
    batch_size: int = 1,
    arrival_rate: float = 1.0,
    tokenizer=None,
    total_input_tokens: Optional[int] = None,
    image_resolution: Optional[int] = None,
    aspect_ratio: Optional[float] = None,
    actual_image_tokens: Optional[int] = None,
    num_video_frames: Optional[int] = None,
    video_duration_sec: Optional[float] = None,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    actual_video_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Build workload dict for a batch."""
    input_ids = tokenizer(prompt).input_ids if tokenizer else []

    workload = {
        "modality": modality.value,
        "modality_name": modality.name,
        "input_length": total_input_tokens if total_input_tokens else len(input_ids),
        "output_length": light_cfg.get("max_tokens", None),
        "batch_size": batch_size,
        "arrival_rate": arrival_rate,
        "max_num_batched_tokens": heavy_cfg.get("max_num_batched_tokens", 0),
        "max_num_seqs": heavy_cfg.get("max_num_seqs", DEFAULT_MAX_NUM_SEQS),
        # Image metrics
        "image_resolution": image_resolution or 0,
        "aspect_ratio": aspect_ratio or 0.0,
        "actual_image_tokens": actual_image_tokens or 0,
        # Video metrics (these will now vary!)
        "num_video_frames": num_video_frames or 0,
        "video_duration_sec": video_duration_sec or 0.0,
        "frame_width": frame_width or 0,
        "frame_height": frame_height or 0,
        "actual_video_tokens": actual_video_tokens or 0,
    }

    return workload


def _resize_frames(
    frames: List[Image.Image],
    target_short_side: int,
) -> List[Image.Image]:
    """Resize frames so the shorter side equals target_short_side, preserving aspect ratio."""
    resized = []
    for frame in frames:
        w, h = frame.size
        if min(w, h) == target_short_side:
            resized.append(frame)
            continue
        scale = target_short_side / min(w, h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        resized.append(frame.resize((new_w, new_h), Image.LANCZOS))
    return resized


def extract_item_data(
    item: Dict[str, Any],
    modality: Modality,
    num_video_frames: int = 16,
    model_config = None,
    target_video_resolution: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract all relevant data from a dataset item.

    Args:
        target_video_resolution: If set, resize video frames so their shorter
            side equals this value (e.g. 240, 360, 480, 720). Useful for
            testing the effect of resolution on token count and latency.
    """
    
    data = {
        "text": _get_caption(item),
        "image": None,
        "video_frames": None,
        "image_resolution": 0,
        "aspect_ratio": 0.0,
        "actual_image_tokens": 0,
        "num_video_frames": 0,
        "video_duration_sec": 0.0,
        "frame_width": 0,
        "frame_height": 0,
        "actual_video_tokens": 0,
    }
    
    if modality == Modality.TEXT_IMAGE:
        img = _get_image(item)
        if img is not None:
            data["image"] = img
            w, h = img.size
            data["image_resolution"] = max(w, h)
            data["aspect_ratio"] = w / h
            
            if model_config is not None:
                data["actual_image_tokens"] = compute_image_tokens(img, model_config)
            else:
                data["actual_image_tokens"] = (w // 14) * (h // 14) // 4
                
            debug_log(f"Image: {w}x{h}, tokens={data['actual_image_tokens']}")
    
    elif modality in (Modality.VIDEO_ONLY, Modality.TEXT_VIDEO):
        frames = None
        video_meta = None
        
        # Method 1: Try to get video bytes and decode
        video_bytes = _get_video_bytes(item)
        if video_bytes is not None:
            debug_log(f"Got video bytes: {len(video_bytes)/1024:.1f} KB")
            try:
                frames, video_meta = _get_video_frames_from_bytes(
                    video_bytes, 
                    num_frames=num_video_frames,
                    adaptive=True,
                )
            except Exception as e:
                debug_log(f"Failed to decode video bytes: {e}")
        
        # Method 2: Try video path directly
        if frames is None:
            video_path = _get_video_path_from_item(item)
            if video_path and os.path.exists(video_path):
                debug_log(f"Using video path: {video_path}")
                try:
                    frames, video_meta = _get_video_frames_from_path(
                        video_path,
                        num_frames=num_video_frames,
                        adaptive=True,
                    )
                except Exception as e:
                    debug_log(f"Failed to decode video path: {e}")
        
        # Method 3: Try HuggingFace Video feature directly
        if frames is None and "video" in item and item["video"] is not None:
            video = item["video"]
            if hasattr(video, 'get_frame_at') or hasattr(video, 'get_frames_in_range'):
                debug_log("Using HuggingFace Video feature")
                try:
                    frames, video_meta = _get_video_frames_from_hf_video(
                        video,
                        num_frames=num_video_frames,
                        adaptive=True,
                    )
                except Exception as e:
                    debug_log(f"Failed to decode HF Video: {e}")
        
        # Process extracted frames
        if frames is not None and video_meta is not None:
            if target_video_resolution is not None:
                frames = _resize_frames(frames, target_video_resolution)
                debug_log(f"Resized frames to short-side={target_video_resolution}: "
                         f"{frames[0].width}x{frames[0].height}")
            data["video_frames"] = frames
            data["num_video_frames"] = video_meta.get("num_frames", len(frames))
            data["video_duration_sec"] = video_meta.get("duration_sec", 0.0)
            data["frame_width"] = frames[0].width if frames else video_meta.get("frame_width", 0)
            data["frame_height"] = frames[0].height if frames else video_meta.get("frame_height", 0)
            
            debug_log(f"Video: {data['num_video_frames']} frames, "
                     f"{data['frame_width']}x{data['frame_height']}, "
                     f"duration={data['video_duration_sec']:.1f}s")
            
            if model_config is not None and data["frame_width"] > 0:
                data["actual_video_tokens"] = compute_video_tokens(
                    num_frames=data["num_video_frames"],
                    frame_width=data["frame_width"],
                    frame_height=data["frame_height"],
                    model_config=model_config,
                )
                debug_log(f"Video tokens: {data['actual_video_tokens']}")
        else:
            debug_log("Could not extract video frames from any source")
    
    return data


# ════════════════════════════════════════════════════════════════════════════
#  BATCHING
# ════════════════════════════════════════════════════════════════════════════

def build_token_batches(
    ds,
    tokenizer,
    max_num_batched_tokens: int,
    modality: Modality,
    image_tokens: int = 576,
    video_tokens_per_frame: int = 256,
    num_video_frames: int = 16,
    model_config = None,
    target_video_resolution: Optional[int] = None,
):
    """Group dataset items into batches based on token count."""
    current_batch = []
    current_tokens = 0

    for i, item in enumerate(ds):
        if i % 10 == 0:
            debug_log(f"Processing item {i}/{len(ds)}")

        item_data = extract_item_data(item, modality, num_video_frames, model_config,
                                      target_video_resolution=target_video_resolution)
        text = item_data["text"]
        
        text_tokens = len(tokenizer(text).input_ids) if text else 0
        
        if modality == Modality.TEXT_ONLY:
            token_count = text_tokens
        elif modality == Modality.TEXT_IMAGE:
            img_tokens = item_data.get("actual_image_tokens", 0) or image_tokens
            token_count = text_tokens + img_tokens
        elif modality in (Modality.VIDEO_ONLY, Modality.TEXT_VIDEO):
            vid_tokens = item_data.get("actual_video_tokens", 0)
            if vid_tokens == 0:
                vid_tokens = video_tokens_per_frame * num_video_frames
            token_count = text_tokens + vid_tokens
        else:
            token_count = text_tokens
        
        item_data["token_count"] = token_count
        
        if token_count >= max_num_batched_tokens:
            if current_batch:
                yield current_batch, current_tokens
                current_batch = []
                current_tokens = 0
            yield [item_data], token_count
            continue
        
        if current_tokens + token_count > max_num_batched_tokens:
            yield current_batch, current_tokens
            current_batch = []
            current_tokens = 0
        
        current_batch.append(item_data)
        current_tokens += token_count
    
    if current_batch:
        yield current_batch, current_tokens


def build_sentence_batches(
    ds,
    max_sentences: int,
    modality: Modality,
    tokenizer=None,
    image_tokens: int = 576,
    video_tokens_per_frame: int = 256,
    num_video_frames: int = 16,
    model_config = None,
    target_video_resolution: Optional[int] = None,
):
    """Group dataset items into batches by sentence count."""
    current_batch = []
    current_tokens = 0

    for item in ds:
        item_data = extract_item_data(item, modality, num_video_frames, model_config,
                                      target_video_resolution=target_video_resolution)
        text = item_data["text"]
        
        if tokenizer:
            text_tokens = len(tokenizer(text).input_ids) if text else 0
            if modality == Modality.TEXT_ONLY:
                token_count = text_tokens
            elif modality == Modality.TEXT_IMAGE:
                img_tokens = item_data.get("actual_image_tokens", 0) or image_tokens
                token_count = text_tokens + img_tokens
            else:
                vid_tokens = item_data.get("actual_video_tokens", 0)
                if vid_tokens == 0:
                    vid_tokens = video_tokens_per_frame * num_video_frames
                token_count = text_tokens + vid_tokens
        else:
            token_count = 0
        
        item_data["token_count"] = token_count
        
        if len(current_batch) >= max_sentences:
            yield current_batch, current_tokens
            current_batch = []
            current_tokens = 0
        
        current_batch.append(item_data)
        current_tokens += token_count
    
    if current_batch:
        yield current_batch, current_tokens


# ════════════════════════════════════════════════════════════════════════════
#  PROGRESS & CSV MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════

def init_progress_tracking(heavy_configs, light_configs, output_dir="./results"):
    progress_file = os.path.join(output_dir, "profiling_progress.json")

    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                progress = json.load(f)
            print(f"Resuming. Completed: {len(progress['completed_configs'])}")
            return progress, progress_file
        except json.JSONDecodeError:
            backup = f"{progress_file}.corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(progress_file, backup)

    progress = {
        "start_time": datetime.now().isoformat(),
        "completed_configs": [],
        "failed_configs": [],
        "total_configs": len(heavy_configs) * len(light_configs),
        "csv_fields": None,
    }
    return progress, progress_file


def init_csv_fields(heavy_configs, light_configs, progress):
    if progress.get("csv_fields"):
        return progress["csv_fields"]

    # "power" is the time-weighted mean over the window, summed over the GPUs
    # the engine occupies; "energy" is the Zeus counter over those same GPUs,
    # divided by batch size. n_gpus_measured/power_source record which
    # instrument produced them, so a run that silently fell back to two-point
    # device-0 sampling is identifiable after the fact instead of quietly
    # mixing conventions the way results/ and results_zeus/ now do.
    # max_generated_tokens: longest output in the batch (= decode steps), since
    # requests that stop early make max_tokens an overestimate.
    fixed_fields = ["prompt", "generated_tokens", "max_generated_tokens", "latency",
                    "power", "energy", "n_gpus_measured", "power_source", "measurement_repeats"]
    
    workload_fields = [
        "modality", "modality_name", "input_length",
        "image_resolution", "aspect_ratio", "actual_image_tokens",
        "num_video_frames", "video_duration_sec", "frame_width", 
        "frame_height", "actual_video_tokens",
        "batch_size", "arrival_rate", "max_num_batched_tokens", "max_num_seqs",
    ]
    
    model_fields = [
        "num_layers", "hidden_size", "num_attention_heads", "num_kv_heads",
        "head_dim", "intermediate_size", "model_params_billion",
        "active_params_billion", "num_experts", "num_experts_per_tok", "vocab_size",
        "image_token_count_estimate", "video_token_count_estimate",
    ]
    
    heavy_fields = sorted(set().union(*[set(cfg.keys()) for cfg in heavy_configs]))
    light_fields = sorted(set().union(*[set(cfg.keys()) for cfg in light_configs]))

    csv_fields = {
        "fixed": fixed_fields,
        "workload": workload_fields,
        "model": model_fields,
        "heavy": heavy_fields,
        "light": light_fields,
        "all": fixed_fields + workload_fields + model_fields + heavy_fields + light_fields,
    }

    progress["csv_fields"] = csv_fields
    return csv_fields


def init_csv_file(csv_path, csv_fields):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_fields["all"])


def save_progress(progress_file, progress):
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2, cls=NumpyEncoder)


def _build_model_message(item_data: Dict[str, Any], modality: Modality) -> Dict[str, Any]:
    """Build message dict for the model."""
    message = {"text": item_data["text"]}
    
    if modality == Modality.TEXT_IMAGE and item_data.get("image"):
        message["image"] = item_data["image"]
    elif modality in (Modality.VIDEO_ONLY, Modality.TEXT_VIDEO):
        if item_data.get("video_frames"):
            message["video_frames"] = item_data["video_frames"]
    
    return message


class EngineDead(RuntimeError):
    """The vLLM engine died mid-config (e.g. a TP/PP worker hit an illegal
    memory access); every later generate() on it would fail too."""


def _engine_dead(e):
    return (type(e).__name__ == "EngineDeadError" or "EngineCore" in str(e)
            or "CUDA error" in str(e))


MAX_ENGINE_RESTARTS = 3


def _process_batch_and_write(
    model,
    batch_items: List[Dict[str, Any]],
    modality: Modality,
    light_cfg: Dict[str, Any],
    heavy_cfg: Dict[str, Any],
    workload: Dict[str, Any],
    model_info: Dict[str, Any],
    csv_fields: Dict[str, List[str]],
    writer,
    f,
) -> bool:
    """Process a batch and write results to CSV."""
    batch_size = len(batch_items)
    first_text = batch_items[0]["text"]
    
    messages = [_build_model_message(item, modality) for item in batch_items]

    try:
        _, tokens, metrics = model.generate(
            messages=messages,
            sampling_params=light_cfg,
        )
        latency = metrics.get("total_latency")
        avg_power = metrics.get("avg_power_watts")
        energy = metrics.get("energy_joules")
        if energy != "":
            energy = energy / batch_size
        measured = {
            "generated_tokens": tokens,
            "max_generated_tokens": metrics.get("max_generated_tokens", ""),
            "latency": latency,
            "power": avg_power,
            "energy": energy,
            "n_gpus_measured": metrics.get("n_gpus_measured", ""),
            "power_source": metrics.get("power_source", ""),
            "measurement_repeats": metrics.get("measurement_repeats", 1),
        }
    except EngineHangError:
        # Engine deadlocked — unrecoverable in-process. Do not swallow: let it
        # propagate so the run aborts fast instead of hanging until walltime.
        raise
    except Exception as e:
        if _engine_dead(e):
            raise EngineDead(str(e)) from e
        print(f"Generation error: {e}")
        measured = {}

    prompt_summary = f"[batch of {batch_size}] {first_text[:50]}..."

    def _val(v):
        return "none" if v is None else v

    # Driven by csv_fields rather than a fixed list: the header is cached in the
    # progress file, so a run resumed against an older checkpoint keeps writing
    # that run's columns instead of desyncing.
    row = [prompt_summary]
    for k in csv_fields["fixed"]:
        if k != "prompt":
            row.append(_val(measured.get(k, "")))

    for k in csv_fields["workload"]:
        row.append(_val(workload.get(k, "")))
    for k in csv_fields["model"]:
        row.append(_val(model_info.get(k, "")))
    for k in csv_fields["heavy"]:
        row.append(_val(heavy_cfg.get(k, "")))
    for k in csv_fields["light"]:
        row.append(_val(light_cfg.get(k, "")))

    writer.writerow(row)
    f.flush()
    os.fsync(f.fileno())
    return True


def load_model_meta(model_name, trust_remote_code=False, num_video_frames=16):
    """Architecture facts written to every CSV row and used for KV-cache estimates."""
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    # Extract model architecture. Multimodal configs nest the language model under
    # different keys: HF VLMs use `text_config`/`get_text_config()`, while InternVL
    # uses `llm_config` (and some checkpoints `language_config`). Pick whichever
    # nested config actually carries the transformer dimensions.
    def _has_dims(cfg):
        nl = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))
        hs = getattr(cfg, "hidden_size", getattr(cfg, "d_model", 0))
        return bool(nl) and bool(hs)

    arch_cfg = model_config
    if not _has_dims(arch_cfg):
        candidates = []
        if hasattr(model_config, "get_text_config"):
            try:
                candidates.append(model_config.get_text_config())
            except Exception:
                pass
        for attr in ("text_config", "llm_config", "language_config", "llm_cfg"):
            candidates.append(getattr(model_config, attr, None))
        for cand in candidates:
            if cand is not None and _has_dims(cand):
                arch_cfg = cand
                break

    num_attention_heads = getattr(arch_cfg, "num_attention_heads", getattr(arch_cfg, "n_head", 1))
    num_layers = getattr(arch_cfg, "num_hidden_layers", getattr(arch_cfg, "n_layer", 0))
    hidden_size = getattr(arch_cfg, "hidden_size", getattr(arch_cfg, "d_model", 0))
    num_kv_heads = getattr(arch_cfg, "num_key_value_heads", num_attention_heads)
    intermediate_size = getattr(arch_cfg, "intermediate_size", 0)
    vocab_size = getattr(arch_cfg, "vocab_size", getattr(model_config, "vocab_size", 0))
    has_attention = hasattr(arch_cfg, "num_attention_heads") or hasattr(arch_cfg, "n_head")

    if num_layers <= 0 or hidden_size <= 0:
        raise ValueError(f"Cannot infer model dimensions for {model_name}")

    head_dim = hidden_size // num_attention_heads
    params_total, params_active, num_experts, experts_per_tok = _estimate_params_billion(
        arch_cfg, num_layers, hidden_size, num_attention_heads, num_kv_heads,
        intermediate_size, vocab_size,
    )

    model_info = {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "model_params_billion": round(params_total, 3),
        "active_params_billion": round(params_active, 3),
        "num_experts": num_experts,
        "num_experts_per_tok": experts_per_tok,
        "vocab_size": vocab_size,
        "image_token_count_estimate": _estimate_image_tokens_static(model_config),
        "video_token_count_estimate": _estimate_video_tokens_static(model_config, num_video_frames),
    }

    return {
        "model_config": model_config, "model_info": model_info, "has_attention": has_attention,
        "num_attention_heads": num_attention_heads, "num_layers": num_layers, "head_dim": head_dim,
    }


# ════════════════════════════════════════════════════════════════════════════
#  MAIN PROFILING FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def profile_workloads(
    ds,
    heavy_configs: List[Dict[str, Any]],
    light_configs: List[Dict[str, Any]],
    model_name: str,
    modality: Optional[Modality] = None,
    trust_remote_code: bool = False,
    max_model_len: int = 2048,
    num_samples_per_config: int = 1,
    num_video_frames: int = 16,
    output_dir: str = "./results",
    target_video_resolution: Optional[int] = None,
):
    """Profile workloads across configurations.

    Args:
        target_video_resolution: Resize video frames so their shorter side
            equals this value before feeding to the model. None = native.
    """
    
    # Dataset inspection
    print("\n" + "=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)
    try:
        first_item = ds[0]
    except RuntimeError as _e:
        if "torchcodec" not in str(_e) and "libtorchcodec" not in str(_e):
            raise
        # torchcodec unavailable (missing libnvrtc / FFmpeg); read raw Arrow data
        # to avoid triggering the video decoder during inspection
        raw = ds.data.slice(0, 1).to_pydict()
        first_item = {k: v[0] for k, v in raw.items()}
    print(f"Dataset size: {len(ds)} items")
    print(f"Fields: {list(first_item.keys())}")
    
    # Check video format
    if "video" in first_item:
        video = first_item["video"]
        if isinstance(video, dict):
            if "bytes" in video and video["bytes"]:
                print(f"Video format: embedded bytes ({len(video['bytes'])/1024:.1f} KB)")
            elif "path" in video:
                print(f"Video format: file path")
        elif isinstance(video, bytes):
            print(f"Video format: direct bytes ({len(video)/1024:.1f} KB)")
    print("=" * 70 + "\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    if modality is None:
        modality = detect_modality(first_item)
        print(f"Auto-detected modality: {modality.name}")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "profiling_metrics.csv")

    progress, progress_file = init_progress_tracking(heavy_configs, light_configs, output_dir)
    csv_fields = init_csv_fields(heavy_configs, light_configs, progress)
    init_csv_file(csv_path, csv_fields)

    meta = load_model_meta(model_name, trust_remote_code, num_video_frames)
    model_config, model_info = meta["model_config"], meta["model_info"]
    has_attention, num_attention_heads = meta["has_attention"], meta["num_attention_heads"]
    num_layers, head_dim = meta["num_layers"], meta["head_dim"]

    num_gpus = _accel().device_count()
    n_warmup = 2

    for h_idx, heavy_cfg in enumerate(heavy_configs):
        all_light_done = all(
            f"h{h_idx}_l{l_idx}" in progress["completed_configs"]
            for l_idx in range(len(light_configs))
        )
        if all_light_done:
            continue

        tp_size = heavy_cfg.get("tensor_parallel_size", 1)
        pp_size = heavy_cfg.get("pipeline_parallel_size", 1)

        if tp_size * pp_size > num_gpus:
            for l_idx in range(len(light_configs)):
                progress["completed_configs"].append(f"h{h_idx}_l{l_idx}")
            save_progress(progress_file, progress)
            continue

        if has_attention and num_attention_heads % tp_size != 0:
            for l_idx in range(len(light_configs)):
                progress["completed_configs"].append(f"h{h_idx}_l{l_idx}")
            save_progress(progress_file, progress)
            continue

        free_gb, _ = get_free_vram_gb()
        dtype_bytes = 4 if heavy_cfg.get("dtype") == "float32" else 2

        # vLLM pages the KV cache and pre-empts requests when it is full, so an
        # engine only needs room for ONE max-length sequence; requiring room for
        # max_num_seqs of them skipped every high-concurrency config (S=512 on
        # Llama-8B "needed" ~1 TB). KV heads, not attention heads, size the cache.
        kv_gb = 0.0
        if has_attention:
            kv_gb = estimate_kv_cache_gb(
                num_layers, model_info.get("num_kv_heads") or num_attention_heads, head_dim,
                max_model_len, 1, dtype_bytes
            )

        if free_gb < kv_gb + 2.0:
            for l_idx in range(len(light_configs)):
                progress["completed_configs"].append(f"h{h_idx}_l{l_idx}")
            save_progress(progress_file, progress)
            continue

        heavy_cfg.setdefault("dtype", "bfloat16")

        # A failure while building or exercising one heavy config (a worker
        # killed by OOM or a CUDA fault, or an engine deadlock caught by the
        # generate() watchdog) must not abort the entire sweep. If the engine
        # dies mid-config, the batch it died on is skipped, the engine is
        # reloaded and the config resumes at the next batch (rows are
        # checkpointed per batch in progress["batches_done"]), up to
        # MAX_ENGINE_RESTARTS times. Any other failure marks the config's
        # remaining light configs failed so a resume skips them.
        batches_done = progress.setdefault("batches_done", {})
        skipped = progress.setdefault("skipped_batches", {})
        restarts = 0
        while True:
            # Refuse to load if another job is on our GPUs (see gpu_guard.py).
            # Outside the try below on purpose: a contention stop must not be
            # recorded as a failed config, or a resume would skip it forever.
            assert_gpus_exclusive(f"before loading config h{h_idx}")
            model = None
            try:
                model = Model(
                    name=model_name,
                    max_model_len=max_model_len,
                    trust_remote_code=trust_remote_code,
                    **heavy_cfg,
                )

                # Warmup
                for _ in range(n_warmup):
                    item_data = extract_item_data(ds[0], modality, num_video_frames, model_config,
                                                  target_video_resolution=target_video_resolution)
                    warmup_msg = _build_model_message(item_data, modality)
                    model.generate(messages=warmup_msg, sampling_params=light_configs[0])

                for l_idx, light_cfg in enumerate(light_configs):
                    config_id = f"h{h_idx}_l{l_idx}"
                    if config_id in progress["completed_configs"]:
                        continue

                    first = batches_done.get(config_id, 0)
                    # Rows of this session start here; on a contention stop
                    # they are cut off so the resume redoes them cleanly.
                    csv_start = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0

                    try:
                        with open(csv_path, "a", newline="") as f:
                            writer = csv.writer(f)

                            if "max_num_batched_tokens" in heavy_cfg:
                                batch_iter = build_token_batches(
                                    ds, tokenizer, heavy_cfg["max_num_batched_tokens"],
                                    modality,
                                    image_tokens=model_info["image_token_count_estimate"],
                                    video_tokens_per_frame=(
                                        model_info["video_token_count_estimate"] // num_video_frames
                                        if num_video_frames > 0 else 0
                                    ),
                                    num_video_frames=num_video_frames,
                                    model_config=model_config,
                                    target_video_resolution=target_video_resolution,
                                )
                            elif "max_num_seqs" in heavy_cfg:
                                batch_iter = build_sentence_batches(
                                    ds, heavy_cfg["max_num_seqs"], modality,
                                    tokenizer=tokenizer,
                                    image_tokens=model_info["image_token_count_estimate"],
                                    video_tokens_per_frame=(
                                        model_info["video_token_count_estimate"] // num_video_frames
                                        if num_video_frames > 0 else 0
                                    ),
                                    num_video_frames=num_video_frames,
                                    model_config=model_config,
                                    target_video_resolution=target_video_resolution,
                                )
                            else:
                                batch_iter = (
                                    ([extract_item_data(item, modality, num_video_frames, model_config,
                                                        target_video_resolution=target_video_resolution)], 0)
                                    for item in ds
                                )

                            for b_idx, (batch_items, total_tokens) in enumerate(batch_iter):
                                if b_idx >= num_samples_per_config:
                                    break
                                if b_idx < first:
                                    continue

                                first_item = batch_items[0]

                                workload = build_workload(
                                    prompt=first_item["text"],
                                    heavy_cfg=heavy_cfg,
                                    light_cfg=light_cfg,
                                    modality=modality,
                                    batch_size=len(batch_items),
                                    total_input_tokens=total_tokens,
                                    tokenizer=tokenizer,
                                    image_resolution=first_item.get("image_resolution"),
                                    aspect_ratio=first_item.get("aspect_ratio"),
                                    actual_image_tokens=first_item.get("actual_image_tokens", 0),
                                    num_video_frames=first_item.get("num_video_frames", 0),
                                    video_duration_sec=first_item.get("video_duration_sec", 0.0),
                                    frame_width=first_item.get("frame_width", 0),
                                    frame_height=first_item.get("frame_height", 0),
                                    actual_video_tokens=first_item.get("actual_video_tokens", 0),
                                )

                                assert_gpus_exclusive(f"during {config_id}")
                                try:
                                    _process_batch_and_write(
                                        model, batch_items, modality, light_cfg, heavy_cfg,
                                        workload, model_info, csv_fields, writer, f
                                    )
                                except EngineDead:
                                    skipped.setdefault(config_id, []).append(b_idx)
                                    batches_done[config_id] = b_idx + 1
                                    save_progress(progress_file, progress)
                                    raise
                                batches_done[config_id] = b_idx + 1
                                save_progress(progress_file, progress)
                        # A job that arrived during the last batch shows up here.
                        assert_gpus_exclusive(f"after {config_id}")
                    except GPUContentionError:
                        os.truncate(csv_path, csv_start)
                        batches_done[config_id] = first
                        save_progress(progress_file, progress)
                        print(f"[gpu_guard] removed the partial rows of {config_id} "
                              f"from {csv_path}", flush=True)
                        raise

                    progress["completed_configs"].append(config_id)
                    save_progress(progress_file, progress)
                break

            except GPUContentionError:
                raise                      # not a config failure: stop, don't mark
            except Exception as e:
                restarts += 1
                if isinstance(e, EngineDead) and restarts <= MAX_ENGINE_RESTARTS:
                    print(f"[config h{h_idx}] engine died ({e}); skipped that batch, "
                          f"reloading ({restarts}/{MAX_ENGINE_RESTARTS})", flush=True)
                    continue
                # Includes EngineHangError (deadlock watchdog) and hard engine
                # failures such as a TP worker dying during warm-up. Mark this
                # heavy config's remaining light configs done so a resume skips
                # them, record the failure for visibility, and carry on.
                print(f"[config h{h_idx}] FAILED: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                progress.setdefault("failed_configs", [])
                for l_idx in range(len(light_configs)):
                    cid = f"h{h_idx}_l{l_idx}"
                    if cid not in progress["completed_configs"]:
                        progress["completed_configs"].append(cid)
                    if cid not in progress["failed_configs"]:
                        progress["failed_configs"].append(cid)
                save_progress(progress_file, progress)
                break

            finally:
                if model is not None:
                    try:
                        model.close()
                    except Exception:
                        pass
                    model = None
                gc.collect()
                try:
                    _accel().synchronize()
                except Exception:
                    # The CUDA context may be dead after a worker crash; ignore.
                    pass
                try:
                    _accel().empty_cache()
                except Exception:
                    pass
                time.sleep(10)
