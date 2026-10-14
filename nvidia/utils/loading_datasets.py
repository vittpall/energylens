from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download
from source.dataloader import Dataloader
from source.profiling import profile_workloads, Modality
import tarfile
from pathlib import Path
import os

def load_ucf101_subset(num_samples=200, extract_dir="/tmp/ucf101_subset"):
    """
    Download and load UCF101 subset as a HuggingFace dataset.
    This dataset has 405 videos stored locally (no YouTube needed).
    Varying resolutions and durations for good token variation!
    """
    # Download the tar.gz file
    print("Downloading UCF101 subset...")
    file_path = hf_hub_download(
        repo_id="sayakpaul/ucf101-subset",
        filename="UCF101_subset.tar.gz",
        repo_type="dataset"
    )
    
    # Extract
    dataset_path = Path(extract_dir) / "UCF101_subset"
    if not dataset_path.exists():
        print(f"Extracting to: {extract_dir}")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(extract_dir)
    
    # Find all videos
    video_files = list(dataset_path.glob("**/*.avi")) + list(dataset_path.glob("**/*.mp4"))
    video_files = video_files[:num_samples]
    
    print(f"Found {len(video_files)} videos")
    
    # Build dataset with video bytes
    data = []
    for video_path in video_files:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        
        # Extract label from path (UCF101 structure: split/class/video.avi)
        label = video_path.parent.name
        
        data.append({
            "video": {"bytes": video_bytes, "path": str(video_path)},
            "label": label,
            "question": f"Describe the action in this video showing: {label.replace('_', ' ').lower()}",
        })
    
    ds = Dataset.from_list(data)
    print(f"Created dataset with {len(ds)} videos")
    return ds
