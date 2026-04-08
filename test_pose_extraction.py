from pathlib import Path
import sys
import os

os.chdir("d:/UGANDA SIGN LANGUAGE INSTRUCTOR/models_dataset1")
sys.path.insert(0, "d:/UGANDA SIGN LANGUAGE INSTRUCTOR/models_dataset1/ctr_gcn")
sys.path.insert(0, "d:/UGANDA SIGN LANGUAGE INSTRUCTOR/models_dataset1/shared")

from config import ensure_directories, KEYPOINT_DIR
from prepare_pose_data import extract_pose_tensor
import numpy as np

ensure_directories()

dataset_root = Path("d:/UGANDA SIGN LANGUAGE INSTRUCTOR/SIGN LANGUAGE DISEASES FINISHED")
videos = list(dataset_root.rglob("*.mp4"))

if videos:
    test_vid = videos[0]
    print(f"Testing on: {test_vid.name}")
    
    pose = extract_pose_tensor(test_vid, max_frames=50)
    print(f"Extracted shape: {pose.shape}")
    print(f"Sample keypoint[0,0,:]: {pose[0,0,:]}")
    
    save_path = KEYPOINT_DIR / f"test_sample.npy"
    np.save(save_path, pose)
    print(f"✓ Saved test to {save_path}")
    print(f"✓ Pose extraction works!")
else:
    print("No videos found")
