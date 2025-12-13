import os
BASE_DIR = r"D:\ISL_Translate\data"

# YOUR PATHS (CHANGE ONLY THESE)
METADATA_CSV   = os.path.join(BASE_DIR, "iSign_v1.1.csv")  # contains metadata about videos and poses
RAW_POSES_DIR  = os.path.join(BASE_DIR, "extracted_poses&videos\iSign-poses_v1.1")        # contains .pose files
VIDEOS_DIR     = os.path.join(BASE_DIR, "extracted_poses&videos\iSign-videos_v1.1")          # contains .mp4 files

# AUTO CREATED
WORKSPACE      = os.path.join(BASE_DIR, "workspace")
POSES_NPY      = os.path.join(WORKSPACE, "poses_npy")

# Settings
MAX_FRAMES = 120
# Lower batch size to avoid OOM on machines without sufficient RAM/GPU memory
# Increase this if you train on a GPU with more memory.
BATCH_SIZE = 16
EPOCHS     = 5
os.makedirs(WORKSPACE, exist_ok=True)
os.makedirs(POSES_NPY, exist_ok=True)