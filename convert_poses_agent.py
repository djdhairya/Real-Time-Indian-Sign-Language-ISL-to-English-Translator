
from config import *
import numpy as np
import os
from tqdm import tqdm

def convert_pose(filepath):
    with open(filepath, "rb") as f:
        raw = f.read()
    if len(raw) < 543*5*4:
        return np.zeros((1, 297), dtype=np.float32)
    
    data = np.frombuffer(raw, dtype=np.float32)
    total_floats = len(data)
    frames = total_floats // (543 * 5)
    if frames == 0:
        return np.zeros((1, 297), dtype=np.float32)
    
    arr = data[:frames * 543 * 5].reshape(frames, 543, 5)

    indices = [
        0, 1, 4, 5,                    # face (4)
        *range(11, 44),                # pose 33 (11→43)
        *range(468, 489),              # left hand 21
        *range(522, 543)               # right hand 21
    ]
    selected = arr[:, indices, :3]     # (T, 99, 3)
    return selected.reshape(frames, 297).astype(np.float32)

print("CONVERTING .pose → .npy (297 features)")

files = [f for f in os.listdir(RAW_POSES_DIR) if f.endswith(".pose")]
for f in tqdm(files, desc="Converting"):
    uid = f.replace(".pose", "")
    out = os.path.join(POSES_NPY, f"{uid}.npy")
    if os.path.exists(out):
        continue
    try:
        seq = convert_pose(os.path.join(RAW_POSES_DIR, f))
        np.save(out, seq)
    except:
        np.save(out, np.zeros((1, 297), dtype=np.float32))

print(f"CONVERSION DONE → {len(files)} files (T, 297)")