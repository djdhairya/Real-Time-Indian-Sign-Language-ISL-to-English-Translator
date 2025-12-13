# Real-Time Indian Sign Language (ISL) to English Translator


## 📌 Project Overview

This project develops an **end-to-end deep learning system** for translating **continuous Indian Sign Language (ISL)** gestures into **fluent English text** in real time.

The system relies exclusively on **pose keypoints extracted from webcam video**, ensuring a **privacy-preserving**, lightweight, and efficient translation pipeline suitable for **consumer-grade hardware**.

The goal is to bridge communication gaps for **India’s 6+ million deaf and hard-of-hearing community**.

---

## 📊 Dataset & Model Summary

- **Dataset:** iSign v1.1  
- **Training Samples:** 1,27,236 annotated videos  
- **Model:** Transformer Encoder–Decoder with Cross-Attention  
- **Input:** Pose keypoints (MediaPipe-based)  
- **Output:** English text  
- **Key Innovation:** Pose-only translation (no raw video)

---

## 🧠 Model Architecture

- Pose keypoints extracted from video frames
- Temporal pose embedding
- Transformer Encoder for sequence understanding
- Transformer Decoder with Cross-Attention for text generation
- Token-level English sentence output

Architecture Type: **Transformer Encoder–Decoder with Cross-Attention**

---

## 🏆 Results

| Metric | Value |
|------|------|
| Training Samples | 1,27,236 |
| Epochs | 5 |
| Final Accuracy | **82.84%** |
| Final Loss | 1.5708 |
| Inference Time | ~2–4 seconds per sign |
| Real-Time Capable | Yes |

The model achieves **research-level accuracy** within only **5 epochs**, demonstrating strong generalization and efficiency.

---

## 📂 Dataset Details

- **Dataset Name:** iSign v1.1  
- **Source:** https://huggingface.co/datasets/Exploration-Lab/iSign/tree/main  

### System Requirements for Dataset
- Storage: **200GB+**
- RAM: **16GB minimum (32GB recommended)**
- GPU: **NVIDIA CUDA-enabled GPU**

---

## ⬇️ Dataset Download & Setup

1. Visit the dataset repository:
   ```
   https://huggingface.co/datasets/Exploration-Lab/iSign/tree/main
   ```

2. Download the following files:
   - `iSign-videos_v1.1.zip` (~150GB)
   - `iSign-poses_v1.1.zip` (~50GB)

3. Extract them into the project directory as shown below.

---

## 🗂️ Project Structure

```
ISL_Translate/
├── config.py                  # Paths, constants, vocabulary
├── train_agent.py             # Training pipeline
├── app.py                     # Live webcam demo
├── convert_pose_agent.py      # .pose → .npy converter
├── ISL_TRANSFORMER_FINAL.keras# Trained model
├── data/
│   ├── extracted_videos/
|   |   |--iSign-poses_v1.1
|   |   |--iSign-videos_v1.1      
│   └── workspace/
│   |    └── poses_npy/         # Pose NumPy arrays
|   ├── iSign_v1.1.csv             # Metadata
└── README.md                  # Documentation
```

---

## 🔄 Pose Conversion

To improve training efficiency, raw `.pose` files are converted into `.npy` format.

Run the conversion script if required:

```bash
python convert_pose_agent.py
```

⚠️ **Warning:**  
Pose extraction and conversion may take several hours and requires **200GB+ free storage**.

---

## ⚙️ Requirements

### Software
- Python 3.10+
- TensorFlow 2.16+ (CUDA enabled)
- OpenCV
- MediaPipe
- NumPy
- Pandas

### Installation
```bash
pip install tensorflow opencv-python mediapipe numpy pandas
```

### Hardware
- Storage: 200GB+
- RAM: 16GB minimum (32GB recommended)
- GPU: NVIDIA CUDA-enabled (MX330 or better)

---

## ▶️ How to Run

1. Place `ISL_TRANSFORMER_FINAL.keras` in the project root directory
2. Update dataset and model paths in `config.py`
3. Run the live demo:

```bash
python app.py
```

- Perform ISL gestures in front of the webcam
- Translation appears within **2–4 seconds**
- Press **Q** to exit

---

## 📸 Screenshots & Diagrams

- Model Architecture Diagram
<img width="823" height="443" alt="Screenshot 2025-12-13 084324" src="https://github.com/user-attachments/assets/93106c8f-196d-4594-9ec6-80e5c1193b71" />

- Training Accuracy vs Loss Graph
<img width="1973" height="878" alt="graph" src="https://github.com/user-attachments/assets/a09598f3-0b44-4983-8b43-d8ee94d2ecb4" />

- Activity Diagram
<img width="975" height="603" alt="Screenshot 2025-12-13 084832" src="https://github.com/user-attachments/assets/dc83606d-8b5b-42e6-9df8-2c7c98798268" />

- Live Translation Demo
<img width="417" height="389" alt="Screenshot 2025-12-13 084542" src="https://github.com/user-attachments/assets/3c85f520-e578-4b5a-91e0-8327a62d0fd0" />



---

## 🏅 Key Achievements

- Trained on the full public iSign dataset
- Privacy-preserving pose-only translation
- Real-time inference on consumer hardware
- Custom pose-to-NumPy conversion pipeline
- Clean and professional user interface
- High accuracy achieved in minimal epochs

---

## 🔮 Future Scope

- Hindi language translation
- Continuous sentence generation
- Mobile deployment (TensorFlow Lite)
- Voice output (Text-to-Speech)
- Multi-signer support


## Output
<img width="797" height="614" alt="Screenshot 2025-12-12 054925" src="https://github.com/user-attachments/assets/426bd572-7f34-48ac-8748-b8ecee300ee2" />
<img width="984" height="806" alt="Screenshot 2025-12-13 090618" src="https://github.com/user-attachments/assets/ecceea26-f4d0-4a39-ae22-76be65c43d86" />
<img width="799" height="633" alt="Screenshot 2025-12-12 051340" src="https://github.com/user-attachments/assets/2ea9c811-7df6-4e7b-98a2-8b20a2fd8687" />
<img width="1072" height="842" alt="Screenshot 2025-12-13 090637" src="https://github.com/user-attachments/assets/8d193c00-e2a5-4fcf-a0e5-4ca2b8a3d8a7" />











