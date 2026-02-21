# 📺 Video Temporal Error Detector

**NITT Hackathon Submission | Developed by Rohit**

This project is a high-speed system designed to detect and classify temporal inconsistencies in video streams. It identifies exactly where encoding or transmission failures occurred by analyzing **Motion Consistency** across consecutive frames.

### 🎯 Problem Statement Fulfillment

The system is designed to automatically classify every frame into one of three categories:

* **Normal**: Consistent motion flow matching the video's expected baseline.
* **Frame Drop**: Detected via massive motion intensity spikes (teleportation artifacts).
* **Frame Merge**: Identified by near-zero motion energy or unnatural temporal blending.

---

### 🧠 My Technical Approach: Motion Consistency Analysis

While the problem statement allows for timestamp analysis, I focused on **Motion Consistency** using **Mean Squared Error (MSE)** because it is more robust for detecting errors within the video content itself, even when the container timestamps appear "normal."

#### 1. Optimization for Accuracy

* **160p Signal Conditioning**: I downscale frames to  and apply a grayscale filter. This isolates the "Global Motion" and filters out high-frequency noise (like grass textures in sports or sensor grain), making the temporal errors much sharper in the data.

#### 2. Detection Logic

* **For Frame Drops**: I calculate the delta between Frame  and . If the ball or object jumps a physically impossible distance, the MSE creates a **Localized Spike**. I implemented a **Recovery Rule** to ensure the spike is a glitch (recovering at ) and not just high-speed motion.
* **For Frame Merges**: I look for "Motion Sinks"—where the MSE drops below 15% of the local median. This identifies frames that are identical or incorrectly combined by the encoder.

#### 3. Handling Environmental Noise (Adaptive Thresholding)

I wrote an **Adaptive Multiplier** logic to handle camera jitter. By calculating the **Standard Deviation ()** of the entire stream, the detector automatically adjusts its sensitivity. If the video is handheld/shaky, the threshold becomes stricter to prevent "Normal" movement from being flagged as a "Drop."

---

### 🛠️ The Tech Stack

* **OpenCV**: For frame extraction and seek-bar control.
* **PyTorch (CUDA Accelerated)**: All matrix subtractions are offloaded to the GPU to ensure the analysis is fast enough for long video streams.
* **Pandas**: For vectorized time-series analysis and signal smoothing.
* **Streamlit**: For the interactive "Forensic Spotlight" dashboard used to verify classified frames.

---

### 🏃 How to Run

1. **Setup Environment**:
```bash
pip install -r requirements.txt

```


2. **Execute System**:
```bash
streamlit run app.py

```



---

### 📊 Classification Output

The system generates a full report of every frame index, marking them as **Normal**, **Drop**, or **Merge**. Users can use the **Forensic Spotlight** slider to jump directly to any detected anomaly to verify the classification visually.
