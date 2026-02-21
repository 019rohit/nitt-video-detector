import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import tempfile
import os
import uuid

# --- Engine Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(page_title="Frame Drop and Merge Detector", layout="wide", page_icon="🏏")

@st.cache_data(show_spinner=False)
def analyze_momentum_final(file_path, _run_id):
    cap = cv2.VideoCapture(file_path)
    mse_list = []
    prev_gray = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        # Target 160p: Ball momentum focus ke liye
        gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            if DEVICE.type == 'cuda':
                t1 = torch.from_numpy(prev_gray).to(DEVICE).float()
                t2 = torch.from_numpy(gray).to(DEVICE).float()
                mse = torch.mean(torch.pow(t1 - t2, 2)).item()
            else:
                mse = np.mean((gray.astype(float) - prev_gray.astype(float))**2)
            mse_list.append(mse)
        else: mse_list.append(0.0)
        prev_gray = gray
    cap.release()
    return mse_list

def extract_frame(video_path, frame_no):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    if ret: return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

# --- UI Layout ---
st.title("Video Temporal Error Detection")

uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4"])

if uploaded_file:
    # Static path to stop React loops
    temp_path = os.path.join(tempfile.gettempdir(), f"nitt_full_{uploaded_file.name}")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    run_id = str(uuid.uuid4())
    data = analyze_momentum_final(temp_path, run_id)
    df = pd.DataFrame({'mse': data})
    
    # --- LOGIC (UNCHANGED ADAPTIVE) ---
    median_motion = df['mse'].median()
    std_dev = df['mse'].std()
    
    # Threshold Calculation: $Threshold = Median + (Multiplier \times StdDev)$
    drop_multiplier = 2.5 if std_dev > 15 else 1.8 
    
    df['prev'] = df['mse'].shift(1).fillna(0)
    df['next'] = df['mse'].shift(-1).fillna(0)

    is_drop = (df['mse'] > median_motion * drop_multiplier) & (df['mse'] > df['next']) & (df['mse'] > 20)
    drops = set(df[is_drop].index.tolist())

    is_merge = (df['mse'] < median_motion * 0.15) & (df['prev'] > 5.0)
    merges = set(df[is_merge].index.tolist())

    # --- TOP ROW: VIDEO & GRAPH ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📹 Video Playback")
        st.video(uploaded_file)
        
    with col2:
        st.subheader("📊 Momentum Graph")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=data, name="Momentum", line=dict(color='#00ff88', width=2)))
        # Marking detected anomalies
        if drops:
            fig.add_trace(go.Scatter(x=list(drops), y=df['mse'].iloc[list(drops)], mode='markers', name='Drop', marker=dict(color='red', size=12, symbol='star')))
        if merges:
            fig.add_trace(go.Scatter(x=list(merges), y=df['mse'].iloc[list(merges)], mode='markers', name='Merge', marker=dict(color='yellow', size=10, symbol='circle')))
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # --- FULL FRAME SPOTLIGHT ---
    st.divider()
    st.subheader("🔍 Full-Sequence Frame Spotlight")
    
    total_frames = len(data)
    # Slider to scrub through ALL frames
    selected_frame = st.slider("Scrub through all frames:", 1, total_frames, 1)
    
    c_info, c_img = st.columns([1, 2])
    
    with c_info:
        current_idx = selected_frame - 1
        current_mse = df['mse'].iloc[current_idx]
        
        st.write(f"**Frame Index:** {selected_frame}")
        st.write(f"**Momentum Value:** {current_mse:.2f}")
        
        # Status highlight
        if current_idx in drops:
            st.error("⚠️ Status: DETECTED DROP")
        elif current_idx in merges:
            st.warning("🧊 Status: DETECTED MERGE")
        else:
            st.success("✅ Status: NORMAL")

    with c_img:
        img = extract_frame(temp_path, current_idx)
        if img is not None:
            # Compatible with older versions
            st.image(img, caption=f"Visual at Frame {selected_frame}", use_column_width=True)

    # Clean up button
    if st.sidebar.button("Clear Cache"):
        if os.path.exists(temp_path):
            os.remove(temp_path)
            st.sidebar.success("Cache cleared!")