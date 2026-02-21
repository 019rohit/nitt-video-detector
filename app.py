import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import tempfile
import os
import uuid

#configuration based on gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Frame Drop and Merge Detector",
    layout="wide",
    page_icon="🏏"
)

#analysis
@st.cache_data(show_spinner=False)
def analyze_momentum_final(file_path, _run_id):
    cap = cv2.VideoCapture(file_path)
    mse_list = []
    prev_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(
            cv2.resize(frame, (160, 90)),
            cv2.COLOR_BGR2GRAY
        )

        if prev_gray is not None:
            if DEVICE.type == "cuda":
                t1 = torch.from_numpy(prev_gray).to(DEVICE).float()
                t2 = torch.from_numpy(gray).to(DEVICE).float()
                mse = torch.mean((t1 - t2) ** 2).item()
            else:
                mse = np.mean(
                    (gray.astype(float) - prev_gray.astype(float)) ** 2
                )
            mse_list.append(mse)
        else:
            mse_list.append(0.0)

        prev_gray = gray

    cap.release()
    return mse_list


#extracting frames 
def extract_frame(video_path, frame_no):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None



st.title("Video Temporal Error Detection)")

uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4"])

if uploaded_file:

    temp_path = os.path.join(
        tempfile.gettempdir(),
        f"nitt_full_{uploaded_file.name}"
    )

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    #total fps and frames
    cap_info = cv2.VideoCapture(temp_path)
    fps = cap_info.get(cv2.CAP_PROP_FPS)
    total_frames_actual = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_info.release()

    run_id = str(uuid.uuid4())
    data = analyze_momentum_final(temp_path, run_id)
    df = pd.DataFrame({"mse": data})

    #Detection Logic
    median_motion = df["mse"].median()
    std_dev = df["mse"].std()
    drop_multiplier = 2.5 if std_dev > 15 else 1.8

    df["prev"] = df["mse"].shift(1).fillna(0)
    df["next"] = df["mse"].shift(-1).fillna(0)

    is_drop = (
        (df["mse"] > median_motion * drop_multiplier)
        & (df["mse"] > df["next"])
        & (df["mse"] > 20)
    )
    drop_indices = set(df[is_drop].index.tolist())

    is_merge = (
        (df["mse"] < median_motion * 0.15)
        & (df["prev"] > 5.0)
    )
    merge_indices = set(df[is_merge].index.tolist())

    total_frames = len(data)

    #slider and dynamic counter
    selected_frame = st.slider(
        "Frame Timeline",
        1,
        total_frames,
        1
    )
    current_idx = selected_frame - 1
    dynamic_drops = len([d for d in drop_indices if d <= current_idx])
    dynamic_merges = len([m for m in merge_indices if m <= current_idx])

    #Layout 
    col_graph, col_video = st.columns([1.4, 1])

    #graph
    with col_graph:

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=data[:selected_frame],
            name="Momentum",
            line=dict(color="#00ff88", width=2)
        ))

        current_drops = [d for d in drop_indices if d < selected_frame]
        current_merges = [m for m in merge_indices if m < selected_frame]

        if current_drops:
            fig.add_trace(go.Scatter(
                x=current_drops,
                y=df["mse"].iloc[current_drops],
                mode="markers",
                name="Drop",
                marker=dict(color="red", size=8, symbol="star")
            ))

        if current_merges:
            fig.add_trace(go.Scatter(
                x=current_merges,
                y=df["mse"].iloc[current_merges],
                mode="markers",
                name="Merge",
                marker=dict(color="yellow", size=6)
            ))

        fig.add_vline(
            x=current_idx,
            line_width=1,
            line_dash="dash",
            line_color="white"
        )

        fig.update_layout(
            template="plotly_dark",
            height=520,
            margin=dict(l=10, r=10, t=10, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

    # ================= FRAME PLAYBACK =================
    with col_video:

        st.markdown(f"""
        <div style="font-size:13px; line-height:1.5">
        🎞 <b>Total Frames:</b> {total_frames_actual} &nbsp;&nbsp; |
        🎬 <b>FPS:</b> {fps:.2f} <br>
        ⚠️ <b>Drops So Far:</b> {dynamic_drops} &nbsp;&nbsp; |
        🧊 <b>Merges So Far:</b> {dynamic_merges} <br>
        📍 <b>Current Frame:</b> {selected_frame} <br>
        📈 <b>MSE:</b> {df["mse"].iloc[current_idx]:.2f}
        </div>
        """, unsafe_allow_html=True)

        if current_idx in drop_indices:
            st.error("⚠️ DROP DETECTED")
        elif current_idx in merge_indices:
            st.warning("🧊 MERGE DETECTED")
        else:
            st.success("✅ NORMAL FRAME")

        frame_img = extract_frame(temp_path, current_idx)

        if frame_img is not None:
            st.image(frame_img, use_column_width=True)
    # -------- Clear Cache --------
    if st.sidebar.button("Clear Cache"):
        if os.path.exists(temp_path):
            os.remove(temp_path)
            st.sidebar.success("Cache cleared!")