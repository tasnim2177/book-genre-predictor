# app.py
import streamlit as st
from PIL import Image
import io
import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from efficientnet_pytorch import EfficientNet

# ---------------- USER CONFIG ----------------
MODEL_PATH = "efficientnet_b4_mapped.pth"
# release URL you provided:
RELEASE_URL = "https://github.com/tasnim2177/book-genre-predictor/releases/download/v1.0/efficientnet_b4_mapped.pth"
MODEL_NAME = "efficientnet-b4"
IMG_SIZE = 224
NUM_CLASSES = 32
LABELS_PATH = "labels.txt"
TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------

# ---------- ensure model is present (download from GitHub Releases if needed) ----------
def download_with_progress(url, dst):
    if os.path.exists(dst):
        return
    # stream-download
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(dst, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

if not os.path.exists(MODEL_PATH):
    download_with_progress(RELEASE_URL, MODEL_PATH)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Book Cover → Genre (EfficientNet-B4)", layout="centered")
st.title("📚 Book Cover → Genre Predictor")

# ---------- helper functions ----------
@st.cache_resource(show_spinner=False)
def load_labels(path=LABELS_PATH, n=NUM_CLASSES):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        if len(labels) < n:
            labels += [f"class_{i}" for i in range(len(labels), n)]
        elif len(labels) > n:
            labels = labels[:n]
        return labels
    else:
        return [f"class_{i}" for i in range(n)]

@st.cache_resource(show_spinner=True)
def build_and_load_model(model_name=MODEL_NAME, model_path=MODEL_PATH, num_classes=NUM_CLASSES, device=DEVICE):
    model = EfficientNet.from_name(model_name)
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)       # strict=True
    model.to(device)
    model.eval()
    return model

def get_transforms(img_size=IMG_SIZE):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

def predict(model, pil_img, transforms, topk=TOP_K, device=DEVICE):
    x = transforms(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        top_probs, top_idxs = torch.topk(probs, k=topk, dim=1)
    return top_probs.cpu().numpy().flatten().tolist(), top_idxs.cpu().numpy().flatten().tolist()

# ---------- load model & labels ----------
try:
    with st.spinner("Loading model..."):
        model = build_and_load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

labels = load_labels()
transforms = get_transforms()

# ---------- UI ----------
uploaded = st.file_uploader("Upload book cover (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    except Exception:
        st.error("Could not read the uploaded image.")
        st.stop()

    st.image(img, caption="Uploaded Image", width=360)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            probs, idxs = predict(model, img, transforms)

        st.success("Prediction Complete")
        st.write("### Top Predictions")
        for p, idx in zip(probs, idxs):
            label = labels[idx] if idx < len(labels) else f"class_{idx}"
            st.write(f"- **{label}** — {p*100:.2f}%")

else:
    st.info("Upload an image to continue.")
