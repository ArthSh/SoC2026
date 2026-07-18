"""
AI-Powered Industrial Quality Assurance System
Detects steel surface defects with YOLOv8 and generates an
LLM-written inspection report using a local Ollama (Llama 3.2) model.
"""

import json
from datetime import datetime

import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = "models/best.pt"          # path to your trained YOLOv8 weights
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
CONF_THRESHOLD = 0.25

st.set_page_config(page_title="Steel Defect Inspector", layout="wide")


# --------------------------------------------------------------------------
# Cached model loader (loads once, not on every rerun)
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


# --------------------------------------------------------------------------
# Run YOLO inference and return (annotated_image, list_of_detections)
# --------------------------------------------------------------------------
def run_detection(model, image):
    results = model.predict(image, conf=CONF_THRESHOLD, verbose=False)
    result = results[0]

    annotated = result.plot()  # numpy array (BGR) with boxes drawn

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        detections.append({"class": cls_name, "confidence": round(conf * 100, 1)})

    return annotated, detections


# --------------------------------------------------------------------------
# Build the prompt and call the local Ollama model
# --------------------------------------------------------------------------
def generate_report(detections):
    if not detections:
        defect_lines = "No defects were detected."
    else:
        defect_lines = "\n".join(
            f"- {d['class']} ({d['confidence']}% confidence)" for d in detections
        )

    prompt = f"""You are a quality-control inspector at a steel manufacturing plant.
Based on the following YOLOv8 defect detections on a steel sheet surface, write a
professional inspection report.

Detected Defects:
{defect_lines}

Write the report with exactly these sections:
1. Summary (2-3 sentences describing what was found)
2. Severity (one word: Low, Medium, or High)
3. Recommended Action (3-4 concise bullet points)

Keep the tone professional and concise. Do not repeat the raw defect list back verbatim,
synthesize it into the summary."""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Could not connect to Ollama. Make sure it's running locally "
            "(`ollama serve`) and that the model is pulled (`ollama pull llama3.2`)."
        )
    except Exception as e:
        return f"⚠️ Error generating report: {e}"


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🔍 AI-Powered Steel Surface Inspection")
st.caption("Upload a steel surface image to detect defects and generate an AI inspection report.")

uploaded_file = st.file_uploader("Upload a steel surface image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Loading detection model..."):
        model = load_model()

    with st.spinner("Detecting defects..."):
        annotated_img, detections = run_detection(model, image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Detected Defects")
        st.image(annotated_img, channels="BGR", use_container_width=True)

    st.divider()
    st.subheader("📋 Detection Summary")
    if detections:
        for d in detections:
            st.write(f"• **{d['class']}** ({d['confidence']}%)")
    else:
        st.info("No defects detected above the confidence threshold.")

    st.divider()
    st.subheader("🧾 Inspection Report")

    if st.button("Generate AI Report", type="primary"):
        with st.spinner("Generating report with Llama 3.2..."):
            report_text = generate_report(detections)

        st.markdown(f"**Inspection Date:** {datetime.now().strftime('%d %B %Y')}")
        st.markdown(report_text)

        st.download_button(
            "Download Report",
            data=(
                f"Inspection Report\n"
                f"Inspection Date: {datetime.now().strftime('%d %B %Y')}\n\n"
                f"Detected Defects:\n"
                + "\n".join(f"- {d['class']} ({d['confidence']}%)" for d in detections)
                + f"\n\n{report_text}"
            ),
            file_name="inspection_report.txt",
        )
else:
    st.info("👆 Upload an image to get started.")
