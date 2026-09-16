import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Urban Intelligence AI",
    page_icon="🏙️",
    layout="wide"
)

# -----------------------------
# CUSTOM STYLE
# -----------------------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 18px;
    opacity: 0.75;
    margin-bottom: 25px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 15px;
}

.metric {
    font-size: 30px;
    font-weight: 700;
}

.small {
    opacity: 0.7;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    '<div class="title">🏙️ Urban Intelligence AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered urban scene analysis and intelligent transportation monitoring'
    '</div>',
    unsafe_allow_html=True
)

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("❌ best.pt was not found. Make sure it is in the same folder as app.py.")
    st.stop()

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("❌ Could not load the AI model.")
    st.code(str(e))
    st.stop()


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.40,
    step=0.05
)

st.sidebar.info(
    "Upload an urban/transportation image and the trained AI model "
    "will identify objects in the scene."
)


# -----------------------------
# UPLOAD
# -----------------------------
st.subheader("📷 Upload Urban Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Two columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original Image")
        st.image(image, use_container_width=True)

    # -----------------------------
    # RUN AI
    # -----------------------------
    with st.spinner("🤖 AI is analyzing the urban scene..."):

        results = model.predict(
            source=image,
            conf=confidence,
            verbose=False
        )

    result = results[0]

    # Annotated image
    annotated = result.plot()

    with col2:
        st.markdown("### AI Detection")
        st.image(
            annotated,
            channels="BGR",
            use_container_width=True
        )

    # -----------------------------
    # RESULTS
    # -----------------------------
    st.divider()

    st.subheader("📊 Detection Results")

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:

        st.warning(
            "No objects were detected above the selected confidence threshold."
        )

    else:

        detections = []

        for box in boxes:
            class_id = int(box.cls[0])
            confidence_score = float(box.conf[0])

            class_name = model.names[class_id]

            detections.append({
                "Object": class_name,
                "Confidence": f"{confidence_score * 100:.1f}%"
            })

        # Metrics
        total = len(detections)
        unique_objects = len(set(d["Object"] for d in detections))

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown("### Objects Detected")
            st.markdown(
                f'<div class="metric">{total}</div>',
                unsafe_allow_html=True
            )

        with m2:
            st.markdown("### Object Classes")
            st.markdown(
                f'<div class="metric">{unique_objects}</div>',
                unsafe_allow_html=True
            )

        with m3:
            avg_conf = sum(
                float(d["Confidence"].replace("%", ""))
                for d in detections
            ) / total

            st.markdown("### Average Confidence")
            st.markdown(
                f'<div class="metric">{avg_conf:.1f}%</div>',
                unsafe_allow_html=True
            )

        st.divider()

        st.markdown("### Detected Objects")

        for i, detection in enumerate(detections, start=1):

            st.write(
                f"**{i}. {detection['Object']}** "
                f"— {detection['Confidence']}"
            )

        # -----------------------------
        # DOWNLOAD RESULT
        # -----------------------------
        import cv2

        success, encoded_image = cv2.imencode(
            ".jpg",
            annotated
        )

        if success:

            st.download_button(
                label="⬇️ Download AI Detection Result",
                data=encoded_image.tobytes(),
                file_name="urban_ai_detection.jpg",
                mime="image/jpeg"
            )

else:

    st.info(
        "👆 Upload an image above to start AI-powered urban analysis."
    )

    st.markdown("""
    ### 🚦 How it works

    1. **Upload** an urban/transportation image
    2. **AI analyzes** the image using the trained YOLO model
    3. **Objects are detected** with bounding boxes
    4. **Confidence scores** are calculated
    5. **Results can be downloaded** for further analysis
    """)