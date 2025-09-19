import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Maize Leaf Disease Classification (MDM)",
    page_icon=""
)

# -------------------------------------------------
# Download & load model
# -------------------------------------------------
MODEL_FILE_ID = "1tjCTytzJtUML4lxFtLm0ZlyTICtU43L6"
MODEL_PATH = "mdm_vgg_net16.h5"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

model = download_and_load_model()
CLASS_NAMES = ["Unhealthy", "Healthy"]

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("Maize Leaf Disease Classification Dashboard (MDM)")

# Two-column layout
col1, col2 = st.columns([1, 1])

# -------------------------------------------------
# Left Column : Upload & Prediction
# -------------------------------------------------
with col1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        confidence = np.max(predictions) * 100
        predicted_class = CLASS_NAMES[np.argmax(predictions)]

        if confidence >= 50:
            st.success(f"**Prediction:** {predicted_class}: Maize Downy Mildew Disease")
            st.info(f"**Confidence:** {confidence:.2f}%")
        else:
            st.success("**Prediction:** Healthy")
            st.info(f"**Confidence:** {100 - confidence:.2f}%")

# -------------------------------------------------
# Right Column : Fungicide Advisory
# -------------------------------------------------
with col2:
    st.subheader("Maize Downy Mildew – Fungicide Advisory")

    # Add CSS for nicer bullet spacing
    st.markdown(
        """
        <style>
        ul li { margin-bottom: 0.6em; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- General Guidelines ---
    st.markdown(
        """
        **General Guidelines**  
        * Downy mildew is seed- and soil-borne and spreads rapidly under **high humidity** and **20–25 °C** conditions.  
        * **Seed treatment is essential** for primary prevention.  
        * Foliar sprays are needed only as a follow-up under **high disease pressure**.  
        * **Integrated approach** (Seed Dressing + Foliar Spray) offers the best control.
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Severity-Based Spraying")

    # --- Markdown Table with Bold Text ---
    st.markdown(
        """
| Disease Pressure | Crop Stage | Spray / Treatment Recommendation |
|------------------|-----------|----------------------------------|
| **Preventive (No visible symptoms)** | Before sowing | **Seed treatment mandatory** → **Metalaxyl 4% + Mancozeb 64% WP @ 3 g/kg seed**. |
| **Low (≤10% PDI)** | Seedling stage (up to 25 DAS) | Seed treatment with **Metalaxyl 4% + Mancozeb 64% WP @ 3 g/kg seed** alone is effective. |
| **Moderate (10–20% PDI)** | Early vegetative stage (25–35 DAS) | Seed treatment (**Metalaxyl 4% + Mancozeb 64% WP @ 3 g/kg seed**) **+** foliar spray with **Carbendazim + Mancozeb @ 2 g/L water** or **Azoxystrobin + Cyproconazole @ 1 ml/L water**. |
| **High (>20% PDI)** | 30–45 DAS (humid/wet) | Seed treatment (**Metalaxyl 4% + Mancozeb 64% WP @ 3 g/kg seed**) **+** foliar spray with **Azoxystrobin + Difenoconazole @ 1 ml/L water** (most effective). **Repeat after 15–20 days** if disease persists. |
        """,
        unsafe_allow_html=True
    )

    # --- Advisory Highlights ---
    st.markdown(
        """
        **Advisory Highlights**  

        * **Seed Treatment:** Mandatory with **Metalaxyl + Mancozeb** for all maize sowings in endemic areas.  
        * **Best Strategy:** Seed treatment + **Azoxystrobin + Difenoconazole** foliar spray → *up to 97.6 % disease control*.  
        * **Moderate Strategy:** **Metalaxyl seed treatment alone** provides strong control under low disease pressure.  
        * **Avoid Sole Foliar Sprays:** Treatments like **Carbendazim + Mancozeb**, **Azoxystrobin + Difenoconazole**, or **Azoxystrobin + Cyproconazole** alone are less effective **without seed treatment**.  
        * **Spray Volume:** Maintain **500 L water/ha** for uniform coverage and penetration.  
        * **Resistance Management:** Avoid continuous use of **Metalaxyl** or strobilurins (Azoxystrobin, Pyraclostrobin).  
        * Rotate fungicides and mix different modes of action.
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <style>
    .developed-by {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .person {
        font-size: 16px;
        margin-bottom: 2px;
    }
    </style>
    <div class="developed-by">Developed by</div>
    <div class="person"><b>Anurag Dhole</b> - Researcher at MIT, Manipal</div>
    <div class="person"><b>Dr. Jadesha G</b> - Assistant Professor at GKVK, UAS, Bangalore</div>
    <div class="person"><b>Dr. Deepak D.</b> - Professor at MIT, Manipal</div>
    """,
    unsafe_allow_html=True
)


