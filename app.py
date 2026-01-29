import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="AI Digit Recognizer",
    page_icon="🧠",
    layout="centered"
)

# Load model
@st.cache_resource
def load_mlp_model():
    return load_model("mnist_mlp_model.keras")

model = load_mlp_model()

# Title UI
st.markdown("<h1 style='text-align: center;'>🧠 AI Handwritten Digit Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Deep Learning | Multi-Layer Perceptron | MNIST</p>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar info panel
with st.sidebar:
    st.header("📌 Project Info")
    st.write("**Model:** Multi-Layer Perceptron (MLP)")
    st.write("**Dataset:** MNIST")
    st.write("**Accuracy:** ~98%")
    st.write("**Framework:** TensorFlow + Streamlit")
    st.markdown("---")
    st.write("👨‍💻 Developed by: **Your Name**")
    st.write("📌 GitHub: https://github.com/yourusername")

# Tabs
tab1, tab2 = st.tabs(["✏ Draw Digit", "📤 Upload Image"])

# ---------- DRAW TAB ----------
with tab1:
    st.subheader("Draw a Digit (0–9)")

    canvas = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("🔍 Predict Digit"):
        if canvas.image_data is not None:
            img = Image.fromarray(canvas.image_data.astype("uint8"))
            img = img.convert("L")
            img = img.resize((28, 28))

            img_array = np.array(img)
            img_array = img_array / 255.0
            img_array = img_array.reshape(1, 28, 28)

            prediction = model.predict(img_array)[0]
            predicted_digit = np.argmax(prediction)

            st.success(f"🎯 Predicted Digit: {predicted_digit}")

            # Probability visualization
            fig, ax = plt.subplots()
            ax.bar(range(10), prediction)
            ax.set_xticks(range(10))
            ax.set_xlabel("Digit")
            ax.set_ylabel("Confidence")
            ax.set_title("Prediction Confidence")
            st.pyplot(fig)

# ---------- UPLOAD TAB ----------
with tab2:
    st.subheader("Upload an Image")

    uploaded_file = st.file_uploader("Upload 28x28 or normal image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=200)

        img = image.convert("L")
        img = img.resize((28, 28))

        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28)

        prediction = model.predict(img_array)[0]
        predicted_digit = np.argmax(prediction)

        st.success(f"🎯 Predicted Digit: {predicted_digit}")

        fig, ax = plt.subplots()
        ax.bar(range(10), prediction)
        ax.set_xticks(range(10))
        ax.set_xlabel("Digit")
        ax.set_ylabel("Confidence")
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ using Deep Learning & Streamlit</p>", unsafe_allow_html=True)
