import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Class labels
class_names = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 
               'Class_6', 'Class_7', 'Class_8', 'Class_9', 'Class_10', 'Class_11']

# Title
st.title("🐟 Multiclass Fish Image Classifier")
st.markdown("Upload a fish image and select a model to classify the fish!")

# Sidebar - Model selection
model_name = st.sidebar.selectbox(
    "Choose a pre-trained model",
    ("vgg16_best.h5", "best_resnet_model.h5", "best_mobilenet_model.h5", 
     "best_inception_model.h5", "best_efficientnet_model.h5")
)

# ✅ Full model path
model_path = os.path.join(os.path.dirname(__file__), model_name)

# Image upload
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

# Prediction logic
def preprocess_image(img, target_size):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def predict(img, model):
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx]
    return class_names[class_idx], confidence

# Run if image is uploaded
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    img = Image.open(uploaded_file)

    with st.spinner("Loading model and predicting..."):
        # ✅ Load model first
        model = load_model(model_path)

        # ✅ Get required input shape from model
        input_shape = model.input_shape[1:3]  # (height, width)

        # ✅ Preprocess using model's expected input size
        preprocessed = preprocess_image(img, input_shape)

        # Predict
        label, confidence = predict(preprocessed, model)

    st.success(f"Prediction: **{label}** ({confidence*100:.2f}% confidence)")