import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.title("Robótica Móvel - Comparação de Modelos")

model_base = YOLO("models/yolo11s.pt")
model_custom = YOLO("models/my_model.pt")

uploaded_file = st.file_uploader("Envie uma imagem para análise", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.button("Comparar Modelos"):
        file_path = os.path.join("temp_uploaded_image.jpg")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        results_base = model_base.predict(source=file_path, conf=0.25, save=True, classes=[0])
        results_custom = model_custom.predict(source=file_path, conf=0.25, save=True)

        save_dir_base = results_base[0].save_dir
        saved_img_path_base = os.path.join(save_dir_base, os.path.basename(file_path))

        save_dir_custom = results_custom[0].save_dir
        saved_img_path_custom = os.path.join(save_dir_custom, os.path.basename(file_path))

        img_base = Image.open(saved_img_path_base)
        img_custom = Image.open(saved_img_path_custom)

        col1, col2 = st.columns(2)

        with col1:
            st.header("Modelo Base")
            st.image(img_base, use_container_width=True)

        with col2:
            st.header("Meu Modelo")
            st.image(img_custom, use_container_width=True)
else:
    st.warning("Por favor, envie uma imagem no formato JPG, JPEG ou PNG para continuar.")
