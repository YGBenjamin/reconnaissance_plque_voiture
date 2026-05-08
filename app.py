import streamlit as st
from ultralytics import YOLO
from easyocr import Reader
import time
import cv2
import os
from detect_and_recognize import detect_number_plates, recognize_number_plate

st.set_page_config(page_title="Détecteur de plaques d'immatriculation", page_icon=":car", layout="wide")

st.title("Détecteur automatique de plaques d'immatriculation - Yang Benjamin")
st.markdown("---")

uploaded_file = st.file_uploader("Importer une image", type=["png", "jpg", "jpeg"])
upload_path = "uploads"

if uploaded_file is not None:
    image_path = os.path.sep.join([upload_path, uploaded_file.name])
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("En cours de téléchargement..."):
        model = YOLO("runs/detect/train/weights/best.pt")
        reader = Reader(["en"])

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image_copy = image.copy()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Image Originale")
            st.image(image)

        number_plate_list = detect_number_plates(image, model)

        if number_plate_list != []:
            number_plate_list = recognize_number_plate(image_path, reader, number_plate_list, write_to_csv=True)

            for box, text in number_plate_list:
                cropped_number_plate = image_copy[box[1]:box[3], box[0]:box[2]]

                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(image, text, (box[0], box[3]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                with col2:
                    st.subheader("Détection de Plaque")
                    st.image(image)

                st.subheader("Plaque d'immatriculation")
                st.image(cropped_number_plate)
                st.success(f"Number plate text : {text}")


        else:
            st.error("Pas de plaque détectée :()")


else:
    st.info("Importer une image pour commencer !")