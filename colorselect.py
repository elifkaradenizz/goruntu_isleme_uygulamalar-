import streamlit as st
import numpy as np
from PIL import Image

def color_selection():

    def rgb_to_grayscale(image_array):
        grayscale_image = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
        return grayscale_image.astype(np.uint8)


    def grayscale_to_binary(grayscale_image, threshold=128):
        binary_image = (grayscale_image > threshold) * 255
        return binary_image.astype(np.uint8)


    st.title("Görüntü Dönüştürme")
    uploaded_file = st.file_uploader("Bir görüntü yükleyin", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        image_array = np.array(image)

        st.image(image, caption='Yüklenen Görüntü', use_column_width=True)

        conversion_type = st.selectbox(
            "Dönüştürme Tipini Seçin",
            ("Orijinal", "Gri Seviye", "Binary")
        )

        if conversion_type == "Orijinal":
            st.image(image, caption='Orijinal Görüntü', use_column_width=True)

            # Orijinal görüntü için bit derinliği seçeneği ekleme
            bit_depth = st.select_slider("Bit Derinliği Seçin", options=[2, 4, 8])
            if bit_depth != 8:
                image_array = np.right_shift(image_array, 8 - bit_depth)
            st.image(image_array, caption=f"{bit_depth}-Bit Orijinal Görüntü", use_column_width=True)

        elif conversion_type == "Gri Seviye":
            grayscale_image = rgb_to_grayscale(image_array)
            st.image(grayscale_image, caption='Gri Seviye Görüntü', use_column_width=True, channels='GRAY')

            # Gri seviye görüntü için bit derinliği seçeneği ekleme
            bit_depth = st.select_slider("Bit Derinliği Seçin", options=[2, 4, 8])
            if bit_depth != 8:
                grayscale_image = np.right_shift(grayscale_image, 8 - bit_depth)
            st.image(grayscale_image, caption=f"{bit_depth}-Bit Gri Seviye Görüntü", use_column_width=True, channels='GRAY')

        elif conversion_type == "Binary":
            grayscale_image = rgb_to_grayscale(image_array)
            binary_image = grayscale_to_binary(grayscale_image)
            st.image(binary_image, caption='Binary Görüntü', use_column_width=True, channels='GRAY')

            # Binary görüntü için bit derinliği seçeneği ekleme
            bit_depth = st.select_slider("Bit Derinliği Seçin", options=[2, 4, 8])
            if bit_depth != 8:
                binary_image = np.right_shift(binary_image, 8 - bit_depth)
            st.image(binary_image, caption=f"{bit_depth}-Bit Binary Görüntü", use_column_width=True, channels='GRAY')