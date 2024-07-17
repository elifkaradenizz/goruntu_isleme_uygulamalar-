import streamlit as st
import numpy as np
from PIL import Image

def pixel():
    st.title("Görüntüyü Piksellere Ayırma ve Koordinatları Gösterme")
    uploaded_file = st.file_uploader("Bir görüntü yükleyin", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)

        st.image(image, caption='Yüklenen Görüntü', use_column_width=True)

        if st.button('Pikselleri Göster'):
            st.write("Görüntü Boyutları: ", image_array.shape)
            height, width, _ = image_array.shape

            pixels_info = []
            for y in range(height):
                for x in range(width):
                    color = image_array[y, x]
                    pixel_info = {
                        "Koordinatlar": f"({x}, {y})",
                        "Renk Değeri": f"{color}"
                    }
                    pixels_info.append(pixel_info)
            
            # Piksellerin bilgilerini tablo olarak göstermek
            st.dataframe(pixels_info)

if __name__ == "__main__":
    pixel()
