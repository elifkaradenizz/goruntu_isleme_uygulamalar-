import streamlit as st
import numpy as np
from PIL import Image
from skimage.segmentation import active_contour, flood

def region_growing(image, seed_point, threshold):
    filled = flood(image, seed_point, tolerance=threshold)
    result_image = np.zeros_like(image)
    result_image[filled] = 255
    return result_image

def active_contour_segmentation(image, init_contour, alpha=0.015, beta=10, gamma=0.001):
    snake = active_contour(image, init_contour, alpha=alpha, beta=beta, gamma=gamma)
    return snake

def contour_operations():
    st.title('Görüntü İşleme Uygulaması')

    uploaded_file = st.file_uploader("Bir görüntü dosyası seçin", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)

        st.subheader('Orijinal Görüntü')
        st.image(image, caption='Orijinal Görüntü', use_column_width=True)

        operation = st.selectbox('İşlem Seçin:', ('Region Growing', 'Active Contour'))

        if operation == 'Region Growing':
            x = st.slider('Başlangıç Noktası X:', 0, image.shape[1] - 1, image.shape[1] // 2)
            y = st.slider('Başlangıç Noktası Y:', 0, image.shape[0] - 1, image.shape[0] // 2)
            threshold = st.slider('Eşik Değeri:', 0, 255, 50)
            seed_point = (y, x)

            if st.button('Region Growing Uygula'):
                result_image = np.zeros_like(image)
                for i in range(3):  # Process each channel independently
                    channel = image[:,:,i]
                    result_image[:,:,i] = region_growing(channel, seed_point, threshold)
                st.subheader('Region Growing Uygulanmış Görüntü')
                st.image(result_image, caption='Region Growing Uygulanmış Görüntü', use_column_width=True)

        elif operation == 'Active Contour':
            alpha = st.slider('Alpha Değeri:', 0.001, 0.1, 0.015)
            beta = st.slider('Beta Değeri:', 0.1, 30.0, 10.0)
            gamma = st.slider('Gamma Değeri:', 0.001, 1.0, 0.001)
            x = st.slider('Başlangıç Noktası X:', 0, image.shape[1] - 1, image.shape[1] // 2)
            y = st.slider('Başlangıç Noktası Y:', 0, image.shape[0] - 1, image.shape[0] // 2)
            radius = st.slider('Başlangıç Çember Yarıçapı:', 5, 100, 30)

            s = np.linspace(0, 2 * np.pi, 400)
            init_contour = np.array([y + radius * np.sin(s), x + radius * np.cos(s)]).T

            if st.button('Active Contour Uygula'):
                gray_image = np.mean(image, axis=2).astype(np.uint8)
                snake = active_contour_segmentation(gray_image, init_contour, alpha=alpha, beta=beta, gamma=gamma)
                result_image = np.copy(image)
                for coord in snake:
                    result_image[int(coord[0]), int(coord[1])] = [255, 0, 0]  # Highlight the contour in red
                st.subheader('Active Contour Uygulanmış Görüntü')
                st.image(result_image, caption='Active Contour Uygulanmış Görüntü', use_column_width=True)

if __name__ == '__main__':
    contour_operations()

