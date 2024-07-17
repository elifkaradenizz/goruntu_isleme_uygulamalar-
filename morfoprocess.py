import streamlit as st
import numpy as np
from PIL import Image

def morphological_operations():
    def erode(image, kernel):
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        result = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i + k_h, j:j + k_w]
                result[i, j] = np.min(region[kernel == 1])
                
        return result

    def dilate(image, kernel):
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        result = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i + k_h, j:j + k_w]
                result[i, j] = np.max(region[kernel == 1])
                
        return result

    def opening(image, kernel):
        eroded_image = erode(image, kernel)
        opened_image = dilate(eroded_image, kernel)
        return opened_image

    def closing(image, kernel):
        dilated_image = dilate(image, kernel)
        closed_image = erode(dilated_image, kernel)
        return closed_image

    def gradient(image, kernel):
        dilated_image = dilate(image, kernel)
        eroded_image = erode(image, kernel)
        gradient_image = dilated_image - eroded_image
        return gradient_image

    def top_hat(image, kernel):
        opened_image = opening(image, kernel)
        top_hat_image = image - opened_image
        return top_hat_image

    def black_hat(image, kernel):
        closed_image = closing(image, kernel)
        black_hat_image = closed_image - image
        return black_hat_image

    st.title('Morfolojik İşlemler Uygulaması')

    uploaded_file = st.file_uploader("Bir görüntü dosyası seçin", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        image = np.array(image)

        st.subheader('Orijinal Görüntü')
        st.image(image, caption='Orijinal Görüntü', use_column_width=True, clamp=True, channels='GRAY')

        kernel_size = st.slider('Kernel Boyutu:', 3, 15, 3, step=2)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        operation = st.selectbox('Morfolojik İşlem Seçin:', ('Erozyon', 'Genişleme', 'Açma', 'Kapama', 'Gradyan', 'Top-hat', 'Black-hat', 'Yayma', 'Aşındırma'))

        if st.button('İşlemi Uygula'):
            if operation == 'Erozyon':
                result_image = erode(image, kernel)
            elif operation == 'Genişleme':
                result_image = dilate(image, kernel)
            elif operation == 'Açma':
                result_image = opening(image, kernel)
            elif operation == 'Kapama':
                result_image = closing(image, kernel)
            elif operation == 'Gradyan':
                result_image = gradient(image, kernel)
            elif operation == 'Top-hat':
                result_image = top_hat(image, kernel)
            elif operation == 'Black-hat':
                result_image = black_hat(image, kernel)
            elif operation == 'Yayma':
                result_image = dilate(image, kernel)
            elif operation == 'Aşındırma':
                result_image = erode(image, kernel)

            st.subheader(f'{operation} İşlemi Uygulanmış Görüntü')
            st.image(result_image, caption=f'{operation} İşlemi Uygulanmış Görüntü', use_column_width=True, clamp=True, channels='GRAY')

if __name__ == '__main__':
    morphological_operations()
