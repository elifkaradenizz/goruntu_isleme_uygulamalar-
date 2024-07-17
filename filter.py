import streamlit as st
import cv2
import numpy as np
from PIL import Image

def apply_2d_filter(image_channel, kernel):
    """2D filtreleme işlemi uygular."""
    height, width = image_channel.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Görüntüyü kenarlarından belirtilen miktarda doldurarak genişletme yap
    padded_image = np.pad(image_channel, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Filtreleme işlemi için çıktı matrisini oluştur
    output_image = np.zeros_like(image_channel, dtype=np.float64)

    # Filtreleme işlemi (2D konvolüsyon)
    for i in range(height):
        for j in range(width):
            output_image[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output_image

def apply_blur(image_array):
    """Resmi bulanıklaştırma işlemi uygular."""
    kernel = np.ones((5, 5)) / 25  # 5x5 boyutunda ortalama bir bulanıklık çekirdeği
    blurred_image = np.zeros_like(image_array, dtype=np.float64)  # Float64 tipinde bir boş görüntü oluştur

    # Her bir kanal için ayrı ayrı filtreleme yap
    for c in range(image_array.shape[-1]):  # Her kanal için (RGB)
        blurred_image[:, :, c] = apply_2d_filter(image_array[:, :, c], kernel)

    return np.clip(blurred_image, 0, 255).astype(np.uint8)

def apply_sharpen(image_array):
    """Resmi keskinleştirme (sharpening) işlemi uygular."""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])  # Keskinleştirme çekirdeği
    sharpened_image = np.zeros_like(image_array, dtype=np.float64)  # Float64 tipinde bir boş görüntü oluştur

    # Her bir kanal için ayrı ayrı filtreleme yap
    for c in range(image_array.shape[-1]):  # Her kanal için (RGB)
        sharpened_image[:, :, c] = apply_2d_filter(image_array[:, :, c], kernel)

    return np.clip(sharpened_image, 0, 255).astype(np.uint8)

def apply_edge_detection(image_array):
    """Resmi kenar tespiti (edge detection) işlemi uygular."""
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])  # Kenar tespiti çekirdeği
    edge_image = np.zeros_like(image_array, dtype=np.float64)  # Float64 tipinde bir boş görüntü oluştur

    # Her bir kanal için ayrı ayrı filtreleme yap
    for c in range(image_array.shape[-1]):  # Her kanal için (RGB)
        edge_image[:, :, c] = apply_2d_filter(image_array[:, :, c], kernel)

    return np.clip(edge_image, 0, 255).astype(np.uint8)

def apply_sobel_filter(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    rows, cols = gray.shape
    sobelx = np.zeros((rows, cols))
    sobely = np.zeros((rows, cols))

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = gray[i-1:i+2, j-1:j+2]
            sx = np.sum(Kx * region)
            sy = np.sum(Ky * region)
            sobelx[i, j] = sx
            sobely[i, j] = sy

    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = (sobel / np.max(sobel) * 255).astype(np.uint8)

    # Eşikleme işlemi
    _, thresholded = cv2.threshold(sobel, threshold, 255, cv2.THRESH_BINARY)
    return thresholded

def apply_gaussian_filter(image, kernel_size=5, sigma=1.0, threshold=100):
    def gaussian_kernel(size, sigma):
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        return kernel / np.sum(kernel)

    kernel = gaussian_kernel(kernel_size, sigma)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gauss_filtered = cv2.filter2D(gray, -1, kernel)

    # Eşikleme işlemi
    _, thresholded = cv2.threshold(gauss_filtered, threshold, 255, cv2.THRESH_BINARY)
    return thresholded

def filter():
    st.title('Filtreleme ve Sobel-Gauss Birleştirme')

    uploaded_file = st.file_uploader("Bir görüntü dosyası seçin", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_array = np.array(image)

        st.subheader('Orijinal Görüntü')
        st.image(image_array, caption='Orijinal Görüntü', use_column_width=True)

        operation = st.selectbox('Filtreleme İşlemi Seçin:', ('Bulanıklaştırma', 'Keskinleştirme', 'Kenar Tespiti'))

        if st.button('İşlemi Uygula'):
            if operation == 'Bulanıklaştırma':
                result_image = apply_blur(image_array)
                st.subheader('Bulanıklaştırılmış Görüntü')
                st.image(result_image, caption='Bulanıklaştırılmış Görüntü', use_column_width=True, clamp=True)
            elif operation == 'Keskinleştirme':
                result_image = apply_sharpen(image_array)
                st.subheader('Keskinleştirilmiş Görüntü')
                st.image(result_image, caption='Keskinleştirilmiş Görüntü', use_column_width=True, clamp=True)
            elif operation == 'Kenar Tespiti':
                result_image = apply_edge_detection(image_array)
                st.subheader('Kenar Tespiti Yapılmış Görüntü')
                st.image(result_image, caption='Kenar Tespiti Yapılmış Görüntü', use_column_width=True, clamp=True)

        st.header("Sobel ve Gauss Filtresi Ayarları")

        sobel_threshold = st.slider("Sobel Eşik Değeri", 0, 255, 100)
        if st.button('Sobel Filtresi Uygula'):
            sobel_image = apply_sobel_filter(image_array, sobel_threshold)
            st.image(sobel_image, caption='Sobel Filtresi Uygulanmış Fotoğraf', use_column_width=True)

        gauss_threshold = st.slider("Gauss Eşik Değeri", 0, 255, 100)
        kernel_size = st.number_input("Gauss Kernel Boyutu", 3, 15, step=2, value=5)
        sigma = st.number_input("Gauss Sigma Değeri", 0.1, 10.0, value=1.0)
        if st.button('Gauss Filtresi Uygula'):
            gauss_image = apply_gaussian_filter(image_array, kernel_size, sigma, gauss_threshold)
            st.image(gauss_image, caption='Gauss Filtresi Uygulanmış Fotoğraf', use_column_width=True)

if __name__ == '__main__':
    filter()
