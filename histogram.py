import streamlit as st
import numpy as np
from PIL import Image

def histogram_equal():
    st.header("Histogram Bölümü")

    def calculate_histogram(image):
        if image.mode != "RGB":
            image = image.convert("RGB")  # Resmi RGB moduna çevir

        width, height = image.size
        histogram_gray = np.zeros(256)

        for y in range(height):
            for x in range(width):
                if image.mode == "RGB":
                    r, g, b = image.getpixel((x, y))
                    # Grayscale dönüşümü yaparak pikseli gri seviyesine çevir
                    pixel = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
                    histogram_gray[pixel] += 1

        # Renkli (RGB) histogram için
        histogram_rgb = np.zeros((3, 256))

        for y in range(height):
            for x in range(width):
                r, g, b = image.getpixel((x, y))
                histogram_rgb[0][r] += 1
                histogram_rgb[1][g] += 1
                histogram_rgb[2][b] += 1

        return histogram_gray, histogram_rgb

    def equalize_histogram(image, alpha):
        if image.mode != "L":
            image = image.convert("L")  # Resmi gri tonlamaya çevir

        width, height = image.size
        pixels = list(image.getdata())
        hist = np.zeros(256)

        for pixel in pixels:
            hist[pixel] += 1

        # Histogram eşitleme
        cumsum = np.cumsum(hist)
        norm_cumsum = (cumsum - cumsum.min()) * 255 / (cumsum.max() - cumsum.min())

        equalized_pixels = []
        for pixel in pixels:
            new_pixel = int(alpha * norm_cumsum[pixel] + (1 - alpha) * pixel)
            equalized_pixels.append(new_pixel)

        equalized_image = Image.new("L", (width, height))
        equalized_image.putdata(equalized_pixels)

        return equalized_image

    st.title('Histogram Eşitleme')

    uploaded_file = st.file_uploader("Bir görüntü dosyası seçin", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Orijinal Resim", use_column_width=True)

        if st.button("Histogramları Oluştur"):
            hist_gray, hist_rgb = calculate_histogram(original_image)

            st.write("Gri Tonlama Histogramı:")
            st.bar_chart(hist_gray)

            st.write("Renkli (RGB) Histogramı:")
            for i, color in enumerate(['red', 'green', 'blue']):
                st.line_chart(hist_rgb[i])

        alpha_value = st.text_input("Eşitleme için bir değer girin (0 ile 1 arasında):", "0.5")
        try:
            alpha = float(alpha_value)
            if 0.0 <= alpha <= 1.0:
                if st.button("Histogram Eşitle"):
                    equalized_image = equalize_histogram(original_image, alpha)
                    st.image(equalized_image, caption="Eşitlenmiş Resim", use_column_width=True)
            else:
                st.warning("Lütfen geçerli bir değer girin (0 ile 1 arasında).")
        except ValueError:
            st.warning("Lütfen geçerli bir sayısal değer girin.")

if __name__ == '__main__':
    histogram_equal()
