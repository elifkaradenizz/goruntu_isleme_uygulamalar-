import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour, flood
from colorselect import color_selection
from filter import filter
from histogram import histogram_equal
from pixel import pixel
from contour_operations import contour_operations
from morfoprocess import morphological_operations



# Session state ile aktif butonu takip etme
if 'active_button' not in st.session_state:
    st.session_state.active_button = 'Renk Seçimi'

# Butonların yerleştirilmesi ve stillendirilmesi için yardımcı fonksiyon
def create_button(button_name, display_name):
    style = f"""
    <style>
    .button {{
        background-color: blue !important;
        color: white;
        width: 100%;
        height: 50px;
        border-radius: 5px;
        margin-bottom: 5px;
    }}
    
    </style>
    """
    button_clicked = st.sidebar.button(display_name)
    if button_clicked:
        st.session_state.active_button = button_name
    return button_clicked

# Kenar çubuğunda butonların oluşturulması
st.sidebar.markdown("""
    <style>
    .center-text {
        text-align: center;
        font-weight: bold;
        font-size: 50px;
    }
    </style>
    <div class="center-text">Menü</div>
    """, unsafe_allow_html=True)

if create_button('Renk Seçimi', 'Renk Seçimi'):
    st.session_state.active_button = 'Renk Seçimi'
if create_button('Filtreleme', 'Filtreleme'):
    st.session_state.active_button = 'Filtreleme'
if create_button('Histogram', 'Histogram'):
    st.session_state.active_button = 'Histogram'
if create_button('Pixel', 'Pixel'):
    st.session_state.active_button = 'Pixel'
if create_button('Contour', 'Contour'):
    st.session_state.active_button = 'Contour'
if create_button('Morfolojik İşlem', 'Morfolojik İşlem'):
    st.session_state.active_button = 'Morfolojik İşlem'

# Aktif butona göre işlem yapma
if st.session_state.active_button == 'Renk Seçimi':
    color_selection()
elif st.session_state.active_button == 'Filtreleme': 
    filter() 
elif st.session_state.active_button == 'Histogram':
    histogram_equal()   
elif st.session_state.active_button == 'Pixel':
    pixel()
elif st.session_state.active_button == 'Contour':
   contour_operations()   
elif st.session_state.active_button == 'Morfolojik İşlem':
    morphological_operations()
    
# CSS ile butonların boyutlarını eşitleme
st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        height: 50px;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)