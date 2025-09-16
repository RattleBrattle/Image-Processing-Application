# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import imageprocessing as ip  # Your module!

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="OpenCV Image Processor",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS --------------------
# Inject custom CSS for styling
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; margin-bottom: 1rem;}
    .section-header {font-size: 1.5rem; color: #ff7f0e; border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; margin-top: 2rem;}
    .success-msg {color: #2ecc71; font-weight: bold;}
    .error-msg {color: #e74c3c; font-weight: bold;}
    /* Add more custom styles here */
</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("🖼️ OpenCV Processor")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    # Theme toggle (Light/Dark mode)
    theme = st.selectbox("Theme", ["Light", "Dark"])
    
    # Navigation
    st.markdown("---")
    app_mode = st.selectbox("Choose Processing Type", 
                           ["Home", "Color Transformations", "Geometric Transformations", 
                            "Filtering & Smoothing", "Edge Detection", "Thresholding", 
                            "Histogram Operations", "Contours"])

# -------------------- THEME APPLY --------------------
if theme == "Dark":
    st.markdown("""
    <style>
        .stApp {background-color: #0e1117; color: #fafafa;}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp {background-color: #ffffff; color: #31333f;}
    </style>
    """, unsafe_allow_html=True)

# -------------------- IMAGE PROCESSING --------------------
def get_image():
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image
    return None

# -------------------- HOME PAGE --------------------
if app_mode == "Home":
    st.markdown('<p class="main-header">OpenCV Image Processor</p>', unsafe_allow_html=True)
    st.markdown("""
    Welcome to your complete image processing playground! This app showcases the Python module I built with **25+ computer vision functions**.
    
    **Features include:**
    - Color space conversions
    - Geometric transformations
    - Filtering & smoothing
    - Edge detection
    - Thresholding techniques
    - Histogram operations
    - Contour detection
    
    Upload an image and explore using the sidebar! 👈
    """)
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

# -------------------- COLOR TRANSFORMATIONS --------------------
elif app_mode == "Color Transformations":
    st.markdown('<p class="section-header">Color Space Transformations</p>', unsafe_allow_html=True)
    
    image = get_image()
    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
        
        with col2:
            option = st.selectbox("Select Transformation", 
                                 ["Grayscale", "HSV"])
            
            if option == "Grayscale":
                result = ip.convert_to_grayscale(image)
                st.image(result, caption="Grayscale Image", use_column_width=True, channels='GRAY')
            elif option == "HSV":
                result = ip.convert_to_hsv(image)
                # Convert HSV to RGB for display (HSV display is not straightforward)
                st.image(cv2.cvtColor(result, cv2.COLOR_HSV2RGB), caption="HSV Image", use_column_width=True)

# -------------------- GEOMETRIC TRANSFORMATIONS --------------------
elif app_mode == "Geometric Transformations":
    st.markdown('<p class="section-header">Geometric Transformations</p>', unsafe_allow_html=True)
    
    image = get_image()
    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
        
        with col2:
            option = st.selectbox("Select Transformation", 
                                 ["Flip", "Rotate", "Resize"])
            
            if option == "Flip":
                flip_code = st.selectbox("Flip Direction", 
                                        ["Horizontal (1)", "Vertical (0)", "Both (-1)"])
                code_map = {"Horizontal (1)": 1, "Vertical (0)": 0, "Both (-1)": -1}
                result = ip.flip_image(image, code_map[flip_code])
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Flipped Image", use_column_width=True)
                
            elif option == "Rotate":
                angle = st.slider("Rotation Angle", -180, 180, 45)
                result = ip.rotate_image(image, angle)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"Rotated {angle}°", use_column_width=True)
                
            elif option == "Resize":
                width = st.slider("Width", 50, 1000, image.shape[1])
                height = st.slider("Height", 50, 1000, image.shape[0])
                result = ip.resize_image(image, width=width, height=height)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Resized Image", use_column_width=True)

# -------------------- ADD MORE SECTIONS HERE --------------------
# [Add similar sections for Filtering, Edge Detection, Thresholding, etc.]
# The pattern is the same: get image, show original, provide parameters, show result

elif app_mode == "Filtering & Smoothing":
    st.markdown('<p class="section-header">Filtering & Smoothing</p>', unsafe_allow_html=True)
    # Implement similar to above sections

elif app_mode == "Edge Detection":
    st.markdown('<p class="section-header">Edge Detection</p>', unsafe_allow_html=True)
    # Implement similar to above sections

# ... [Continue for other sections]

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("### 🚀 Built with Python, OpenCV, and Streamlit")
st.markdown("*Image Processing Module by [Your Name]*")
