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
# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    /* Base styles for both themes */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 1.2rem;
    }
    
    /* Light theme with green accents */
    [data-theme="light"],
    [data-theme="light"] .stApp {
        --primary: #ffffff;
        --secondary: #f0f0f0;
        --text-primary: #31333f;
        --text-secondary: #595b6b;
        --accent: #10b981;
        --accent-light: #a7f3d0;
        --border: #e5e7eb;
    }
    
    /* Dark theme with blue accents */
    [data-theme="dark"],
    [data-theme="dark"] .stApp {
        --primary: #0e1117;
        --secondary: #1e2229;
        --text-primary: #fafafa;
        --text-secondary: #d1d5db;
        --accent: #3b82f6;
        --accent-light: #93c5fd;
        --border: #374151;
    }
    
    /* Apply theme variables */
    .stApp {
        background-color: var(--primary);
        color: var(--text-primary);
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1y4p8pa {
        background-color: var(--secondary) !important;
        border-right: 1px solid var(--border);
    }
    
    .stSidebar .stSelectbox, .stSidebar .stMarkdown, .stSidebar .stTitle {
        color: var(--text-primary) !important;
    }
    
    /* Select boxes and inputs */
    .stSelectbox > div > div {
        background-color: var(--secondary);
        border: 1px solid var(--border);
        color: var(--text-primary);
    }
    
    .stSelectbox label {
        color: var(--text-primary) !important;
    }
    
    /* Sliders */
    .stSlider > div > div {
        color: var(--accent) !important;
    }
    
    /* Buttons and interactive elements */
    .stButton > button {
        background-color: var(--accent);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: var(--accent-light);
        color: var(--text-primary);
    }
    
    /* Divider line */
    .stMarkdown > hr {
        border-color: var(--border);
        margin: 2rem 0;
    }
    
    /* Footer styling */
    .stMarkdown:last-child {
        color: var(--text-secondary);
    }
    
    /* Adjust image borders to match theme */
    .stImage > img {
        border: 1px solid var(--border);
        border-radius: 4px;
    }
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
    <script>
        document.body.setAttribute('data-theme', 'dark');
    </script>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <script>
        document.body.setAttribute('data-theme', 'light');
    </script>
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
        st.image(uploaded_file, caption="Uploaded Image", width='stretch')

# -------------------- COLOR TRANSFORMATIONS --------------------
elif app_mode == "Color Transformations":
    st.markdown('<p class="section-header">Color Space Transformations</p>', unsafe_allow_html=True)
    
    image = get_image()
    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", width='stretch')
        
        with col2:
            option = st.selectbox("Select Transformation", 
                                 ["Grayscale", "HSV"])
            
            if option == "Grayscale":
                result = ip.convert_to_grayscale(image)
                st.image(result, caption="Grayscale Image", width='stretch', channels='GRAY')
            elif option == "HSV":
                result = ip.convert_to_hsv(image)
                # Convert HSV to RGB for display (HSV display is not straightforward)
                st.image(cv2.cvtColor(result, cv2.COLOR_HSV2RGB), caption="HSV Image", width='stretch')

# -------------------- GEOMETRIC TRANSFORMATIONS --------------------
elif app_mode == "Geometric Transformations":
    st.markdown('<p class="section-header">Geometric Transformations</p>', unsafe_allow_html=True)
    
    image = get_image()
    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", width='stretch')
        
        with col2:
            option = st.selectbox("Select Transformation", 
                                 ["Flip", "Rotate", "Resize"])
            
            if option == "Flip":
                flip_code = st.selectbox("Flip Direction", 
                                        ["Horizontal (1)", "Vertical (0)", "Both (-1)"])
                code_map = {"Horizontal (1)": 1, "Vertical (0)": 0, "Both (-1)": -1}
                result = ip.flip_image(image, code_map[flip_code])
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Flipped Image", width='stretch')
                
            elif option == "Rotate":
                angle = st.slider("Rotation Angle", -180, 180, 45)
                result = ip.rotate_image(image, angle)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"Rotated {angle}°", width='stretch')
                
            elif option == "Resize":
                width = st.slider("Width", 50, 1000, image.shape[1])
                height = st.slider("Height", 50, 1000, image.shape[0])
                result = ip.resize_image(image, width=width, height=height)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Resized Image", width='stretch')

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
