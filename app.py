# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import imageprocessing as ip

# -------------------- PAGE SETUP --------------------
# Set initial theme based on session state or default to light
if 'theme' not in st.session_state:
    st.session_state.theme = "Light"

st.set_page_config(
    page_title="OpenCV Image Processor",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS --------------------
# Apply CSS based on selected theme
if st.session_state.theme == "Dark":
    css_theme = "dark"
else:
    css_theme = "light"

st.markdown(f"""
<style>
    /* ===== LIGHT MODE (WHITE & GREEN) ===== */
    .stApp:not([data-theme="dark"]) {{
        background: linear-gradient(135deg, #ffffff 0%, #f8fff8 100%);
        color: #2d3748 !important;
    }}
    
    /* Headers - Light Mode */
    .stApp:not([data-theme="dark"]) .main-header {{
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(45deg, #38a169 0%, #48bb78 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        border-radius: 15px;
        background-color: #f0fff4;
        box-shadow: 0 8px 32px rgba(56,161,105,0.15);
    }}
    
    .stApp:not([data-theme="dark"]) .section-header {{
        font-size: 2rem;
        font-weight: 700;
        color: #38a169;
        border-left: 5px solid #38a169;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
        background: linear-gradient(90deg, rgba(56,161,105,0.1) 0%, rgba(255,255,255,0) 100%);
        border-radius: 5px;
    }}
    
    /* Sidebar - Light Mode */
    .stApp:not([data-theme="dark"]) .css-1d391kg {{
        background: linear-gradient(180deg, #2d3748 0%, #38a169 100%) !important;
    }}
    
    .stApp:not([data-theme="dark"]) .css-1d391kg .stSelectbox, 
    .stApp:not([data-theme="dark"]) .css-1d391kg .stFileUploader,
    .stApp:not([data-theme="dark"]) .css-1d391kg .stTextInput {{
        background-color: rgba(255,255,255,0.95) !important;
        color: #2d3748 !important;
        border-radius: 12px;
        padding: 12px;
        margin: 12px 0;
        border: 2px solid rgba(255,255,255,0.3);
    }}
    
    /* Sidebar text - Light Mode */
    .stApp:not([data-theme="dark"]) .css-1d391kg,
    .stApp:not([data-theme="dark"]) .css-1d391kg p,
    .stApp:not([data-theme="dark"]) .css-1d391kg div,
    .stApp:not([data-theme="dark"]) .css-1d391kg span,
    .stApp:not([data-theme="dark"]) .css-1d391kg label,
    .stApp:not([data-theme="dark"]) .css-1d391kg h1,
    .stApp:not([data-theme="dark"]) .css-1d391kg h2,
    .stApp:not([data-theme="dark"]) .css-1d391kg h3 {{
        color: white !important;
        font-weight: 600;
    }}
    
    /* Buttons - Light Mode */
    .stApp:not([data-theme="dark"]) .stButton>button {{
        background: linear-gradient(45deg, #38a169 0%, #48bb78 100%);
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stApp:not([data-theme="dark"]) .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(56,161,105,0.3);
        background: linear-gradient(45deg, #2f855a 0%, #38a169 100%);
    }}
    
    /* Select boxes - Light Mode */
    .stApp:not([data-theme="dark"]) .stSelectbox>div>div {{
        background: white;
        border-radius: 15px;
        border: 2px solid #cbd5e0;
        transition: all 0.3s ease;
        color: #2d3748 !important;
    }}
    
    .stApp:not([data-theme="dark"]) .stSelectbox>div>div:hover {{
        border-color: #38a169;
        box-shadow: 0 0 15px rgba(56,161,105,0.2);
    }}
    
    /* Images - Light Mode */
    .stApp:not([data-theme="dark"]) .stImage {{
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(56,161,105,0.15);
        transition: all 0.3s ease;
        margin: 1rem 0;
        border: 3px solid #e2e8f0;
    }}
    
    .stApp:not([data-theme="dark"]) .stImage:hover {{
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(56,161,105,0.25);
        border-color: #38a169;
    }}
    
    /* ===== DARK MODE (DARK & YELLOW) ===== */
    .stApp[data-theme="dark"] {{
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        color: #e2e8f0 !important;
    }}
    
    /* Headers - Dark Mode */
    .stApp[data-theme="dark"] .main-header {{
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(45deg, #ecc94b 0%, #f6e05e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        border-radius: 15px;
        background-color: #2d3748;
        box-shadow: 0 8px 32px rgba(236,201,75,0.2);
    }}
    
    .stApp[data-theme="dark"] .section-header {{
        font-size: 2rem;
        font-weight: 700;
        color: #ecc94b;
        border-left: 5px solid #ecc94b;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
        background: linear-gradient(90deg, rgba(236,201,75,0.15) 0%, rgba(255,255,255,0) 100%);
        border-radius: 5px;
    }}
    
    /* Sidebar - Dark Mode */
    .stApp[data-theme="dark"] .css-1d391kg {{
        background: linear-gradient(180deg, #d69e2e 0%, #ecc94b 100%) !important;
    }}
    
    .stApp[data-theme="dark"] .css-1d391kg .stSelectbox,
    .stApp[data-theme="dark"] .css-1d391kg .stFileUploader,
    .stApp[data-theme="dark"] .css-1d391kg .stTextInput {{
        background-color: rgba(45,55,72,0.95) !important;
        color: #e2e8f0 !important;
        border-radius: 12px;
        padding: 12px;
        margin: 12px 0;
        border: 2px solid rgba(236,201,75,0.3);
    }}
    
    /* Sidebar text - Dark Mode */
    .stApp[data-theme="dark"] .css-1d391kg,
    .stApp[data-theme="dark"] .css-1d391kg p,
    .stApp[data-theme="dark"] .css-1d391kg div,
    .stApp[data-theme="dark"] .css-1d391kg span,
    .stApp[data-theme="dark"] .css-1d391kg label,
    .stApp[data-theme="dark"] .css-1d391kg h1,
    .stApp[data-theme="dark"] .css-1d391kg h2,
    .stApp[data-theme="dark"] .css-1d391kg h3 {{
        color: #1a202c !important;
        font-weight: 600;
    }}
    
    /* Buttons - Dark Mode */
    .stApp[data-theme="dark"] .stButton>button {{
        background: linear-gradient(45deg, #d69e2e 0%, #ecc94b 100%);
        color: #1a202c !important;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stApp[data-theme="dark"] .stButton>button:hover {{
        box-shadow: 0 8px 25px rgba(236,201,75,0.3);
        background: linear-gradient(45deg, #b7791f 0%, #d69e2e 100%);
    }}
    
    /* Select boxes - Dark Mode */
    .stApp[data-theme="dark"] .stSelectbox>div>div {{
        background: #2d3748;
        border-radius: 15px;
        border: 2px solid #4a5568;
        transition: all 0.3s ease;
        color: #e2e8f0 !important;
    }}
    
    .stApp[data-theme="dark"] .stSelectbox>div>div:hover {{
        border-color: #ecc94b;
        box-shadow: 0 0 15px rgba(236,201,75,0.2);
    }}
    
    /* Images - Dark Mode */
    .stApp[data-theme="dark"] .stImage {{
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(236,201,75,0.15);
        transition: all 0.3s ease;
        margin: 1rem 0;
        border: 3px solid #4a5568;
    }}
    
    .stApp[data-theme="dark"] .stImage:hover {{
        box-shadow: 0 20px 40px rgba(236,201,75,0.25);
        border-color: #ecc94b;
    }}
    
    /* ===== COMMON STYLES ===== */
    /* Sliders */
    .stApp:not([data-theme="dark"]) .stSlider>div>div>div {{
        background: linear-gradient(90deg, #38a169 0%, #48bb78 100%);
    }}
    
    .stApp[data-theme="dark"] .stSlider>div>div>div {{
        background: linear-gradient(90deg, #d69e2e 0%, #ecc94b 100%);
    }}
    
    /* Cards */
    .custom-card {{
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }}
    
    .stApp:not([data-theme="dark"]) .custom-card {{
        background: white;
        border-left: 4px solid #38a169;
    }}
    
    .stApp[data-theme="dark"] .custom-card {{
        background: #2d3748;
        box-shadow: 0 5px 20px rgba(236,201,75,0.15);
        border-left: 4px solid #ecc94b;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-radius: 20px;
        border: 2px solid rgba(56,161,105,0.2);
    }}
    
    .stApp:not([data-theme="dark"]) .footer {{
        background: linear-gradient(90deg, rgba(56,161,105,0.1) 0%, rgba(72,187,120,0.1) 100%);
    }}
    
    .stApp[data-theme="dark"] .footer {{
        background: linear-gradient(90deg, rgba(214,158,46,0.1) 0%, rgba(236,201,75,0.1) 100%);
        border-color: rgba(236,201,75,0.2);
    }}
    
    /* ===== GLOBAL FIXES ===== */
    /* Ensure all text is readable */
    [data-testid="stMarkdownContainer"], 
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] div {{
        color: inherit !important;
    }}
    
    /* Fix dropdown options */
    .stSelectbox option {{
        color: #2d3748 !important;
        background: white !important;
    }}
    
    .stApp[data-theme="dark"] .stSelectbox option {{
        color: #e2e8f0 !important;
        background: #2d3748 !important;
    }}
    
    /* Theme-specific body class for better targeting */
    body.theme-light {{
        --primary-color: #38a169;
        --background-color: #ffffff;
    }}
    
    body.theme-dark {{
        --primary-color: #ecc94b;
        --background-color: #1a202c;
    }}
</style>
""", unsafe_allow_html=True)

# Add body class for theme
st.markdown(f'<body class="theme-{css_theme.lower()}">', unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("🖼️ OpenCV Processor")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    # Theme toggle with proper functionality
    theme = st.selectbox("Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1, key="theme_selector")
    
    # Update session state when theme changes
    if theme != st.session_state.theme:
        st.session_state.theme = theme
        st.rerun()
    
    # Navigation
    st.markdown("---")
    app_mode = st.selectbox("Choose Processing Type", 
                           ["Home", "Color Transformations", "Geometric Transformations", 
                            "Filtering & Smoothing", "Edge Detection", "Thresholding", 
                            "Histogram Operations", "Contours"])

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
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

# -------------------- COLOR TRANSFORMATIONS --------------------
elif app_mode == "Color Transformations":
    st.markdown('<p class="section-header">Color Space Transformations</p>', unsafe_allow_html=True)
    
    image = get_image()
    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
        
        with col2:
            option = st.selectbox("Select Transformation", 
                                 ["Grayscale", "HSV"])
            
            if option == "Grayscale":
                result = ip.convert_to_grayscale(image)
                st.image(result, caption="Grayscale Image", use_container_width=True, channels='GRAY')
            elif option == "HSV":
                result = ip.convert_to_hsv(image)
                # Convert HSV to RGB for display (HSV display is not straightforward)
                st.image(cv2.cvtColor(result, cv2.COLOR_HSV2RGB), caption="HSV Image", use_container_width=True)

# -------------------- GEOMETRIC TRANSFORMATIONS --------------------
elif app_mode == "Geometric Transformations":
    st.markdown('<p class="section-header">Geometric Transformations</p>', unsafe_allow_html=True)
    
    image = get_image()
    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
        
        with col2:
            option = st.selectbox("Select Transformation", 
                                 ["Flip", "Rotate", "Resize"])
            
            if option == "Flip":
                flip_code = st.selectbox("Flip Direction", 
                                        ["Horizontal (1)", "Vertical (0)", "Both (-1)"])
                code_map = {"Horizontal (1)": 1, "Vertical (0)": 0, "Both (-1)": -1}
                result = ip.flip_image(image, code_map[flip_code])
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Flipped Image", use_container_width=True)
                
            elif option == "Rotate":
                angle = st.slider("Rotation Angle", -180, 180, 45)
                result = ip.rotate_image(image, angle)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"Rotated {angle}°", use_container_width=True)
                
            elif option == "Resize":
                width = st.slider("Width", 50, 1000, image.shape[1])
                height = st.slider("Height", 50, 1000, image.shape[0])
                result = ip.resize_image(image, width=width, height=height)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Resized Image", use_container_width=True)

# -------------------- ADD MORE SECTIONS HERE --------------------
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