import streamlit as st
import pytesseract as tess
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from googletrans import Translator, LANGUAGES
import os
import base64
import subprocess
import sys
import io
import numpy as np
import cv2
from langdetect import detect, DetectorFactory

# For more consistent language detection
DetectorFactory.seed = 0

# Check if running in Streamlit environment
def is_running_in_streamlit():
    return 'streamlit' in sys.modules

# Set Tesseract path (update this to your path)
tess.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def get_image_base64(image_path):
    """Convert image to base64 for HTML embedding"""
    with open(image_path, "rb") as f:
        img_data = f.read()
    return base64.b64encode(img_data).decode()

def detect_text_language(text):
    """Enhanced language detection with fallback"""
    try:
        if len(text.strip()) > 10:  # Need sufficient text for detection
            return detect(text)
        return 'en'  # Default to English if detection fails
    except:
        return 'en'

def analyze_image(image):
    """Enhanced image analysis with more parameters"""
    grayscale = image.convert('L')
    np_img = np.array(grayscale)
    
    # Calculate advanced image statistics
    avg_brightness = np.mean(np_img)
    contrast = np.std(np_img)
    darkness_ratio = np.sum(np_img < 50) / np_img.size
    lightness_ratio = np.sum(np_img > 200) / np_img.size
    
    # Edge detection to determine text density
    edges = cv2.Canny(np_img, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Determine image type with more criteria
    is_document = (avg_brightness > 160) and (contrast > 40) and (edge_density > 0.01)
    is_dark = avg_brightness < 100
    is_low_contrast = contrast < 30
    
    # Calculate automatic parameters with more granularity
    params = {
        'is_document': is_document,
        'is_dark': is_dark,
        'is_low_contrast': is_low_contrast,
        'target_brightness': 170 if is_document else 140,
        'contrast_factor': min(3.0, max(1.0, 60 / max(1, contrast))),
        'sharpness_factor': 1.8 if is_document else 2.2,
        'scale_factor': 2.0 if is_document else 1.5,
        'threshold': 180 if is_document else int(avg_brightness * 0.9),
        'psm_mode': "Single Column (PSM 4)" if is_document else "Sparse Text (PSM 11)",
        'edge_density': edge_density,
        'needs_binarization': lightness_ratio < 0.7 or darkness_ratio > 0.3
    }
    
    return params

def auto_preprocess_image(image, params):
    """Enhanced preprocessing pipeline"""
    # Convert to grayscale
    img = image.convert('L')
    
    # Handle dark images differently
    if params['is_dark']:
        # Invert dark images with light text
        img = ImageOps.invert(img)
    
    # Brightness adjustment
    enhancer = ImageEnhance.Brightness(img)
    current_brightness = np.mean(np.array(img))
    brightness_factor = params['target_brightness'] / max(1, current_brightness)
    img = enhancer.enhance(min(3.0, max(0.3, brightness_factor)))
    
    # Contrast enhancement
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(params['contrast_factor'])
    
    # Sharpness enhancement
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(params['sharpness_factor'])
    
    # Adaptive thresholding if needed
    if params['needs_binarization']:
        # Convert to OpenCV format for advanced thresholding
        np_img = np.array(img)
        
        # Use adaptive thresholding for non-uniform lighting
        if params['is_low_contrast'] or not params['is_document']:
            np_img = cv2.adaptiveThreshold(
                np_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            # Simple threshold for documents
            _, np_img = cv2.threshold(
                np_img, params['threshold'], 255, cv2.THRESH_BINARY
            )
        
        img = Image.fromarray(np_img)
    
    # Noise reduction with selective filtering
    if params['edge_density'] > 0.05:  # High edge density (likely text)
        img = img.filter(ImageFilter.MedianFilter(size=1))
    else:
        img = img.filter(ImageFilter.MedianFilter(size=3))
    
    # Scaling with different methods based on content
    original_width, original_height = img.size
    new_size = (int(original_width * params['scale_factor']), 
                int(original_height * params['scale_factor']))
    
    if params['is_document']:
        img = img.resize(new_size, Image.LANCZOS)
    else:
        # For photos, use bicubic which is better for smooth areas
        img = img.resize(new_size, Image.BICUBIC)
    
    return img

def postprocess_text(text, lang):
    """Clean up extracted text for better translation"""
    # Remove common OCR artifacts
    text = text.replace('\x0c', '')  # Form feed character
    text = text.replace('-\n', '')   # Hyphenated line breaks
    text = text.replace('\n', ' ')   # Replace newlines with spaces
    
    # Language-specific cleanup
    if lang in ['hi', 'mr', 'ne']:  # Indic languages
        text = ''.join(c for c in text if ord(c) < 1280)  # Remove non-Indic garbage
    
    # Remove isolated characters (likely OCR errors)
    words = text.split()
    cleaned_words = [word for word in words if len(word) > 1 or word.isalnum()]
    text = ' '.join(cleaned_words)
    
    return text.strip()

def get_tesseract_lang_code(detected_lang):
    """Map detected language to Tesseract language codes"""
    lang_map = {
        'hi': 'hin',  # Hindi
        'mr': 'mar',  # Marathi
        'ta': 'tam',  # Tamil
        'te': 'tel',  # Telugu
        'ur': 'urd',  # Urdu
        'es': 'spa',  # Spanish
        'ru': 'rus',  # Russian
        'en': 'eng',  # English
        'fr': 'fra',  # French
        'de': 'deu',  # German
        'zh': 'chi_sim',  # Chinese Simplified
        'ja': 'jpn',  # Japanese
        'ko': 'kor',  # Korean
        'ar': 'ara',  # Arabic
    }
    return lang_map.get(detected_lang, 'eng')  # Default to English

def translate_text(text, src_lang, dest_lang):
    """Improved translation function that handles long texts"""
    translator = Translator()
    max_chunk_size = 4500  # Conservative limit to avoid API issues
    
    # Split text into chunks if too long
    if len(text) <= max_chunk_size:
        try:
            translation = translator.translate(text, src=src_lang, dest=dest_lang)
            return translation.text
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            return "Translation failed"
    
    # Split into sentences first to avoid breaking mid-sentence
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + '. '
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Translate each chunk with progress feedback
    translated_chunks = []
    progress_bar = st.progress(0)
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        try:
            progress_bar.progress((i + 1) / total_chunks)
            translation = translator.translate(chunk, src=src_lang, dest=dest_lang)
            translated_chunks.append(translation.text)
        except Exception as e:
            st.error(f"Translation error for chunk {i+1}/{total_chunks}: {str(e)}")
            translated_chunks.append(f"[TRANSLATION ERROR IN PART {i+1}]")
    
    progress_bar.empty()
    return ' '.join(translated_chunks)

def main():
    # Set page config
    if is_running_in_streamlit():
        st.set_page_config(
            page_title="OCR Translation Tool",
            page_icon="OCR Page Icon Image.png",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    # Get background image
    try:
        bg_base64 = get_image_base64("C:\\Users\\admin1\\Downloads\\Improved Codes\\Improved Codes\\WhatsApp Image.jpeg")
    except FileNotFoundError:
        bg_base64 = ""

    # Custom CSS - RESTORED YOUR ORIGINAL STYLING
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{bg_base64});
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}

    /* Semi-transparent content area */
    .main {{
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin: 2rem;
    }}

    /* Modern button styling */
    .stButton>button {{
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }}

    .stButton>button:hover {{
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }}

    /* Text area styling */
    .text-box {{
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 500px;
        overflow-y: auto;
    }}

    .title-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 10px;
    }}

    .title-image {{
        width: 80px;
        height: 80px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #4CAF50;
        margin-right: 10px;
    }}

    /* Header styling */
    .header {{
        color: #a9badb;
        padding-bottom: 10px;
        border-bottom: 2px solid #6c5ce7;
        margin-bottom: 20px;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }}

    /* Language dropdown styling */
    .stSelectbox>div>div>select {{
        background-color: rgba(255,255,255,0.9);
        border: 1px solid #6c5ce7;
    }}
    
    /* Image comparison styling */
    .image-row {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }}
    .image-container {{
        width: 48%;
        text-align: center;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = ""
    if 'target_lang' not in st.session_state:
        st.session_state.target_lang = "Spanish"
    if 'lang_code' not in st.session_state:
        st.session_state.lang_code = "es"
    if 'processed_img' not in st.session_state:
        st.session_state.processed_img = None
    if 'image_params' not in st.session_state:
        st.session_state.image_params = {}
    if 'source_lang' not in st.session_state:
        st.session_state.source_lang = "en"

    # Main UI
    with st.container():
        # Title and Image
        title_image =r"C:\\Users\\admin1\\Downloads\\Improved Codes\\Improved Codes\\OCR Page Icon Image.png"
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{get_image_base64(title_image)}" class="title-image">
                <h1 class="header" style="margin-left: 10px;">OCR Translation Tool</h1>
            </div>  
            <div style="text-align: center;">
                <p style="color: white; font-size: 1.1rem;">Extract and translate text from images with ease</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Language selection
        def update_language():
            st.session_state.target_lang = st.session_state.language_selectbox
            lang_code_list = [code for code, name in LANGUAGES.items() if name == st.session_state.target_lang]
            st.session_state.lang_code = lang_code_list[0] if lang_code_list else "es"

        target_lang = st.selectbox(
            "Select Target Language",
            options=list(LANGUAGES.values()),
            index=37,
            key="language_selectbox",
            on_change=update_language
        )

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload an image or document",
            type=["jpg", "jpeg", "png", "webp", "tif"],
            accept_multiple_files=False
        )

        # Process buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            process_btn = st.button("🚀 Extract & Translate", type="primary", use_container_width=True)
        with col2:
            translate_btn = st.button("🔁 Re-translate", 
                                    disabled=not st.session_state.get('extracted_text', False), 
                                    use_container_width=True)
        with col3:
            clear_btn = st.button("🧹 Clear All", use_container_width=True)

        if clear_btn:
            st.session_state.extracted_text = ""
            st.session_state.translated_text = ""
            st.session_state.processed_img = None
            st.experimental_rerun()

        # Image display
        if uploaded_file is not None:
            # Create columns for image comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(uploaded_file, caption="Original Image", width=400)
            
            with col2:
                if st.session_state.processed_img is not None:
                    st.image(st.session_state.processed_img, caption="Processed Image", width=400)
                else:
                    st.info("Processed image will appear here")

            if process_btn:
                with st.spinner("Analyzing and processing image..."):
                    try:
                        img = Image.open(uploaded_file)
                        
                        # Analyze image and determine optimal parameters
                        st.session_state.image_params = analyze_image(img)
                        
                        # Preprocess image with auto-determined parameters
                        processed_img = auto_preprocess_image(img, st.session_state.image_params)
                        st.session_state.processed_img = processed_img
                        
                        # Show processing parameters
                        with st.expander("Auto-Detected Settings", expanded=True):
                            st.markdown(f"""
                            <div style="background-color: rgba(200, 230, 255, 0.3); border-radius: 10px; padding: 15px;">
                                <p><strong>Image Type:</strong> {'Document' if st.session_state.image_params['is_document'] else 'Photo/Scene Text'}</p>
                                <p><strong>Applied Settings:</strong></p>
                                <ul>
                                    <li>Contrast Enhancement: {st.session_state.image_params['contrast_factor']:.1f}x</li>
                                    <li>Sharpness Enhancement: {st.session_state.image_params['sharpness_factor']:.1f}x</li>
                                    <li>Image Scaling: {st.session_state.image_params['scale_factor']:.1f}x</li>
                                    <li>Threshold: {st.session_state.image_params['threshold']}</li>
                                    <li>OCR Mode: {st.session_state.image_params['psm_mode']}</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # OCR processing with language detection
                        psm_map = {
                            "Auto (PSM 3)": "3",
                            "Single Column (PSM 4)": "4",
                            "Single Line (PSM 7)": "7", 
                            "Sparse Text (PSM 11)": "11",
                            "Sparse Text with OSD (PSM 12)": "12"
                        }
                        psm = psm_map[st.session_state.image_params['psm_mode']]
                        
                        # First pass with English to detect if we have Latin script
                        extracted = tess.image_to_string(
                            processed_img, 
                            config=f"--oem 3 --psm {psm}"
                        )
                        
                        # Detect language from extracted text
                        detected_lang = detect_text_language(extracted)
                        st.session_state.source_lang = detected_lang
                        
                        # Second pass with detected language if not English
                        if detected_lang != 'en':
                            tess_lang = get_tesseract_lang_code(detected_lang)
                            extracted = tess.image_to_string(
                                processed_img,
                                lang=tess_lang,
                                config=f"--oem 3 --psm {psm}"
                            )
                        
                        if not extracted.strip():
                            st.error("No text could be extracted. Try a different image.")
                            st.session_state.extracted_text = ""
                        else:
                            # Post-process the extracted text
                            cleaned_text = postprocess_text(extracted, detected_lang)
                            st.session_state.extracted_text = cleaned_text
                            
                            # Translation with progress feedback
                            with st.spinner("Translating text..."):
                                st.session_state.translated_text = translate_text(
                                    st.session_state.extracted_text,
                                    src_lang=detected_lang,
                                    dest_lang=st.session_state.lang_code
                                )
                        
                        st.experimental_rerun()
                
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")

        # Re-translate functionality
        if translate_btn and st.session_state.get('extracted_text'):
            with st.spinner("Translating..."):
                try:
                    st.session_state.translated_text = translate_text(
                        st.session_state.extracted_text,
                        src_lang=st.session_state.source_lang,
                        dest_lang=st.session_state.lang_code
                    )
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Translation Error: {str(e)}")

        # Text display
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Extracted Text")
            st.text_area(
                "Extracted",
                value=st.session_state.get('extracted_text', ''),
                height=300,
                key="extracted_display",
                label_visibility="collapsed"
            )
        with col2:
            st.markdown("### Translated Text")
            st.text_area(
                "Translated",
                value=st.session_state.get('translated_text', ''),
                height=300,
                key="translated_display",
                label_visibility="collapsed"
            )

        # Download buttons
        if st.session_state.get('extracted_text'):
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Extracted Text",
                    data=st.session_state.extracted_text,
                    file_name="extracted_text.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col2:
                if st.session_state.get('translated_text'):
                    st.download_button(
                        label="📥 Download Translated Text",
                        data=st.session_state.translated_text,
                        file_name="translated_text.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

if __name__ == "__main__":
    if not st.runtime.exists():
        import subprocess
        subprocess.run(["streamlit", "run", __file__])
    else:
        main()