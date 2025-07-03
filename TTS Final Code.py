import streamlit as st
import speech_recognition as sr
from googletrans import Translator, LANGUAGES
from gtts import gTTS
import playsound
import os
import base64
import time

# Set page config
st.set_page_config(
    page_title="SpeechTrans - Real-time Translation",
    page_icon="TTS Page Icon Image.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to convert image to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        img_data = f.read()
    return base64.b64encode(img_data).decode()

# Custom CSS for styling
def set_custom_style():
    # Load background image
    try:
        bg_base64 = get_image_base64("C:\\Users\\admin1\\Downloads\\Improved Codes\\Improved Codes\\TTS Background.png")
    except FileNotFoundError:
        st.error("Background image not found. Using default gradient.")
        bg_base64 = ""

    st.markdown(f"""
    <style>
    /* Main background with image */
    .stApp {{
        background-image: url(data:image/png;base64,{bg_base64});
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}

    /* Remove all horizontal rules */
    hr {{
        display: none !important;
    }}

    /* Remove extra spacing */
    .stMarkdown {{
        margin-bottom: 0 !important;
    }}

    /* Adjusted spacing for content */
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select {{
        margin-top: 0.5rem !important;
    }}

    /* Button styling */
    .stButton>button {{
        border: none;
        border-radius: 25px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s;
       background: linear-gradient(135deg, #FF4081, #4A90E2);
        color: white;
        margin-top: 1rem !important;
    }}

    .stButton>button:hover {{
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }}

    /* Text area styling */
    .stTextArea>div>div>textarea {{
        border-radius: 10px;
        padding: 15px;
        font-size: 16px;
        min-height: 250px;
    }}

    /* Output text area styling */
    .stTextArea>div>div>textarea[disabled] {{
        color: white !important;
        -webkit-text-fill-color: white !important;
        opacity: 1 !important;
    }}

    /* Select box styling */
    .stSelectbox>div>div>select {{
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }}

    /* Header styling */
    .header {{
        color: white;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
        margin-bottom: 0 !important;
    }}

    /* Title container with icon */
    .title-container {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        margin-bottom: 0 !important;
    }}

    .title-image {{
        width: 80px;
        height: 80px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #4CAF50;
    }}

    /* Subtitle styling */
    .subtitle {{
        color: white;
        text-align: center;
        margin-top: 0 !important;
        margin-bottom: 1rem !important;  /* Reduced from 30px */
        font-size: 18px;
    }}

    /* Animation for recording */
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.1); }}
        100% {{ transform: scale(1); }}
    }}

    .pulse-animation {{
        animation: pulse 1.5s infinite;
    }}

    /* Footer styling */
    .footer {{
        text-align: center;
        color: white;
        padding-top: 1rem !important;  /* Reduced from 20px */
    }}
    
    /* Column styling */
    .st-emotion-cache-1v0mbdj {{
        padding-top: 0.5rem !important;
    }}

    .translated-label {{
        color: white !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# [Rest of your code remains exactly the same...]

# Function to recognize speech
def recognize_speech(input_language):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        with st.spinner("Adjusting for ambient noise..."):
            recognizer.adjust_for_ambient_noise(source, duration=3)

        recording_placeholder = st.empty()
        recording_placeholder.markdown(
            '<div class="pulse-animation" style="color: red; font-size: 24px; text-align: center;">🎤 Recording... Speak now</div>',
            unsafe_allow_html=True
        )

        try:
            audio = recognizer.listen(source, timeout=5)
            recording_placeholder.empty()

            with st.spinner("Processing your speech..."):
                text = recognizer.recognize_google(audio, language=input_language)
                return text
        except sr.WaitTimeoutError:
            st.error("No speech detected. Please try again.")
            return None
        except sr.UnknownValueError:
            st.error("Could not understand the audio")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results: {e}")
            return None

# Function to translate text
def translate_text(text, target_language):
    translator = Translator()
    try:
        translation = translator.translate(text, dest=target_language)
        return translation.text
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return None

# Function to convert text to speech
def text_to_speech(text, language):
    try:
        tts = gTTS(text=text, lang=language)
        filename = "translated_speech.mp3"
        tts.save(filename)

        # Auto-play the audio (works in some browsers)
        audio_bytes = open(filename, 'rb').read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
        """
        st.components.v1.html(audio_html, height=0)

        # Clean up
        os.remove(filename)
    except Exception as e:
        st.error(f"Speech synthesis failed: {e}")

# Main app function
def main():
    set_custom_style()

    # Main app UI with title and image
    title_image = r"C:\\Users\\admin1\\Downloads\\Improved Codes\\Improved Codes\\TTS Page Icon Image.png"

    # Header section - completely clean with no extra spacing
    st.markdown(
        f"""
        <div class="title-container">
            <img src="data:image/png;base64,{get_image_base64(title_image)}" class="title-image">
            <h1 class="header">Speech Trans</h1>
        </div>
        <p class="subtitle">Real-time Speech Translation Platform</p>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = ""

    # Create two columns
    col1, col2 = st.columns(2, gap="large")

    # Input section (left column) - removed card divs
    with col1:
        st.markdown("### Input Settings")
        from_lang = st.selectbox(
            "Source Language",
            options=list(LANGUAGES.values()),
            index=0,
            key="from_lang"
        )

        st.markdown("### Input Text")
        input_text = st.text_area(
            "Enter text or use speech recognition",
            value=st.session_state.input_text,
            height=250,
            key="input_text_area",
            placeholder="Type here or click the record button below..."
        )

        if st.button("🎤 Record Speech", key="record_btn"):
            input_lang_code = [code for code, name in LANGUAGES.items() if name == st.session_state.from_lang][0]
            recognized_text = recognize_speech(input_lang_code)
            if recognized_text:
                st.session_state.input_text = recognized_text
                st.experimental_rerun()

    # Output section (right column) - removed card divs
    with col2:
        st.markdown("### Output Settings")
        to_lang = st.selectbox(
            "Target Language",
            options=list(LANGUAGES.values()),
            index=1,
            key="to_lang"
        )

        st.markdown('<h4 class="translated-label">Translated Text</h4>', unsafe_allow_html=True)
        translated_text = st.text_area(
            "Translation will appear here",
            value=st.session_state.translated_text,
            height=250,
            key="translated_text_area",
            disabled=True
        )

        # Action buttons
        col1_btn, col2_btn, col3_btn = st.columns(3)
        with col1_btn:
            if st.button("Translate", key="translate_btn"):
                current_input = st.session_state.input_text_area  # Get current text from textarea
                if current_input:
                    target_lang_code = [code for code, name in LANGUAGES.items() if name == st.session_state.to_lang][0]
                    translated = translate_text(current_input, target_lang_code)
                    if translated:
                        st.session_state.translated_text = translated
                        st.experimental_rerun()
                else:
                    st.warning("Please enter some text or record your voice first")

        with col2_btn:
            if st.button("Speak", key="speak_btn", disabled=not st.session_state.translated_text):
                target_lang_code = [code for code, name in LANGUAGES.items() if name == st.session_state.to_lang][0]
                text_to_speech(st.session_state.translated_text, target_lang_code)

        with col3_btn:
            if st.button("Clear", key="clear_btn"):
                st.session_state.input_text = ""
                st.session_state.translated_text = ""
                st.experimental_rerun()

    # Footer
    st.markdown(
        """
        <div class="footer">
            <p>SpeechTrans v1.0 | Contact Us | FAQ | Privacy Policy</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    if not st.runtime.exists():
        import subprocess
        subprocess.run(["streamlit", "run", __file__])
    else:
        main()