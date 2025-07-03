import streamlit as st
import os
import tempfile
import base64
import warnings
import re
import time
import logging
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
from pydub import AudioSegment, effects
from moviepy.config import change_settings
import whisper_timestamped as whisper
from gtts import gTTS
from deep_translator import GoogleTranslator
import soundfile as sf
from librosa.feature import rms

# Suppress Streamlit warnings
warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext")

# Configure ImageMagick path (update this to your actual ImageMagick path)
change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_translator.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            transition: all 0.5s ease;
            background-image: url(data:image/webp;base64,{b64_encoded});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        .stApp:hover {{
            opacity: 1 !important;
        }}
        
        .ui-hidden {{
            opacity: 0;
        }}
        
        .ui-hidden .main, 
        .ui-hidden .stHeader, 
        .ui-hidden .stButton,
        .ui-hidden .stVideo,
        .ui-hidden .stDownloadButton {{
            visibility: hidden;
            transition: visibility 0s 0.5s, opacity 0.5s ease;
        }}
        
        .main {{
            background-color: rgba(255, 255, 255, 0.1);
            padding: 2rem;
            border-radius: 10px;
            margin: 2rem 0;
            transition: all 0.5s ease;
        }}
        
        .title-container {{
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 30px;
            transition: all 0.5s ease;
        }}
        
        .title-image {{
            width: 80px;
            height: 80px;
            border-radius: 50%;
            object-fit: cover;
            border: 3px solid #4CAF50;
            transition: all 0.5s ease;
        }}
        
        /* Keep video player visible during playback */
        .stVideo {{
            transition: opacity 0.5s ease;
        }}
        
        .ui-hidden .stVideo:has(video[autoplay]) {{
            visibility: visible;
            opacity: 1;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
    
    # Add JavaScript for auto-hide functionality
    st.markdown("""
    <script>
    // Auto-hide UI after 5 seconds of inactivity
    let timeout;
    const delay = 5000; // 5 seconds
    
    function resetTimer() {
        clearTimeout(timeout);
        document.querySelector('.stApp').classList.remove('ui-hidden');
        timeout = setTimeout(hideUI, delay);
    }
    
    function hideUI() {
        // Don't hide if video is playing
        if(document.querySelector('video[autoplay]') === null) {
            document.querySelector('.stApp').classList.add('ui-hidden');
        }
    }
    
    // Track mouse movement and clicks
    document.addEventListener('mousemove', resetTimer);
    document.addEventListener('click', resetTimer);
    document.addEventListener('keypress', resetTimer);
    
    // Initialize
    resetTimer();
    </script>
    """, unsafe_allow_html=True)

def get_image_base64(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    return base64.b64encode(img_data).decode()

class VideoTranslator:
    def __init__(self):
        self.retry_count = 3
        self.silence_threshold = -50  # dB for silence detection
        self.min_silence_len = 200    # ms
        self.background_volume = 0.4  # Volume level for background music
        
        # Special terms dictionary
        self.glossary = {
            'Bankai': {'ja': '卍解', 'en': 'Bankai'},
            'Hollow': {'ja': '虚', 'en': 'Hollow'},
            'Zanpakutō': {'ja': '斬魄刀', 'en': 'Zanpakutō'},
            'Naruto': {'ja': 'ナルト', 'en': 'Naruto'}
        }

        # Emotion parameters
        self.emotion_map = {
            'happy': {'speed': 1.1, 'pitch': 1.2, 'volume': 1.1},
            'sad': {'speed': 0.9, 'pitch': 0.8, 'volume': 0.9},
            'angry': {'speed': 1.3, 'pitch': 1.1, 'volume': 1.3},
            'neutral': {'speed': 1.0, 'pitch': 1.0, 'volume': 1.0}
        }

    def _clean_text(self, text):
        """Remove special characters but preserve glossary terms"""
        protected = []
        for term in self.glossary:
            if term in text:
                text = text.replace(term, f"@@{term}@@")
                protected.append(term)
        
        text = re.sub(r'[^\w\s.,!?@]', '', text)
        
        for term in protected:
            text = text.replace(f"@@{term}@@", term)
        return text

    def _detect_emotion(self, audio_segment, text):
        """Detect emotion from audio features and text content"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                audio_segment.export(tmp.name, format='wav')
                y, sr = sf.read(tmp.name)
                os.unlink(tmp.name)
            
            energy = np.mean(rms(y=y))
            pitch = np.mean(np.abs(np.diff(y)))
            
            if energy > 0.2 and pitch > 0.1:
                if '!' in text or any(word in text.lower() for word in ['angry', 'mad', 'hate']):
                    return 'angry'
                return 'happy'
            elif energy < 0.1 and pitch < 0.05:
                if any(word in text.lower() for word in ['sad', 'cry', 'tear']):
                    return 'sad'
                return 'sad'
            return 'neutral'
        except:
            return 'neutral'

    def translate_segment(self, text, src_lang, target_lang, emotion='neutral'):
        """Translate text with glossary protection and emotion preservation"""
        if not text.strip():
            return ""
            
        for attempt in range(self.retry_count):
            try:
                for term in self.glossary:
                    if term in text:
                        replacement = self.glossary[term].get(target_lang, term)
                        text = text.replace(term, replacement)
                
                if emotion != 'neutral':
                    if emotion == 'happy':
                        text = f"😊 {text} 😊"
                    elif emotion == 'sad':
                        text = f"😢 {text} 😢"
                    elif emotion == 'angry':
                        text = f"😠 {text} 😠"
                
                translator = GoogleTranslator(source=src_lang, target=target_lang)
                translated = translator.translate(text)
                return translated
                
            except Exception as e:
                if attempt == self.retry_count - 1:
                    logger.error(f"Translation failed: {str(e)}")
                    return text
                time.sleep(1)

    def generate_tts(self, text, lang, duration_ms, emotion='neutral'):
        """Generate emotional speech matching original duration"""
        try:
            text = text.replace('😊', '').replace('😢', '').replace('😠', '').strip()
            params = self.emotion_map.get(emotion, self.emotion_map['neutral'])
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tts = gTTS(text=self._clean_text(text), lang=lang, slow=False)
                tts.save(tmp.name)
                
                speech = AudioSegment.from_mp3(tmp.name)
                
                if params['speed'] != 1.0:
                    speech = speech.speedup(playback_speed=params['speed'])
                
                speech = speech + (20 * np.log10(params['volume']))
                
                if len(speech) > duration_ms:
                    return speech[:duration_ms]
                elif len(speech) < duration_ms:
                    return speech + AudioSegment.silent(duration=duration_ms - len(speech))
                return speech
        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)}")
            return AudioSegment.silent(duration=duration_ms)

    def transcribe_audio(self, audio_path, language=None):
        """Get timed transcript with silence detection and emotion analysis"""
        try:
            audio = whisper.load_audio(audio_path)
            model = whisper.load_model("base")
            raw_audio = AudioSegment.from_file(audio_path)
            result = whisper.transcribe(model, audio, language=language)
            
            segments = []
            last_end = 0
            
            for seg in result['segments']:
                if seg['start'] > last_end:
                    segments.append({
                        'text': "",
                        'start': last_end,
                        'end': seg['start'],
                        'type': 'silence'
                    })
                
                start_ms = int(seg['start'] * 1000)
                end_ms = int(seg['end'] * 1000)
                audio_segment = raw_audio[start_ms:end_ms]
                emotion = self._detect_emotion(audio_segment, seg['text'])
                
                segments.append({
                    'text': seg['text'],
                    'start': seg['start'],
                    'end': seg['end'],
                    'type': 'speech',
                    'emotion': emotion
                })
                last_end = seg['end']
            
            return segments
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return None

    def create_translated_audio(self, segments, original_audio_path, output_path, target_lang):
        """Rebuild audio with translated speech and original background music"""
        try:
            original_audio = AudioSegment.from_file(original_audio_path)
            final_audio = AudioSegment.silent(duration=len(original_audio))
            background_audio = original_audio - (20 * (1 - self.background_volume))
            
            for seg in segments:
                if seg['type'] == 'speech':
                    speech_dur = (seg['end'] - seg['start']) * 1000
                    translated = self.translate_segment(
                        seg['text'], 
                        'auto', 
                        target_lang,
                        emotion=seg.get('emotion', 'neutral')
                    )
                    tts_audio = self.generate_tts(
                        translated, 
                        target_lang, 
                        speech_dur,
                        emotion=seg.get('emotion', 'neutral')
                    )
                    
                    start_ms = seg['start'] * 1000
                    final_audio = final_audio.overlay(tts_audio, position=start_ms)
            
            final_audio = final_audio.overlay(background_audio)
            final_audio.export(output_path, format="mp3")
            return True
            
        except Exception as e:
            logger.error(f"Audio creation failed: {str(e)}")
            return False

    def process_video(self, video_path, target_lang):
        """Complete translation pipeline with emotional processing"""
        try:
            # 1. Extract audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_tmp:
                with VideoFileClip(video_path) as video:
                    video.audio.write_audiofile(audio_tmp.name, codec='pcm_s16le')
                
                # 2. Transcribe with timing and emotion
                segments = self.transcribe_audio(audio_tmp.name)
                if not segments:
                    raise Exception("No segments found in audio")
                
                # 3. Create translated audio
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as translated_audio:
                    if not self.create_translated_audio(segments, audio_tmp.name, translated_audio.name, target_lang):
                        raise Exception("Audio translation failed")
                    
                    # 4. Merge with video
                    output_path = f"{os.path.splitext(video_path)[0]}_{target_lang}_emotional.mp4"
                    with VideoFileClip(video_path) as video:
                        with AudioFileClip(translated_audio.name) as audio:
                            subtitle_clips = []
                            for seg in [s for s in segments if s['type'] == 'speech']:
                                english_text = self.translate_segment(
                                    seg['text'], 
                                    'auto', 
                                    'en',
                                    emotion=seg.get('emotion', 'neutral')
                                )
                                if not english_text.strip():
                                    continue
                                
                                emotion = seg.get('emotion', 'neutral')
                                color = 'white'
                                if emotion == 'happy':
                                    color = 'yellow'
                                elif emotion == 'sad':
                                    color = 'blue'
                                elif emotion == 'angry':
                                    color = 'red'
                                
                                txt_clip = TextClip(
                                    english_text,
                                    font='Arial-Unicode-MS',
                                    fontsize=40,
                                    color=color,
                                    bg_color='black',
                                    size=(video.size[0]*0.9, None),
                                    method='caption'
                                ).set_start(seg['start']).set_end(seg['end']).set_position(('center', 'bottom'))
                                subtitle_clips.append(txt_clip)
                            
                            final = CompositeVideoClip([video.set_audio(audio)] + subtitle_clips)
                            final.write_videofile(
                                output_path,
                                codec='libx264',
                                audio_codec='aac',
                                threads=4,
                                preset='fast',
                                fps=video.fps
                            )
                    
                    return output_path
    
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return None
        finally:
            if 'audio_tmp' in locals() and os.path.exists(audio_tmp.name):
                os.unlink(audio_tmp.name)
            if 'translated_audio' in locals() and os.path.exists(translated_audio.name):
                os.unlink(translated_audio.name)

def main():
    # Set page config
    st.set_page_config(
        page_title="Video Translator Pro",
        page_icon="Video Trans Page Icon Image.png",
        layout="centered"
    )

    # Set background image with auto-hide UI functionality
    set_background("C:\\Users\\admin1\\Downloads\\UI Images\\UI Images\\Background\\Video Trans Background.png")

    # Main app UI with title and image
    title_image = "C:\\Users\\admin1\\Downloads\\New Integrated UI Codes\\New Integrated UI Codes\\Video Trans Page Icon Image.png"
    st.markdown(
        f"""
        <div class="title-container">
            <img src="data:image/png;base64,{get_image_base64(title_image)}" class="title-image">
            <div>
                <h1 style="margin:0;padding:0">Video Translator Pro</h1>
                <h3 style="margin:0;padding:0">Translate video audio with emotional expression</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Language selection with extended options
    lang_options = {
        "🇬🇧 English": "en",
        "🇪🇸 Spanish": "es",
        "🇫🇷 French": "fr",
        "🇩🇪 German": "de",
        "🇮🇹 Italian": "it",
        "🇮🇳 Hindi": "hi",
        "🇯🇵 Japanese": "ja",
        "🇰🇷 Korean": "ko",
        "🇨🇳 Chinese (Simplified)": "zh-cn",
        "🇹🇼 Chinese (Traditional)": "zh-tw"
    }
    
    target_lang = st.selectbox(
        "Select Target Language",
        options=list(lang_options.keys()),
        index=0
    )

    # File upload
    uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file:
        st.video(uploaded_file)
        
        if st.button("🚀 Start Translation", type="primary"):
            with st.spinner("Processing your video ..."):
                try:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        # Save uploaded file
                        video_path = os.path.join(tmp_dir, "original.mp4")
                        with open(video_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process with advanced translator
                        translator = VideoTranslator()
                        output_path = translator.process_video(video_path, lang_options[target_lang])
                        
                        if output_path and os.path.exists(output_path):
                            st.success("Translation Complete!")
                            st.balloons()
                            
                            # Show preview
                            st.video(output_path)
                            
                            # Download button
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    "⬇️ Download Translated Video",
                                    f,
                                    file_name=f"emotional_translation_{lang_options[target_lang]}.mp4",
                                    mime="video/mp4"
                                )
                        else:
                            st.error("Translation failed - check logs for details")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    if not st.runtime.exists():
        import subprocess
        subprocess.run(["streamlit", "run", __file__])
    else:
        main()