import streamlit as st
import google.generativeai as genai
from PIL import Image
import pyttsx3
import pytesseract  
from langchain_google_genai import GoogleGenerativeAI
import os

# Initialize Google Generative AI with API Key
GEMINI_API_KEY = "............"  # Replace with your valid API key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

llm = GoogleGenerativeAI(model="gemini-1.5-pro", api_key=GEMINI_API_KEY)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Streamlit App
st.set_page_config(layout="wide")
st.title(":blue[👨‍🦯 Assistive AI for the Visually Impaired]💬")
st.subheader("Analyze Images with AI")
st.markdown("""
Upload an image to get:
1. Scene understanding (detailed description of the image).
2. Text recognition and speech output.
""")
st.sidebar.title("🔧 Features")
st.sidebar.markdown("""
- Scene Understanding
- Text-to-Speech
- Object & Obstacle Detection
""")

# Functions for functionality
def extract_text_from_image(image):
    """Extracts text from the given image using OCR."""
    return pytesseract.image_to_string(image)

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed
    engine.setProperty('volume', 1)  # Volume
    audio_file = "text-to-speech-local.mp3"
    
    try:
        engine.save_to_file(text, audio_file)
        engine.runAndWait()
        st.audio(audio_file, format="audio/mp3")
    except Exception as e:
        st.error(f"Audio generation failed: {e}")

def generate_scene_description(system_prompt, image_data):
    """Generates a scene description using Google Generative AI."""
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([system_prompt, image_data[0]])
    return response.text

def input_image_setup(uploaded_file):
    """Prepares the uploaded image for processing."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data,
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded.")


# Upload Image
uploaded_file = st.file_uploader("Upload an image for analysis...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width="auto")

# Input Prompt for Scene Understanding
system_prompt = """
You are an AI assistant helping visually impaired individuals by describing the scene in the image. Provide:
1. List of items detected in the image with their purpose.
2. Overall description of the image.
3. Suggestions for actions or precautions for the visually impaired.
"""

scene_button = st.button("🔍Scene Description")
ocr_button = st.button("📝 Extract Text")
tts_button = st.button("🔊 Convert Text-to-Speech")

# Process user interactions
if uploaded_file:
    image_data = input_image_setup(uploaded_file)

    if scene_button:
        with st.spinner("Generating scene description..."):
            response = generate_scene_description(system_prompt, image_data)
            st.markdown("<h3 class='feature-header'>🔍 Scene Description</h3>", unsafe_allow_html=True)
            st.write(response)

    if ocr_button:
        with st.spinner("Extracting text from the image..."):
            text = extract_text_from_image(image)
            st.markdown("<h3 class='feature-header'>📝 Extracted Text</h3>", unsafe_allow_html=True)
            st.text_area("Extracted Text", text, height=150)

    if tts_button:
        with st.spinner("Converting text to speech..."):
            text = extract_text_from_image(image)
            if text.strip():
                text_to_speech(text)
                st.success("Text-to-Speech Conversion Completed!")
            else:
                st.warning("No text found to convert.")
