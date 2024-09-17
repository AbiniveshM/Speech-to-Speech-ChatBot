import streamlit as st
import pyaudio
import wave
import requests
from io import BytesIO
from gtts import gTTS
from transformers import pipeline
import time

# Constants for Audio Recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
OUTPUT_FILENAME = "audio.wav"

# AssemblyAI API Key
ASSEMBLYAI_API_KEY = "eb515c98fc8d4b5b8b8a93d4e3b4c5b9"

# Initialize PyAudio
p = pyaudio.PyAudio()

def record_audio():
    """Records audio for a fixed duration."""
    st.info("Recording... Press 'Stop' button to finish recording.")
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    # Recording for a fixed duration
    for _ in range(0, int(RATE / CHUNK * 5)):  # 5 seconds recording
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    # Save the recorded data as a WAV file
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return OUTPUT_FILENAME

def upload_audio(file_path):
    """Uploads audio to AssemblyAI and returns the URL."""
    headers = {'authorization': ASSEMBLYAI_API_KEY}
    with open(file_path, 'rb') as audio_file:
        response = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, data=audio_file)
    return response.json().get('upload_url')

def transcribe_audio(audio_url):
    """Transcribes the uploaded audio using AssemblyAI."""
    headers = {'authorization': ASSEMBLYAI_API_KEY}
    transcript_request = {'audio_url': audio_url}
    response = requests.post('https://api.assemblyai.com/v2/transcript', json=transcript_request, headers=headers)
    transcript_id = response.json().get('id')

    # Poll for transcription result
    while True:
        result_response = requests.get(f'https://api.assemblyai.com/v2/transcript/{transcript_id}', headers=headers)
        result_data = result_response.json()
        if result_data['status'] == 'completed':
            return result_data['text']
        elif result_data['status'] == 'failed':
            return "Transcription failed."
        else:
            time.sleep(2)

def get_response(prompt):
    """Generates a response using GPT-2."""
    generator = pipeline('text-generation', model='gpt2')
    responses = generator(prompt, max_length=50, num_return_sequences=1)
    return responses[0]['generated_text']

def text_to_speech(text):
    """Converts text to speech and plays it directly."""
    tts = gTTS(text=text, lang='en', slow=False)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    st.audio(audio_buffer, format="audio/mp3")

# Streamlit Interface
st.title("Speech-to-Speech Chatbot")
st.write("This app records your voice, transcribes it, and responds with AI-generated text.")

if st.button("Record"):
    audio_file = record_audio()
    audio_url = upload_audio(audio_file)
    transcription = transcribe_audio(audio_url)

    st.write(f"Recognized Speech: {transcription}")

    if transcription:
        response = get_response(transcription)
        st.write(f"AI Response: {response}")
        text_to_speech(response)
