import streamlit as st
import pyaudio
import wave
import time
import os
import whisper
from gtts import gTTS
import pathlib
import textwrap
import os

#how to install google.generativeai package
import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

# Function to record audio
def record_audio(filename, max_recording_time=10):
    # Set up PyAudio
    audio = pyaudio.PyAudio()

    # Open the microphone stream
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=1024)

    # Create an empty list to store audio frames
    frames = []

    start_time = time.time()

    # Continuously read audio data from the stream
    while True:
        data = stream.read(1024)
        frames.append(data)
        elapsed_time = time.time() - start_time
        if elapsed_time >= max_recording_time:
            break

    # Stop the stream and close PyAudio
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a wave file
    wf = wave.open(filename, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b"".join(frames))
    wf.close()

# Function to transcribe audio
def transcribe_audio(filename):
    # Load the Whisper model
    model = whisper.load_model("base")
    # Transcribe the recorded audio
    result = model.transcribe(filename)
    return result["text"]

# Function to generate response using Gemini API
def generate_response(text):
    # Load the Gemini model
    genai.configure(api_key="AIzaSyCoeHJ7xy6GP2HUIVJMDzw1akwy3FrtBaU")
    gem_model = genai.GenerativeModel('gemini-pro')
    response = gem_model.generate_content(text)
    return response.text

# Function to generate speech from text
def generate_speech(text):
    # Use gTTS to generate speech directly from text
    tts = gTTS(text=text, lang='en')
    tts.save("generated_speech.mp3")

# Streamlit app
def main():
    st.title("Voice to Speech Interface")
    
    # Button to start recording
    if st.button("Record Audio"):
        st.write("Recording started...")
        record_audio("recorded_audio.wav")
        st.write("Recording completed.")
        
        # Transcribe recorded audio
        st.write("Transcribing audio...")
        transcription = transcribe_audio("recorded_audio.wav")
        st.write("Transcription:", transcription)
        
        # Generate response using Gemini API
        st.write("Generating response...")
        response_text = generate_response(transcription)
        st.write("Response:", response_text)
        
        # Generate speech from response
        st.write("Generating speech...")
        generate_speech(response_text)
        st.write("Speech generated.")
        
        # Play the generated speech
        st.audio("generated_speech.mp3", format="audio/mp3")

if __name__ == "__main__":
    main()
