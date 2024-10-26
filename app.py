import gradio as gr
import json
import torch
import numpy as np
import re
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf

# Step 1: Load the models and the pronunciation dictionary
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load pronunciation dictionary from JSON file
with open("pronunciation_dict.json", "r") as f:
    pronunciation_dict = json.load(f)

# Function to preprocess and apply pronunciation dictionary
def preprocess_text(text):
    # Convert text to uppercase for uniformity in matching
    text = text.upper()
    for term, phonetic in pronunciation_dict.items():
        # Replace terms with their phonetic equivalents
        text = text.replace(term.upper(), phonetic)
    return text

# Step 2: Define the TTS function with sentence segmentation
def text_to_speech(input_text):
    # Preprocess and segment text
    processed_text = preprocess_text(input_text)
    # Split the processed text by punctuation to form shorter segments
    segments = re.split(r'(?<=[.!?]) +', processed_text)

    # Load speaker embeddings for consistent voice
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    
    audio_outputs = []

    # Generate speech for each text segment
    for segment in segments:
        if segment.strip():  # Ensure the segment is not empty
            inputs = processor(text=segment, return_tensors="pt")
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
            audio_outputs.append(speech.numpy())

    # Concatenate audio from all segments
    complete_speech = np.concatenate(audio_outputs)
    
    # Save the concatenated speech as a .wav file
    output_file = "speech_output.wav"
    sf.write(output_file, complete_speech, samplerate=16000)

    return output_file

# Step 3: Create Gradio interface
iface = gr.Interface(
    fn=text_to_speech,
    inputs="text",
    outputs="audio",
    title="Fine-tuning TTS for Technical Vocabulary",
    description="Enter text with technical jargon for TTS conversion. The model will handle abbreviations and technical terms for better pronunciation."
)

# Step 4: Launch the app
iface.launch(share=True)
