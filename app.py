import gradio as gr
import json
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf

# Step 3: Load the models and the pronunciation dictionary
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load pronunciation dictionary from JSON file
with open("pronunciation_dict.json", "r") as f:
    pronunciation_dict = json.load(f)

# Function to preprocess the input text
def preprocess_text(text):
    for term, phonetic in pronunciation_dict.items():
        text = text.replace(term, phonetic)
    return text

# Step 4: Define the TTS function
def text_to_speech(input_text):
    # Preprocess the text
    processed_text = preprocess_text(input_text)

    # Convert the processed text to model inputs
    inputs = processor(text=processed_text, return_tensors="pt")

    # Load xvector embeddings from dataset for speaker voice characteristics
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    # Generate speech using the model and vocoder
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    # Save the generated speech as a .wav file
    output_file = "speech_output.wav"
    sf.write(output_file, speech.numpy(), samplerate=16000)

    return output_file

# Step 5: Create Gradio interface
iface = gr.Interface(fn=text_to_speech,
                     inputs="text",
                     outputs="audio",
                     title="Text-to-Speech (TTS) Application",
                     description="Enter text with technical jargon for TTS conversion.")

# Step 6: Launch the app
iface.launch(share=True)
