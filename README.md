<h3><a href="">Text-to-Speech Model for English Technical Speech Using SpeechT5</a></h3>
<a href="https://huggingface.co/spaces/Vinay15/Text-to-Speech_Model_for_English_Technical_Speech"><img src="https://img.shields.io/badge/Huggingface-yellow"></a>
<a href="https://www.linkedin.com/in/vinay-hipparge/"><img src="https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/vinay-hipparge/"></a>
<a href="mailto:vinayhipparge15@gmail.com"><img src="https://img.shields.io/badge/Gmail--informational?style=social&logo=gmail"></a>
<a href="https://colab.research.google.com/drive/1RvkNzRX_OgKK8u_OAy43OczupjXaBpGz?usp=sharing"><img src="https://img.shields.io/badge/Google-Colab-red"></a>

### Live URL
You can access the Live Web Application at My Hugging Face Space: [https://huggingface.co/spaces/Vinay15/Text-to-Speech_Model_for_English_Technical_Speech](https://huggingface.co/spaces/Vinay15/Text-to-Speech_Model_for_English_Technical_Speech)

[Note: The Hugging Face space may be in a sleeping state due to inactivity. Please click the link to restart the space]

This repository contains an implementation of a text-to-speech (TTS) model using the SpeechT5 architecture, tailored for technical jargon. The model improves the pronunciation of technical terms by leveraging a custom pronunciation dictionary. It segments input text, generates speech using speaker embeddings for consistent voice quality, and outputs audio in WAV format. A Gradio interface is included for easy user interaction.

## Task 1 Report
-Task 1 Report Link: https://drive.google.com/file/d/1CfnpeUi18R7De1uhilYuhMYLS_xXjh2Q/view?usp=sharing


Gradio interface: https://huggingface.co/spaces/Vinay15/Text-to-Speech_Model_for_English_Technical_Speech

![Screenshot 11](https://github.com/user-attachments/assets/8a97f209-701c-4b6b-88ac-181e51e0cc6d)



## Audio-Sample-of-Task-1-Pre-trained-Model-Using-Coqui-TTS-and-Fine-Tuned-Model-Using-SpeechT5:

-Task 1 Pre-trained Model Audio using Coqui TTS: https://github.com/user-attachments/assets/2ad57d3f-ba5c-4b6f-9ac9-1d54904d2d57

-Task 1 Fine-tuned Model Audio Sample using SpeechT5: https://github.com/user-attachments/assets/b63f9f41-9714-4634-9786-c11c2a55aa92

## Table of Contents
- [Introduction](#introduction)
- [Task 1 Report](#Task1Report)
- [Audio-Sample-of-Task-1-Pre-trained-Model-Using-Coqui-TTS-and-Fine-Tuned-Model-Using-SpeechT5:](#Audio-Sample-of-Task-1-Pre-trained-Model-Using-Coqui-TTS-and-Fine-Tuned-Model-Using-SpeechT5:)
- [Environment Setup](#environment-setup)
- [Install](#Install)
- [Dataset Preparation](#DatasetPreparation)
- [Fine-tuning](#Fine-tuning)
- [Results](#Results)
- [Comparison-Summary](#Comparison-Summary)
- [Challenges](#Challenges)
- [Sample-Entries-of-Technical-Terms](#Sample-Entries-of-Technical-Terms)
- [Appendix](#Appendix)
- [License](#license)

## Introduction

This project fine-tunes a text-to-speech (TTS) model using SpeechT5, focusing on technical vocabulary and improving pronunciation accuracy for specialized terms. It includes a custom pronunciation dictionary and processes text in segments to ensure clear, accurate speech synthesis, maintaining consistent voice through speaker embeddings. Users interact with the model via a Gradio interface, where they can input technical phrases and receive clear audio output. This setup is ideal for applications that require precise verbalization of technical language.

## Environment Setup

### Requirements

- Necessary Libraries:
  - `gradio`
  - `gradio`
  - `transformers`
  - `datasets`
  - `soundfile`
  - `sentencepiece`

## Install

You can run SpeechT5 TTS locally with the 🤗 Transformers library.

1. First install the 🤗 [Transformers library](https://github.com/huggingface/transformers), sentencepiece, soundfile and datasets(optional):

```
pip install --upgrade pip
pip install --upgrade transformers sentencepiece datasets[audio]
```

2. Run inference via the `Text-to-Speech` (TTS) pipeline. You can access the SpeechT5 model via the TTS pipeline in just a few lines of code!

```python
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.

speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
```

3. Run inference via the Transformers modelling code - You can use the processor + generate code to convert text into a mono 16 kHz speech waveform for more fine-grained control.

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(text="Hello, my dog is cute.", return_tensors="pt")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("speech.wav", speech.numpy(), samplerate=16000)
```

## Dataset Preparation

• Dataset Description: A dataset was created that includes both general English
sentences and technical jargon. Technical terms included "API," "CUDA," "TTS,"
"OAuth," "REST," and more.

• Data Collection: The dataset was synthesized from:

o Interview transcripts from software engineers and data scientists.

o Sentences pulled from technical blogs and articles.

• Data Size: The dataset consisted of approximately 500 sentences, ensuring a good
balance between technical and conversational language.


## Fine-tuning

![screenshot 22](https://github.com/user-attachments/assets/a115891a-8064-44c9-9c05-23042d1a93f4)

## Results

![results](https://github.com/user-attachments/assets/96894b15-05b7-489d-8af8-6751d7db572a)

## Comparison-Summary 

![Comparssion summary](https://github.com/user-attachments/assets/7a7030e4-578d-4a68-ae3d-6c860240be36)

## Challenges

![Challenges](https://github.com/user-attachments/assets/a8d880dc-3271-4bc5-8e44-fc2268e9f7d9)

## Sample-Entries-of-Technical-Terms

![terms sample](https://github.com/user-attachments/assets/bd6c0a48-86c5-4c4b-96bd-24b07b300299)

## Appendix

![appendix](https://github.com/user-attachments/assets/5f9819e3-5ac1-476f-8ac8-a5a488f0e7a2)
 

