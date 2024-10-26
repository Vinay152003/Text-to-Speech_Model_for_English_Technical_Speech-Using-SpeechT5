<h3><a href="">Text-to-Speech Model for English Technical Speech Using SpeechT5</a></h3>
<a href="https://huggingface.co/spaces/Vinay15/Text-to-Speech_Model_for_English_Technical_Speech"><img src="https://img.shields.io/badge/Huggingface-yellow"></a>
<a href="https://www.linkedin.com/in/vinay-hipparge/"><img src="https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/vinay-hipparge/"></a>
<a href="mailto:vinayhipparge15@gmail.com"><img src="https://img.shields.io/badge/Gmail--informational?style=social&logo=gmail"></a>
<a href="https://colab.research.google.com/drive/1RvkNzRX_OgKK8u_OAy43OczupjXaBpGz?usp=sharing"><img src="https://img.shields.io/badge/Google-Colab-red"></a>

### Live URL
You can access the Live Web Application at My Hugging Face Space: [https://huggingface.co/spaces/Vinay15/Text-to-Speech_Model_for_English_Technical_Speech](https://huggingface.co/spaces/Vinay15/Text-to-Speech_Model_for_English_Technical_Speech)
[Note: The Hugging Face space may be in a sleeping state due to inactivity. Please click the link to restart the space]

This repository contains an implementation of a text-to-speech (TTS) model using the SpeechT5 architecture, tailored for technical jargon. The model improves the pronunciation of technical terms by leveraging a custom pronunciation dictionary. It segments input text, generates speech using speaker embeddings for consistent voice quality, and outputs audio in WAV format. A Gradio interface is included for easy user interaction.

Gradio interface: https://huggingface.co/spaces/Vinay15/Text-to-Speech_Model_for_English_Technical_Speech

![Screenshot 11](https://github.com/user-attachments/assets/8a97f209-701c-4b6b-88ac-181e51e0cc6d)


Audio Sample of Task 1 Pre-trained Model Using Coqui TTS and Fine-Tuned Model Using SpeechT5:

-Task 1 Pre-trained Model Audio using Coqui TTS: https://github.com/user-attachments/assets/2ad57d3f-ba5c-4b6f-9ac9-1d54904d2d57

-Task 1 Fine-tuned Model Audio Sample using SpeechT5: https://github.com/user-attachments/assets/b63f9f41-9714-4634-9786-c11c2a55aa92

## Table of Contents
- [Introduction](#introduction)
- [Environment Setup](#environment-setup)
- [Install](#Install)
- [OCR Model Integration](#ocr-model-integration)
- [Web Application Development](#web-application-development)
- [Deployment](#deployment)
- [Live URL](#Live-URL)
- [Usage](#usage)
- [License](#license)

## Introduction

This project fine-tunes a text-to-speech (TTS) model using SpeechT5, focusing on technical vocabulary and improving pronunciation accuracy for specialized terms. It includes a custom pronunciation dictionary and processes text in segments to ensure clear, accurate speech synthesis, maintaining consistent voice through speaker embeddings. Users interact with the model via a Gradio interface, where they can input technical phrases and receive clear audio output. This setup is ideal for applications that require precise verbalization of technical language.

## Environment Setup

### Requirements

- Necessary Libraries:
  -'gradio'
  -'torch'
  -'transformers'
  -'datasets'
  -'soundfile'
  -'sentencepiece'

