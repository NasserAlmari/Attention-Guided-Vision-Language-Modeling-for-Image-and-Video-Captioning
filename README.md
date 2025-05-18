# 🖼️ Attention-Guided-Vision-Language-Modeling-for-Image-and-Video-Captioning

This repository presents an advanced image captioning framework that generates detailed and context-aware captions by combining transformer-based encoders and attention-enhanced decoders. It leverages visual and relational cues in the image to improve caption quality and generalization.

---

## 📌 Overview

This model integrates:

- **BLIP Visual Encoder**: Extracts high-level semantic features from the image
- **GPT-2 Language Decoder**: Generates fluent and descriptive text
- **Scene Attention**: Captures holistic scene-level context
- **Object Attention**: Focuses on detailed object-specific features
- **Graph Interaction Attention**: Models relationships between objects using similarity-based graph reasoning

The combination improves both visual understanding and linguistic coherence.

---

## 🛠 Features

- Transformer-based vision-language architecture
- Fine-grained and contextual attention modules
- Supports MSCOCO, NoCaps, and custom datasets
- Evaluation using CIDEr, BLEU, METEOR, ROUGE-L, and SPICE
- Modular design for easy extension and experimentation

---

## 📁 Directory Structure
├── data/ # Processed datasets and annotations ├── models/ # Model definitions │ ├── blip_encoder.py # Visual encoder │ ├── gpt2_decoder.py # Language decoder │ ├── attention_modules.py # Scene/Object/Graph Attention ├── utils/ # Helper functions and evaluation scripts ├── train.py # Model training script ├── evaluate.py # Model evaluation script ├── configs/ # YAML config files ├── checkpoints/ # Saved model weights └── README.md # Project documentation

