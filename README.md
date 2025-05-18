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
## 📁 File Overview

| File / Notebook            | Description |
|---------------------------|-------------|
| `train.py`                | Main training script for the captioning model |
| `preprocessing.py`        | Preprocessing script to prepare image-caption pairs and features |
| `inference_inhance.ipynb` | Jupyter notebook for generating captions using the trained model (with enhancements) |
| `ViT-Large_train.pkl (MSCOCO dataset 2017)`     | Precomputed vision transformer features for training set |
| `ViT-Large_val.pkl (MSCOCO dataset 2017)`       | Precomputed vision transformer features for validation set |


## 📦 Precomputed Dataset Features

Due to file size limitations, the `.pkl` feature files for the ViT visual embeddings are hosted on Google Drive.

You can download them from the following links:

- [Google Drive](https://drive.google.com/drive/folders/15AoZ7bYJV3DRTzRMspMYJB-lOQDgv-FJ?usp=share_link)



