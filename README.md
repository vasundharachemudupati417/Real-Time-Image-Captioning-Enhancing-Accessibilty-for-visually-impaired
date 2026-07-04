# 🧠 Vision-Language Assistive AI (Image Captioning)

A comprehensive, end-to-end Machine Learning pipeline that evolves from basic data loading to an industrial-grade **Vision-Language Model**. The final SOTA (State-of-the-Art) architecture utilizes **BLIP-Base (Bootstrapping Language-Image Pre-training)**, fine-tuned for high-accuracy image understanding and caption generation.

## 🚀 Project Evolution & Phases

This repository contains the complete progression of the project across multiple phases, demonstrating a deep understanding of the Machine Learning lifecycle:

### 🔹 Phase 1: Data Pipeline & Preprocessing
* **File:** `image_captioning_phase1.ipynb`
* **Focus:** Establishing the foundational data ingestion pipeline.
* **Details:** Handled the Flickr8k dataset, built custom PyTorch DataLoaders, implemented text tokenization, and applied image augmentation/normalization (resizing to 384x384).

### 🔹 Phase 2: Architecture & Baseline Modeling
* **File:** `image_captioning_phase2.ipynb`
* **Focus:** Defining the initial model architecture.
* **Details:** Transitioning from traditional CNN-RNN architectures to modern Vision-Transformers (ViT). Setting up the forward passes, attention mechanisms, and vocabulary mappings.

### 🔹 Phase 3: Training & Initial Evaluation
* **File:** `image_captioning_phase3.ipynb`
* **Focus:** Training loops and baseline metrics.
* **Details:** Implementing Cross-Entropy Loss, AdamW optimization, and training loops. Calculating initial BLEU scores to establish a baseline performance metric.

### 🌟 Phase 5: SOTA BLIP-Base Fine-Tuning & MLOps (Final)
* **File:** `image_captioning_phase5_sota.ipynb`
* **Focus:** Achieving State-of-the-Art results and productizing the model.
* **Details:** 
  * **Two-Phase PEFT Training:** Frozen Vision Encoder (tuning text decoder only), followed by unfrozen end-to-end tuning.
  * **Assistive Tech Demo:** Includes a locally hosted Gradio Web UI with integrated Google Text-to-Speech (gTTS) to read captions aloud for visually impaired users.
  * **Advanced Metrics:** Rigorous industry-standard NLP metrics (BLEU, ROUGE, METEOR, CIDEr).

---

## 📊 Final State-of-The-Art Performance (Phase 5)
Standard classification "accuracy" does not apply to Generative AI. We evaluate linguistic alignment to human ground-truth. Our final fine-tuned model achieved:
- **METEOR Score:** `39.43%` *(Beats most baseline models)*
- **BLEU-4 Score:** `18.2%`
- **ROUGE-L Score:** `42.1%`
- **CIDEr Score:** `61.5%`

## 🔬 MLOps & Research Features
To ensure this model is production-ready, the Phase 5 repository includes advanced analysis methodologies:
1. **Explainable AI (XAI):** Cross-Attention Heatmaps to visualize exactly *where* the model is looking in the image when generating specific words.
2. **Failure Mode Analysis:** Algorithmically isolating and analyzing the worst-performing predictions to understand model limitations.
3. **Production Latency Profiling:** Inference speed testing yielding P95 latency and FPS to prove real-time REST API viability.
4. **Linguistic Diversity (TTR):** Type-Token Ratio analysis proving the model uses a highly diverse vocabulary.
5. **Beam Search Ablation:** Testing beam widths (1, 3, 5, 7) to optimize inference quality vs. compute cost.

## 💻 How to Run the Final Demo
1. Open `image_captioning_phase5_sota.ipynb`.
2. Run the final cell titled **"INTERACTIVE DEMO (GRADIO) + ASSISTIVE TTS"**.
3. The UI will automatically pop open in your web browser. 
4. Drag and drop any image and listen to the AI describe it!
