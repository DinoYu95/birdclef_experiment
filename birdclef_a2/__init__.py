"""BirdCLEF+ assignment tooling (split, BirdNET embeddings + sklearn, mel-CNN scratch, AudioLDM2 synth).

Fine-tuning BirdNET itself is best done with the upstream BirdNET-Analyzer training stack (Keras/TF).
This repo focuses on embeddings + classical ML, a lightweight mel-spectrogram CNN baseline, and
synthetic augmentation via Hugging Face AudioLDM2 (often referenced as AudioLM2 in briefs).
"""

__version__ = "0.2.0"
