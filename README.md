# Advanced Dental Radiology AI (DentalVLM-Seg)

This project represents a "next-level" approach to dental radiology by combining **Deep Medical Segmentation** with **Vision-Language Models (VLMs)** to not only detect pathologies but to automatically generate structured clinical reports.

## 🚀 The Vision
Traditional models just draw bounding boxes. This pipeline:
1. **Segments** panoramic and bitewing X-rays to identify individual teeth, bone loss, and caries (cavities) using an **Attention U-Net** from MONAI (Medical Open Network for AI).
2. **Analyzes** the segmented regions of interest (ROIs) and passes them to a **Vision-Language Model (VLM)**.
3. **Generates** a natural language clinical report ready for dentist review.

## 📁 Architecture
```
.
├── backend/
│   └── main.py                 # FastAPI application serving the inference pipeline
├── core/
│   ├── models/
│   │   └── segmentation.py     # MONAI-based Attention U-Net for medical imaging
│   └── vlm/
│       └── report_generator.py # Integration with Transformers (HuggingFace) for automated reporting
├── docker-compose.yml          # Container orchestration for seamless deployment
└── requirements.txt            # Python dependencies
```

## 🛠️ Tech Stack
*   **Deep Learning:** PyTorch, MONAI (Medical Open Network for AI)
*   **Vision-Language:** Hugging Face `transformers` (LLaVA / Moondream style models)
*   **Backend Inference:** FastAPI, Uvicorn
*   **Computer Vision:** OpenCV, NumPy

## 🚦 Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the API Server:
   ```bash
   uvicorn backend.main:app --reload
   ```
3. (Optional) Run with Docker:
   ```bash
   docker-compose up --build
   ```

## 🔬 Next Steps for Development
- [ ] **Dataset Integration:** Connect to public datasets like UFBA-UESC Dental Images or Tufts Dental Database.
- [ ] **Train the Segmentation Model:** Implement the training loop in `core/models`.
- [ ] **Fine-tune the VLM:** Fine-tune a lightweight Vision-Language model on dental annotations.
