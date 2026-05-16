import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

# Mock imports for our core modules
from core.models.segmentation import DentalSegmentationModel
from core.vlm.report_generator import ClinicalReportGenerator

app = FastAPI(
    title="Advanced Dental Radiology AI API",
    description="Multi-modal AI for dental image segmentation and automated clinical reporting."
)

# Initialize models (in a real app, these would load pre-trained weights)
segmentation_model = DentalSegmentationModel()
report_generator = ClinicalReportGenerator()

@app.post("/api/v1/analyze", summary="Analyze Dental X-Ray")
async def analyze_xray(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # 1. Read and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)

        # 2. Run Advanced Segmentation (Teeth, Cavities, Bone Loss)
        seg_results, mask = segmentation_model.predict(img_array)

        # 3. Generate Clinical Report using Vision-Language Model
        report = report_generator.generate_report(image, seg_results)

        return JSONResponse(content={
            "status": "success",
            "findings": seg_results,
            "clinical_report": report,
            "metadata": {
                "image_size": image.size,
                "model_version": "v2.1-multimodal"
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", summary="Health Check")
def health_check():
    return {"status": "healthy", "systems": ["segmentation", "vlm_generator"]}
