from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
from model import classify_image
import tempfile
import os

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lưu file tạm thời
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Thực hiện dự đoán
        result, confidence = classify_image(temp_path)
        
        # Xóa file tạm
        os.unlink(temp_path)
        
        return {
            "skin_type": result,
            "confidence": f"{confidence:.2%}"
        }
    except Exception as e:
        # Đảm bảo xóa file tạm nếu có lỗi
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise Exception(f"Lỗi khi xử lý ảnh: {str(e)}")

@app.get("/")
async def root():
    return {"message": "API nhận diện loại da"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 