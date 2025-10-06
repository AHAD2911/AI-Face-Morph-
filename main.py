from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles   # <-- this is required
import cv2, numpy as np, os, uuid
from face_merge import merge_faces

app = FastAPI()
UPLOAD_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static folder
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.post("/merge_faces/")
async def merge_faces_api(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = cv2.imdecode(np.frombuffer(await file1.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(await file2.read(), np.uint8), cv2.IMREAD_COLOR)

    try:
        output = merge_faces(img1, img2)
    except Exception as e:
        return {"error": str(e)}

    filename = f"merged_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(UPLOAD_DIR, filename)
    cv2.imwrite(output_path, output)

    uri = f"http://127.0.0.1:8000/outputs/{filename}"
    return {"uri": uri}
