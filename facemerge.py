import cv2
import mediapipe as mp
import numpy as np
import os
import uuid
import shutil
from typing import Optional, Tuple, List
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
import uvicorn

# --- Configuration ---
OUTPUT_DIR = "outputs"
DEBUG_MODE = False  
MAX_DIM = 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
_face_mesh_detector = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------
# Utility Functions
# -----------------------
def resize_if_large(img: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def get_landmarks(image: np.ndarray) -> Optional[np.ndarray]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = _face_mesh_detector.process(rgb)
    if not results.multi_face_landmarks:
        return None
    h, w = image.shape[:2]
    lm = results.multi_face_landmarks[0].landmark
    pts = np.array([[lm[i].x * w, lm[i].y * h] for i in range(len(lm))], dtype=np.float32)
    return pts

def get_triangles(points: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Returns a static list of triangles for a set of face landmarks.
    Using a subset of MediaPipe points (approx 68 equivalent).
    """
    # Specifically selected 68-point equivalent indices from MediaPipe's 468 pts
    indices = [
        162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        # Eyes/Brows
        70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
        # Nose
        168, 6, 197, 195, 5, 4, 1, 19, 94, 2,
        # Mouth
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        # Eyeballs
        33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380
    ]
    
    # We'll use these points to compute Delaunay. 
    # To keep it perfect, we'll only use these points for the warp.
    pts = points[indices]
    rect = (0, 0, 1, 1) # Normalized
    subdiv = cv2.Subdiv2D((0, 0, MAX_DIM*2, MAX_DIM*2))
    
    for p in pts:
        subdiv.insert((float(p[0]), float(p[1])))
    
    tris_list = subdiv.getTriangleList()
    triangles = []
    for t in tris_list:
        pt1 = (t[0], t[1]); pt2 = (t[2], t[3]); pt3 = (t[4], t[5])
        
        idx1 = np.argmin(np.linalg.norm(pts - pt1, axis=1))
        idx2 = np.argmin(np.linalg.norm(pts - pt2, axis=1))
        idx3 = np.argmin(np.linalg.norm(pts - pt3, axis=1))
        
        if len(set([idx1, idx2, idx3])) == 3:
            # Map back to global indices
            triangles.append((indices[idx1], indices[idx2], indices[idx3]))
            
    return triangles

def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1_rect = []; t2_rect = []
    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    size = (r2[2], r2[3])
    M = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    img2_rect = cv2.warpAffine(img1_rect, M, size, None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    img2_rect = img2_rect * mask
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect

# -----------------------
# Advanced Blending
# -----------------------
def match_histograms(src, dst, mask):
    """
    Matches the histogram of src to dst in the masked area.
    """
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    dst_lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
    
    mask_bool = mask > 0
    if not np.any(mask_bool): return src
    
    res_lab = src_lab.copy()
    for i in range(3):
        s_chan = src_lab[..., i][mask_bool]
        d_chan = dst_lab[..., i][mask_bool]
        
        # Simple hist match via mean/std (very stable for skin)
        s_mean, s_std = s_chan.mean(), s_chan.std()
        d_mean, d_std = d_chan.mean(), d_chan.std()
        
        if s_std > 0:
            res_chan = (src_lab[..., i].astype(float) - s_mean) * (d_std / s_std) + d_mean
        else:
            res_chan = src_lab[..., i].astype(float) - s_mean + d_mean
            
        res_lab[..., i] = np.clip(res_chan, 0, 255).astype(np.uint8)
        
    return cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)

# -----------------------
# Core Merge Pipeline
# -----------------------
def smart_face_merge(src_path: str, dst_path: str) -> Optional[str]:
    img1 = cv2.imread(src_path)
    img2 = cv2.imread(dst_path)
    if img1 is None or img2 is None: return None

    img1 = resize_if_large(img1); img2 = resize_if_large(img2)
    
    # 1. Landmarks
    p1 = get_landmarks(img1); p2 = get_landmarks(img2)
    if p1 is None or p2 is None: return None

    # 2. Warp
    img2_new_face = np.zeros_like(img2, dtype=np.float32)
    triangles = get_triangles(p2)
    
    for tri in triangles:
        t1 = [p1[tri[0]], p1[tri[1]], p1[tri[2]]]
        t2 = [p2[tri[0]], p2[tri[1]], p2[tri[2]]]
        warp_triangle(img1.astype(np.float32), img2_new_face, t1, t2)

    img2_new_face = img2_new_face.astype(np.uint8)
    
    # 3. Create Face Mask Hull
    # Get the outer boundary points of the face for a perfect blend
    outer_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    hull = cv2.convexHull(np.int32(p2[outer_indices]))
    mask = np.zeros(img2.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    
    # 4. Color Matching
    img2_new_face = match_histograms(img2_new_face, img2, mask)

    # 5. Seamless Cloning
    # Calculate center of mass for cloning
    M = cv2.moments(mask)
    if M["m00"] == 0: return None
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    try:
        # We use a slightly blurred mask for Poisson blending to avoid hard edges
        mask_clone = cv2.GaussianBlur(mask, (11, 11), 0)
        output = cv2.seamlessClone(img2_new_face, img2, mask_clone, center, cv2.NORMAL_CLONE)
    except Exception:
        # Fallback to alpha blend if Poisson fails (unlikely)
        m_3 = cv2.cvtColor(mask/255.0, cv2.COLOR_GRAY2BGR)
        output = (img2_new_face * m_3 + img2 * (1 - m_3)).astype(np.uint8)

    output_path = os.path.join(OUTPUT_DIR, f"result_{uuid.uuid4().hex[:8]}.jpg")
    cv2.imwrite(output_path, output)
    return output_path

# -----------------------
# App & Entry
# -----------------------
app = FastAPI(title="Gemini Style Face Merge")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

@app.post("/merge_faces/")
async def merge_faces_api(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    s_p = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}_s.jpg")
    d_p = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}_d.jpg")
    with open(s_p, "wb") as f: shutil.copyfileobj(file1.file, f)
    with open(d_p, "wb") as f: shutil.copyfileobj(file2.file, f)
    res = smart_face_merge(s_p, d_p)
    return {"status": "success", "url": f"/outputs/{os.path.basename(res)}"} if res else {"status": "error"}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("=== Gemini Style Face Merge ===")
        src = input("Source Image: ").strip()
        dst = input("Destination Image: ").strip()
        if os.path.exists(src) and os.path.exists(dst):
            res = smart_face_merge(src, dst)
            if res: print(f"Output: {res}")
            else: print("Merge Failed")
        else: print("Paths Invalid")
