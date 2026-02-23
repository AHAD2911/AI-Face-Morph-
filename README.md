# AI Face Morph 🎭

An advanced AI-powered face merging tool built with Python. This project utilizes MediaPipe's High-Fidelity Face Mesh and OpenCV's Seamless Cloning (Poisson Blending) to create highly realistic and artifact-free face merges.

## ✨ Features

- **High-Precision Landmarks**: Uses MediaPipe 468-point face mesh for pixel-perfect alignment.
- **Advanced Blending**: Implements Poisson Seamless Cloning to match skin tones and textures perfectly.
- **Histogram Matching**: Automatically adjusts color profiles between source and destination faces.
- **FastAPI Backend**: Ready-to-use API for programmatic access.
- **Streamlit Frontend**: A sleek, modern web interface for interactive merging.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- `pip` (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AHAD2911/AI-Face-Morph-.git
   cd AI-Face-Morph-
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### 1. Streamlit Interface (Recommended)
Run the following command to start the web app:
```bash
streamlit run app.py
```

#### 2. FastAPI Server
To serve the API for external integrations:
```bash
python facemerge.py serve
```
The API will be available at `http://localhost:8000`.

#### 3. CLI Mode
Merge images directly from the command line:
```bash
python facemerge.py
```

## 🛠️ Technical Overview

The pipeline follows these steps:
1. **Detection**: Detect landmarks on both source and destination images.
2. **Triangulation**: Perform Delaunay triangulation based on destination landmarks.
3. **Warping**: Warp source triangles to fit destination geometry.
4. **Color Correction**: Apply histogram matching in LAB color space.
5. **Blending**: Use `cv2.seamlessClone` (Poisson blending) to integrate the face into the target image while preserving lighting and shadows.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---
Made by [AHAD2911](https://github.com/AHAD2911)
