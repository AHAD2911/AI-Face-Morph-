import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
from facemerge import smart_face_merge

# Set page configuration for a premium look
st.set_page_config(
    page_title="AI Face Morph | Premium Face Merging",
    page_icon="🎭",
    layout="wide",
)

# Custom CSS for a sleek, modern UI
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4b6cb7;
        color: white;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #182848;
        transform: translateY(-2px);
    }
    .upload-box {
        border: 2px dashed #4b6cb7;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
    }
    .title-text {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .subtitle-text {
        font-size: 1.2rem;
        color: #a0aec0;
        margin-bottom: 40px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="title-text">AI Face Morph 🎭</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Professional-grade face merging using MediaPipe and advanced Poisson blending.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Source Face")
        src_file = st.file_uploader("Upload the face you want to use", type=['jpg', 'jpeg', 'png'], key="src")
        if src_file:
            st.image(src_file, caption="Source Face", use_column_width=True)

    with col2:
        st.subheader("2. Destination Image")
        dst_file = st.file_uploader("Upload the target person/body", type=['jpg', 'jpeg', 'png'], key="dst")
        if dst_file:
            st.image(dst_file, caption="Destination Image", use_column_width=True)

    st.markdown("---")

    if st.button("✨ Merge Faces"):
        if src_file and dst_file:
            with st.spinner("🚀 Performing advanced AI merge... This might take a few seconds."):
                # Save uploaded files to temporary locations
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_src:
                    tmp_src.write(src_file.getbuffer())
                    src_path = tmp_src.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_dst:
                    tmp_dst.write(dst_file.getbuffer())
                    dst_path = tmp_dst.name

                try:
                    # Use the core logic from facemerge.py
                    result_path = smart_face_merge(src_path, dst_path)
                    
                    if result_path and os.path.exists(result_path):
                        st.balloons()
                        st.success("Analysis Complete! Here is your result.")
                        
                        final_col1, final_col2 = st.columns([1, 2])
                        
                        output_img = Image.open(result_path)
                        st.image(output_img, caption="Final Result", use_column_width=True)
                        
                        with open(result_path, "rb") as file:
                            btn = st.download_button(
                                label="📥 Download Result",
                                data=file,
                                file_name="merged_face.jpg",
                                mime="image/jpeg"
                            )
                    else:
                        st.error("No faces detected in one or both images. Please try with clearer photos.")
                except Exception as e:
                    st.error(f"Merge error: {e}")
                finally:
                    # Cleanup temp files
                    if os.path.exists(src_path): os.remove(src_path)
                    if os.path.exists(dst_path): os.remove(dst_path)
        else:
            st.warning("Please upload both source and destination images first.")

    
    st.sidebar.title("Guidelines")
    st.sidebar.write("- Both images should have a clear front-facing view.")
    st.sidebar.write("- Good lighting improves landmark detection accuracy.")
    st.sidebar.write("- High-resolution images yield better blending results.")

if __name__ == "__main__":
    main()
