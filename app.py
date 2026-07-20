import streamlit as st
import sys
sys.path.append("src")
from main import run_pipeline

st.title("Radar Target Detection Demo")

if st.button("Run Pipeline"):
    run_pipeline()
    st.success("Done! Results below.")
    st.image("output/02_cfar_detection.png")
    st.image("output/04_confusion_matrix.png")