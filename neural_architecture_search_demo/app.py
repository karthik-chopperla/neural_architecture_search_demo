import streamlit as st
from ui.interface import run_app
from utils.helpers import set_global_seed

def main():
    st.set_page_config(page_title="Neural Architecture Search", layout="centered")
    set_global_seed(42)  # Optional but helpful for reproducibility
    run_app()

if __name__ == "__main__":
    main()
