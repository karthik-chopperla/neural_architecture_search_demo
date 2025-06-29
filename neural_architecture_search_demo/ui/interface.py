# Placeholder for ui/interface.py
# ui/interface.py

import streamlit as st
from data.data_generator import generate_synthetic_data
from logic.architecture_generator import generate_random_architecture
from logic.trainer import train_model
from logic.selector import select_best_model

def run_app():
    st.title("Neural Architecture Search Demo")
    st.markdown("🧠 Auto-generating and testing small neural networks (NAS) — 100% AI logic, no APIs or static data.")

    # Sidebar Inputs
    st.sidebar.header("Dataset Configuration")
    n_samples = st.sidebar.slider("Number of Samples", 200, 2000, 1000, step=100)
    n_features = st.sidebar.slider("Number of Features", 5, 50, 10)
    n_classes = st.sidebar.slider("Number of Classes", 2, 5, 2)

    st.sidebar.header("Search Configuration")
    n_models = st.sidebar.slider("Number of Architectures", 1, 10, 3)
    epochs = st.sidebar.slider("Training Epochs", 5, 50, 10)

    if st.button("Start NAS Process"):
        st.info("🔄 Generating data...")
        X_train, X_val, y_train, y_val = generate_synthetic_data(n_samples, n_features, n_classes)

        results = []
        for i in range(n_models):
            st.write(f"🔧 Generating Model {i+1}")
            model = generate_random_architecture(n_features, n_classes)
            result = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs)
            result["model_index"] = i + 1
            results.append(result)
            st.success(f"Model {i+1} - Val Accuracy: {result['val_accuracy']:.4f}")

        best = select_best_model(results)
        st.subheader("✅ Best Model Selected")
        st.write(f"🏆 Model #{best['model_index']} with Val Accuracy: `{best['val_accuracy']:.4f}`")

        st.markdown("---")
        st.code(str(best["model"]), language='python')
