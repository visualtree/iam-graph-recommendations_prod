import streamlit as st
import joblib
import os

st.write("🧪 **Direct Artifact Loading Test**")

try:
    # Direct loading without prediction_core.py
    artifacts = {
        'candidate_model': joblib.load('artifacts/candidate_model.joblib'),
        'candidate_features': joblib.load('artifacts/candidate_model_features.joblib'),
        'reranker_model': joblib.load('artifacts/reranker_model.joblib'),
        'reranker_features': joblib.load('artifacts/reranker_model_features.joblib'),
        'embeddings_df': joblib.load('artifacts/embeddings.pkl'),
        'users': joblib.load('artifacts/users.pkl')
    }
    st.success("✅ **All artifacts loaded successfully with direct joblib.load!**")
    st.write(f"Embeddings shape: {artifacts['embeddings_df'].shape}")
    st.write(f"Users shape: {artifacts['users'].shape}")
    
except Exception as e:
    st.error(f"❌ **Direct loading failed: {e}**")
    import traceback
    st.code(traceback.format_exc())