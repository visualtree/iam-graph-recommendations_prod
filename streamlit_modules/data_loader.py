"""
Data loading and model management for Streamlit application
"""

import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path

# Import ML pipeline modules
from ml_pipeline import config

@st.cache_resource(show_spinner="🧠 Loading AI models and graph data...")
def load_models_and_data():
    """Load all ML models and supporting data"""
    try:
        # Check if artifacts exist
        artifact_files = [
            'candidate_model.joblib',
            'reranker_model.joblib', 
            'candidate_model_features.joblib',
            'reranker_model_features.joblib',
            'embeddings.pkl',
            'users.pkl',
            'entitlements.pkl',
            'entrecon.pkl'
        ]
        
        missing_files = []
        for file in artifact_files:
            if not os.path.exists(os.path.join(config.ARTIFACT_DIR, file)):
                missing_files.append(file)
        
        if missing_files:
            st.error(f"❌ Missing model artifacts: {', '.join(missing_files)}")
            st.error("Please run the training pipeline first: `python -m ml_pipeline.train`")
            st.stop()
        
        # Load models
        cand_model = joblib.load(os.path.join(config.ARTIFACT_DIR, 'candidate_model.joblib'))
        rerank_model = joblib.load(os.path.join(config.ARTIFACT_DIR, 'reranker_model.joblib'))
        cand_features = joblib.load(os.path.join(config.ARTIFACT_DIR, 'candidate_model_features.joblib'))
        rerank_features = joblib.load(os.path.join(config.ARTIFACT_DIR, 'reranker_model_features.joblib'))
        
        # Load supporting data
        embeddings_df = joblib.load(os.path.join(config.ARTIFACT_DIR, 'embeddings.pkl'))
        
        # Load graph data
        graph_files = {
            'users': 'users.pkl',
            'entitlements': 'entitlements.pkl',
            'entrecon': 'entrecon.pkl',
            'orgs': 'orgs.pkl',
            'endpoints': 'endpoints.pkl',
            'designations': 'designations.pkl'
        }
        
        graph_dfs = {}
        for name, filename in graph_files.items():
            file_path = os.path.join(config.ARTIFACT_DIR, filename)
            if os.path.exists(file_path):
                graph_dfs[name] = joblib.load(file_path)
            else:
                st.warning(f"⚠️ Optional file {filename} not found, continuing without it")
        
        return {
            'candidate_model': cand_model,
            'reranker_model': rerank_model,
            'candidate_features': cand_features,
            'reranker_features': rerank_features,
            'embeddings_df': embeddings_df,
            'graph_dfs': graph_dfs
        }
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

def validate_data_integrity(models_data):
    """Validate loaded data integrity"""
    required_keys = ['candidate_model', 'reranker_model', 'embeddings_df', 'graph_dfs']
    
    for key in required_keys:
        if key not in models_data:
            raise ValueError(f"Missing required data: {key}")
    
    # Validate graph data
    required_graph_data = ['users', 'entitlements', 'entrecon']
    for graph_key in required_graph_data:
        if graph_key not in models_data['graph_dfs']:
            raise ValueError(f"Missing required graph data: {graph_key}")
        
        if models_data['graph_dfs'][graph_key].empty:
            raise ValueError(f"Empty graph data: {graph_key}")
    
    return True

def get_data_statistics(graph_dfs):
    """Get basic statistics about loaded data"""
    stats = {}
    
    for name, df in graph_dfs.items():
        if isinstance(df, pd.DataFrame):
            stats[name] = {
                'count': len(df),
                'columns': list(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
    
    return stats