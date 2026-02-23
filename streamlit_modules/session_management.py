"""
Session state management for Streamlit application
"""

import streamlit as st

def initialize_session_state():
    """Initialize all session state variables"""
    
    # Model loading state
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    
    # Prediction state
    if 'current_predictions' not in st.session_state:
        st.session_state.current_predictions = None
    
    # User selection state
    if 'selected_user' not in st.session_state:
        st.session_state.selected_user = None
    
    # Demo mode state
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = 'Standard Demo'
    
    # Comparison data state
    if 'comparison_data' not in st.session_state:
        st.session_state.comparison_data = None
    
    # Models and data state
    if 'models_data' not in st.session_state:
        st.session_state.models_data = None

def clear_predictions():
    """Clear current predictions from session state"""
    st.session_state.current_predictions = None

def update_demo_mode(mode):
    """Update demo mode in session state"""
    st.session_state.demo_mode = mode

def get_session_summary():
    """Get summary of current session state"""
    return {
        'models_loaded': st.session_state.get('models_loaded', False),
        'has_predictions': st.session_state.current_predictions is not None,
        'selected_user': st.session_state.get('selected_user'),
        'demo_mode': st.session_state.get('demo_mode', 'Standard Demo'),
        'prediction_count': len(st.session_state.current_predictions['predictions']) if st.session_state.current_predictions else 0
    }