#!/usr/bin/env python3
"""
Main Streamlit Application for IAM Access Prediction Demo
Clean, modular architecture with separated concerns
"""

import sys
import importlib

# NUCLEAR OPTION: Remove and reload prediction_core (ONLY ADDITION TO YOUR ORIGINAL)
modules_to_remove = [key for key in sys.modules.keys() if 'prediction_core' in key]
for module in modules_to_remove:
    del sys.modules[module]

import streamlit as st
import time
import os

# Force fresh import of prediction_core
import ml_pipeline.prediction_core
importlib.reload(ml_pipeline.prediction_core)

# Add project paths
sys.path.append('ml_pipeline')
sys.path.append('streamlit_modules')

# Import modular components (YOUR ORIGINAL IMPORTS)
from streamlit_modules.config import setup_page_config, load_custom_css
from streamlit_modules.data_loader import load_models_and_data
from streamlit_modules.ui_components import (
    display_header, 
    display_executive_summary, 
    display_user_profile,
    create_sidebar_controls
)
from streamlit_modules.prediction_engine import run_prediction_pipeline
from streamlit_modules.results_display import display_prediction_results
from streamlit_modules.analysis_modules import (
    display_technical_deep_dive,
    create_model_explainability_showcase, 
    create_comparison_analysis
)
from streamlit_modules.session_management import initialize_session_state

def main():
    """Main application entry point"""
    
    # Setup page configuration and styling
    setup_page_config()
    load_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Display header and executive summary
    display_header()
    display_executive_summary()
    
    # Load models and data
    if not st.session_state.models_loaded:
        with st.spinner("🔄 Initializing AI models and graph database..."):
            models_data = load_models_and_data()
            st.session_state.models_data = models_data
            st.session_state.models_loaded = True
        st.success("✅ AI models loaded successfully!")
    
    # Create sidebar controls and get user selection
    user_selection = create_sidebar_controls()
    
    # Main content area
    if user_selection['selected_user_id']:
        # Display user profile
        display_user_profile(
            user_selection['selected_user_data'], 
            st.session_state.models_data['graph_dfs']
        )
        
        st.markdown("---")
        
        # Handle prediction generation
        if user_selection['generate_predictions']:
            with st.spinner("🤖 Running AI prediction pipeline..."):
                results = run_prediction_pipeline(
                    user_selection['selected_user_id'],
                    st.session_state.models_data,
                    user_selection['top_n'],
                    user_selection['initial_candidates']
                )
                
                if results:
                    st.session_state.current_predictions = results
                    st.session_state.demo_mode = user_selection['demo_mode']
                    st.success("✅ Predictions generated successfully!")
                    time.sleep(1)
                    st.rerun()
        
        # Display results if available
        if st.session_state.current_predictions:
            display_prediction_results(
                st.session_state.current_predictions,
                st.session_state.get('demo_mode', 'Standard Demo'),
                user_selection['selected_user_id']
            )
        
        else:
            # No predictions yet - show getting started
            st.markdown("### 🎯 Get Started")
            st.info("👈 Select a user and click 'Generate Predictions' to see AI-powered access recommendations")
            
            # Show data overview
            from streamlit_modules.data_overview import display_data_overview
            display_data_overview(st.session_state.models_data['graph_dfs'])
    
    else:
        st.info("Please select a user from the sidebar to begin")

# Application debugging panel (YOUR ORIGINAL DEBUG PANEL)
def create_debug_panel():
    """Create debug panel for monitoring"""
    
    with st.expander("🔧 Debug & Monitoring Panel"):
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("System Status")
            
            # Model status
            models_loaded = st.session_state.get('models_loaded', False)
            st.markdown(f"**Models:** {'🟢 Loaded' if models_loaded else '🔴 Not Loaded'}")
            
            # Memory usage
            try:
                import psutil
                memory_usage = psutil.virtual_memory().percent
                st.markdown(f"**Memory Usage:** {memory_usage:.1f}%")
            except ImportError:
                st.markdown("**Memory Usage:** N/A (psutil not available)")
            
            # Data status
            if models_loaded:
                graph_dfs = st.session_state.models_data['graph_dfs']
                st.markdown(f"**Users Loaded:** {len(graph_dfs['users'])}")
                st.markdown(f"**Entitlements:** {len(graph_dfs['entitlements'])}")
        
        with col2:
            st.subheader("Performance Metrics")
            
            # Prediction history
            if st.session_state.current_predictions:
                st.markdown("**Last Prediction:**")
                st.markdown(f"- User ID: {st.session_state.get('selected_user', 'N/A')}")
                st.markdown(f"- Recommendations: {len(st.session_state.current_predictions['predictions'])}")
                st.markdown(f"- Avg Confidence: {st.session_state.current_predictions['predictions']['FinalScore'].mean():.1%}")
        
        with col3:
            st.subheader("Configuration")
            
            from ml_pipeline import config
            st.markdown(f"**Artifact Directory:** `{config.ARTIFACT_DIR}`")
            st.markdown(f"**Embedding Dimension:** {config.EMBEDDING_DIMENSION}")
            st.markdown(f"**Random State:** {config.RANDOM_STATE}")

if __name__ == "__main__":
    # Add debug panel for development (YOUR ORIGINAL DEBUG OPTION)
    if st.checkbox("🔧 Show Debug Panel", value=False):
        create_debug_panel()
        st.markdown("---")
    
    # Run main application
    main()
    
    # Footer (YOUR ORIGINAL FOOTER)
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <strong>Falaina IAM Access Prediction Engine</strong><br>
        Powered by XGBoost + Graph Embeddings + Node2Vec | 
        Built with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)

# #!/usr/bin/env python3
# """
# Main Streamlit Application for IAM Access Prediction Demo
# Clean, modular architecture with separated concerns
# """

# import streamlit as st
# import time
# import sys
# import os

# # Add project paths
# sys.path.append('ml_pipeline')
# sys.path.append('streamlit_modules')

# # Import modular components
# from streamlit_modules.config import setup_page_config, load_custom_css
# from streamlit_modules.data_loader import load_models_and_data
# from streamlit_modules.ui_components import (
    # display_header, 
    # display_executive_summary, 
    # display_user_profile,
    # create_sidebar_controls
# )
# from streamlit_modules.prediction_engine import run_prediction_pipeline
# from streamlit_modules.results_display import display_prediction_results
# from streamlit_modules.analysis_modules import (
    # display_technical_deep_dive,
    # create_model_explainability_showcase, 
    # create_comparison_analysis
# )
# from streamlit_modules.session_management import initialize_session_state

# def main():
    # """Main application entry point"""
    
    # # Setup page configuration and styling
    # setup_page_config()
    # load_custom_css()
    
    # # Initialize session state
    # initialize_session_state()
    
    # # Display header and executive summary
    # display_header()
    # display_executive_summary()
    
    # # Load models and data
    # if not st.session_state.models_loaded:
        # with st.spinner("🔄 Initializing AI models and graph database..."):
            # models_data = load_models_and_data()
            # st.session_state.models_data = models_data
            # st.session_state.models_loaded = True
        # st.success("✅ AI models loaded successfully!")
    
    # # Create sidebar controls and get user selection
    # user_selection = create_sidebar_controls()
    
    # # Main content area
    # if user_selection['selected_user_id']:
        # # Display user profile
        # display_user_profile(
            # user_selection['selected_user_data'], 
            # st.session_state.models_data['graph_dfs']
        # )
        
        # st.markdown("---")
        
        # # Handle prediction generation
        # if user_selection['generate_predictions']:
            # with st.spinner("🤖 Running AI prediction pipeline..."):
                # results = run_prediction_pipeline(
                    # user_selection['selected_user_id'],
                    # st.session_state.models_data,
                    # user_selection['top_n'],
                    # user_selection['initial_candidates']
                # )
                
                # if results:
                    # st.session_state.current_predictions = results
                    # st.session_state.demo_mode = user_selection['demo_mode']
                    # st.success("✅ Predictions generated successfully!")
                    # time.sleep(1)
                    # st.rerun()
        
        # # Display results if available
        # if st.session_state.current_predictions:
            # display_prediction_results(
                # st.session_state.current_predictions,
                # st.session_state.get('demo_mode', 'Standard Demo'),
                # user_selection['selected_user_id']
            # )
        
        # else:
            # # No predictions yet - show getting started
            # st.markdown("### 🎯 Get Started")
            # st.info("👈 Select a user and click 'Generate Predictions' to see AI-powered access recommendations")
            
            # # Show data overview
            # from streamlit_modules.data_overview import display_data_overview
            # display_data_overview(st.session_state.models_data['graph_dfs'])
    
    # else:
        # st.info("Please select a user from the sidebar to begin")

# # Application debugging panel
# def create_debug_panel():
    # """Create debug panel for monitoring"""
    
    # with st.expander("🔧 Debug & Monitoring Panel"):
        
        # col1, col2, col3 = st.columns(3)
        
        # with col1:
            # st.subheader("System Status")
            
            # # Model status
            # models_loaded = st.session_state.get('models_loaded', False)
            # st.markdown(f"**Models:** {'🟢 Loaded' if models_loaded else '🔴 Not Loaded'}")
            
            # # Memory usage
            # try:
                # import psutil
                # memory_usage = psutil.virtual_memory().percent
                # st.markdown(f"**Memory Usage:** {memory_usage:.1f}%")
            # except ImportError:
                # st.markdown("**Memory Usage:** N/A (psutil not available)")
            
            # # Data status
            # if models_loaded:
                # graph_dfs = st.session_state.models_data['graph_dfs']
                # st.markdown(f"**Users Loaded:** {len(graph_dfs['users'])}")
                # st.markdown(f"**Entitlements:** {len(graph_dfs['entitlements'])}")
        
        # with col2:
            # st.subheader("Performance Metrics")
            
            # # Prediction history
            # if st.session_state.current_predictions:
                # st.markdown("**Last Prediction:**")
                # st.markdown(f"- User ID: {st.session_state.get('selected_user', 'N/A')}")
                # st.markdown(f"- Recommendations: {len(st.session_state.current_predictions['predictions'])}")
                # st.markdown(f"- Avg Confidence: {st.session_state.current_predictions['predictions']['FinalScore'].mean():.1%}")
        
        # with col3:
            # st.subheader("Configuration")
            
            # from ml_pipeline import config
            # st.markdown(f"**Artifact Directory:** `{config.ARTIFACT_DIR}`")
            # st.markdown(f"**Embedding Dimension:** {config.EMBEDDING_DIMENSION}")
            # st.markdown(f"**Random State:** {config.RANDOM_STATE}")

# if __name__ == "__main__":
    # # Add debug panel for development
    # if st.checkbox("🔧 Show Debug Panel", value=False):
        # create_debug_panel()
        # st.markdown("---")
    
    # # Run main application
    # main()
    
    # # Footer
    # st.markdown("---")
    # st.markdown("""
    # <div style='text-align: center; color: #666; padding: 1rem;'>
        # <strong>Falaina IAM Access Prediction Engine</strong><br>
        # Powered by XGBoost + Graph Embeddings + Node2Vec | 
        # Built with ❤️ using Streamlit
    # </div>
    # """, unsafe_allow_html=True)
    
# #!/usr/bin/env python3
# """
# Main Streamlit Application for IAM Access Prediction Demo
# Clean, modular architecture with separated concerns
# """
# import sys
# import importlib

# # NUCLEAR OPTION: Remove and reload prediction_core
# modules_to_remove = [key for key in sys.modules.keys() if 'prediction_core' in key]
# for module in modules_to_remove:
    # del sys.modules[module]

# import streamlit as st
# import time
# import os
# import pandas as pd
# import numpy as np

# # Force fresh import
# import ml_pipeline.prediction_core
# importlib.reload(ml_pipeline.prediction_core)

# st.write("🔄 **Forced module reload completed**")

# # Test the loading immediately
# try:
    # from ml_pipeline.prediction_core import PredictionArtifacts, run_prediction_pipeline, calculate_peer_insights
    # st.write("✅ **Import successful after reload**")
    
    # artifacts = PredictionArtifacts.get_artifacts()
    # st.write("✅ **Loading successful after reload!**")
    
    # # Store artifacts in session state
    # if 'artifacts' not in st.session_state:
        # st.session_state.artifacts = artifacts
        # st.session_state.models_loaded = True
    
    # st.write("🎉 **Proceeding with normal app...**")
    
# except Exception as e:
    # st.error(f"❌ **Still failing after reload: {e}**")
    # import traceback
    # st.code(traceback.format_exc())
    # st.stop()  # Stop here if still failing

# # =============================================================================
# # STREAMLIT UI MODULES (Import your existing modules)
# # =============================================================================

# try:
    # # Try to import your existing Streamlit modules
    # from streamlit_modules import (
        # session_management,
        # ui_components, 
        # data_overview,
        # prediction_engine,
        # explainability,
        # results_display,
        # metrics_calculator,
        # analysis_modules
    # )
    # st.write("✅ **All Streamlit modules imported successfully**")
# except ImportError as e:
    # st.warning(f"⚠️ Some Streamlit modules not found: {e}")
    # st.write("**Continuing with basic interface...**")
    
    # # Create placeholder modules if they don't exist
    # class PlaceholderModule:
        # @staticmethod
        # def create_any_function(*args, **kwargs):
            # st.info("This component is not yet implemented")
    
    # session_management = PlaceholderModule()
    # ui_components = PlaceholderModule()
    # data_overview = PlaceholderModule()
    # prediction_engine = PlaceholderModule()
    # explainability = PlaceholderModule()
    # results_display = PlaceholderModule()
    # metrics_calculator = PlaceholderModule()
    # analysis_modules = PlaceholderModule()

# # =============================================================================
# # PAGE CONFIGURATION
# # =============================================================================

# st.set_page_config(
    # page_title="IAM Access Prediction Engine",
    # page_icon="🛡️",
    # layout="wide",
    # initial_sidebar_state="expanded"
# )

# # =============================================================================
# # SESSION STATE MANAGEMENT
# # =============================================================================

# def initialize_session_state():
    # """Initialize all session state variables"""
    # if 'page' not in st.session_state:
        # st.session_state.page = 'Home'
    
    # if 'prediction_results' not in st.session_state:
        # st.session_state.prediction_results = None
    
    # if 'selected_user' not in st.session_state:
        # st.session_state.selected_user = 54
    
    # if 'models_loaded' not in st.session_state:
        # st.session_state.models_loaded = True  # We loaded them above

# initialize_session_state()

# # =============================================================================
# # SIDEBAR NAVIGATION
# # =============================================================================

# def create_sidebar():
    # """Create the main navigation sidebar"""
    # with st.sidebar:
        # st.title("🛡️ IAM Engine")
        
        # # Navigation menu
        # pages = [
            # "🏠 Home",
            # "🎯 Predictions", 
            # "📊 Data Overview",
            # "📈 Analytics",
            # "🔍 Explainability",
            # "⚙️ Configuration"
        # ]
        
        # selected_page = st.selectbox("Navigate to:", pages, key="nav_select")
        # st.session_state.page = selected_page.split(" ", 1)[1]  # Remove emoji
        
        # st.markdown("---")
        
        # # System status
        # st.subheader("📋 System Status")
        
        # # Model status
        # if st.session_state.get('models_loaded', False):
            # st.success("✅ Models Loaded")
        # else:
            # st.error("❌ Models Not Loaded")
        
        # # Data status
        # if st.session_state.get('artifacts'):
            # users_count = len(st.session_state.artifacts['graph_dfs']['users'])
            # entitlements_count = len(st.session_state.artifacts['graph_dfs']['entitlements'])
            # st.info(f"👥 {users_count:,} Users")
            # st.info(f"🔐 {entitlements_count:,} Entitlements")
        
        # st.markdown("---")
        
        # # Quick actions
        # st.subheader("⚡ Quick Actions")
        # if st.button("🔄 Refresh Models"):
            # st.rerun()
        
        # if st.button("📥 Export Results"):
            # if st.session_state.prediction_results:
                # st.download_button(
                    # "💾 Download CSV",
                    # st.session_state.prediction_results.to_csv(index=False),
                    # "predictions.csv",
                    # "text/csv"
                # )

# # =============================================================================
# # DEBUG PANEL
# # =============================================================================

# def create_debug_panel():
    # """Create comprehensive debug and monitoring panel"""
    
    # with st.expander("🔍 Debug & Monitoring Panel"):
        
        # # Tab layout for different debug views
        # tab1, tab2, tab3, tab4 = st.tabs(["🔧 System Status", "📊 Model Performance", "⚙️ Configuration", "📝 Logs"])
        
        # with tab1:
            # st.subheader("System Status")
            
            # # System health checks
            # col1, col2, col3 = st.columns(3)
            
            # with col1:
                # # Database connectivity
                # db_status = True  # You can implement actual check
                # st.metric(
                    # "Database Status",
                    # "Connected" if db_status else "Disconnected",
                    # delta="Healthy" if db_status else "Error"
                # )
            
            # with col2:
                # # Model loading status
                # models_loaded = st.session_state.get('models_loaded', False)
                # st.metric(
                    # "Models Status",
                    # "Loaded" if models_loaded else "Not Loaded",
                    # delta="Ready" if models_loaded else "Loading Required"
                # )
            
            # with col3:
                # # Memory usage
                # try:
                    # import psutil
                    # memory_usage = psutil.virtual_memory().percent
                    # st.metric(
                        # "Memory Usage",
                        # f"{memory_usage:.1f}%",
                        # delta="Normal" if memory_usage < 80 else "High"
                    # )
                # except ImportError:
                    # st.metric("Memory Usage", "N/A", delta="psutil not available")
        
        # with tab2:
            # st.subheader("Model Performance")
            
            # col1, col2, col3, col4 = st.columns(4)
            
            # with col1:
                # st.metric("Candidate AUC", "0.936", delta="Excellent")
            
            # with col2:
                # st.metric("Reranker AUC", "0.999", delta="Outstanding")
            
            # with col3:
                # st.metric("Avg Prediction Time", "2.1s", delta="-0.3s")
            
            # with col4:
                # st.metric("Cache Hit Rate", "87%", delta="5%")
        
        # with tab3:
            # st.subheader("Configuration")
            
            # col1, col2, col3 = st.columns(3)
            
            # with col1:
                # st.write("**Artifact Directory:**")
                # st.code("artifacts")
            
            # with col2:
                # st.write("**Embedding Dimension:**")
                # st.code("64")
            
            # with col3:
                # st.write("**Random State:**")
                # st.code("42")
        
        # with tab4:
            # st.subheader("Application Logs")
            
            # # Recent logs
            # if 'app_logs' not in st.session_state:
                # st.session_state.app_logs = [
                    # f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Application started",
                    # f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Models loaded successfully",
                # ]
            
            # for log in st.session_state.app_logs[-10:]:  # Show last 10 logs
                # st.text(log)

# # =============================================================================
# # PAGE FUNCTIONS
# # =============================================================================

# def show_home_page():
    # """Display the home/landing page"""
    # st.title("🛡️ IAM Access Prediction Engine")
    # st.markdown("**Intelligent Access Rights Prediction with XGBoost + Graph Embeddings + Node2Vec**")
    
    # # Overview metrics
    # col1, col2, col3, col4 = st.columns(4)
    
    # if st.session_state.get('artifacts'):
        # users_count = len(st.session_state.artifacts['graph_dfs']['users'])
        # entitlements_count = len(st.session_state.artifacts['graph_dfs']['entitlements'])
        # access_relationships = len(st.session_state.artifacts['graph_dfs']['entrecon'])
        
        # with col1:
            # st.metric("👥 Total Users", f"{users_count:,}")
        # with col2:
            # st.metric("🔐 Total Entitlements", f"{entitlements_count:,}")
        # with col3:
            # st.metric("🔗 Access Relationships", f"{access_relationships:,}")
        # with col4:
            # st.metric("🎯 Model Accuracy", "93.6%")
    
    # # Quick start guide
    # st.subheader("🚀 Quick Start Guide")
    
    # st.markdown("""
    # 1. **Navigate to Predictions** 📍 Use the sidebar to go to the Predictions page
    # 2. **Enter User ID** 🔢 Input the user ID you want to generate predictions for
    # 3. **Set Parameters** ⚙️ Choose number of recommendations and other settings
    # 4. **Generate Predictions** 🎯 Click the button to run the ML pipeline
    # 5. **Analyze Results** 📊 Review recommendations and explanations
    # """)
    
    # # Recent activity
    # if st.session_state.prediction_results is not None:
        # st.subheader("📈 Recent Predictions")
        # st.write(f"Last prediction generated for User {st.session_state.selected_user}")
        
        # # Show summary of last results
        # results_summary = st.session_state.prediction_results.head(3)
        # st.dataframe(results_summary[['EntitlementId', 'FinalScore']], use_container_width=True)

# def show_predictions_page():
    # """Display the main predictions interface"""
    # st.header("🎯 Generate Access Predictions")
    
    # # User input section
    # col1, col2 = st.columns([2, 1])
    
    # with col1:
        # user_id = st.number_input(
            # "Enter User ID for Prediction",
            # min_value=1,
            # value=st.session_state.selected_user,
            # help="Enter a valid user ID to generate access recommendations"
        # )
        # st.session_state.selected_user = user_id
    
    # with col2:
        # top_n = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)
    
    # # Advanced options
    # with st.expander("🔧 Advanced Options"):
        # initial_candidates = st.slider("Initial Candidates", min_value=50, max_value=500, value=100)
        # show_peer_analysis = st.checkbox("Show Peer Analysis", value=True)
        # generate_explanations = st.checkbox("Generate SHAP Explanations", value=False)
    
    # # Generate predictions button
    # if st.button("🚀 Generate Predictions", type="primary"):
        # if user_id:
            # try:
                # # Create progress tracking
                # progress_bar = st.progress(0)
                # status_text = st.empty()
                
                # def update_progress(step, total, message=""):
                    # progress = step / total
                    # progress_bar.progress(progress)
                    # status_text.text(f"Step {step}/{total}: {message}")
                
                # # Run prediction pipeline
                # with st.spinner("Generating predictions..."):
                    # results = run_prediction_pipeline(
                        # user_id=user_id,
                        # top_n=top_n,
                        # initial_candidates=initial_candidates,
                        # progress_callback=update_progress
                    # )
                
                # # Clear progress indicators
                # progress_bar.empty()
                # status_text.empty()
                
                # if results and not results['predictions'].empty:
                    # st.success(f"✅ Generated {len(results['predictions'])} recommendations for User {user_id}")
                    
                    # # Store results in session state
                    # st.session_state.prediction_results = results['predictions'].copy()
                    
                    # # Display results
                    # show_prediction_results(results, show_peer_analysis, generate_explanations)
                    
                    # # Add to logs
                    # log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Predictions generated for User {user_id}"
                    # if 'app_logs' not in st.session_state:
                        # st.session_state.app_logs = []
                    # st.session_state.app_logs.append(log_entry)
                    
                # else:
                    # st.warning(f"⚠️ No recommendations found for User {user_id}")
                    
            # except Exception as e:
                # st.error(f"❌ Error generating predictions: {str(e)}")
                # with st.expander("🔍 Error Details"):
                    # import traceback
                    # st.code(traceback.format_exc())
        # else:
            # st.warning("⚠️ Please enter a valid User ID")

# def show_prediction_results(results, show_peer_analysis=True, generate_explanations=False):
    # """Display prediction results with analysis"""
    # predictions_df = results['predictions'].copy()
    
    # # Main results table
    # st.subheader("📋 Recommended Access Rights")
    
    # # Format display
    # display_df = predictions_df[['EntitlementId', 'FinalScore', 'CandidateScore']].copy()
    # if 'OriginalEntitlementId' in predictions_df.columns:
        # display_df['OriginalEntitlementId'] = predictions_df['OriginalEntitlementId']
    
    # display_df['FinalScore'] = display_df['FinalScore'].apply(lambda x: f"{x:.4f}")
    # display_df['CandidateScore'] = display_df['CandidateScore'].apply(lambda x: f"{x:.4f}")
    
    # # Rename columns for display
    # column_mapping = {
        # 'EntitlementId': 'Entitlement ID',
        # 'FinalScore': 'Final Score', 
        # 'CandidateScore': 'Initial Score'
    # }
    # if 'OriginalEntitlementId' in display_df.columns:
        # column_mapping['OriginalEntitlementId'] = 'Original ID'
    
    # display_df = display_df.rename(columns=column_mapping)
    # st.dataframe(display_df, use_container_width=True)
    
    # # Insights section
    # col1, col2, col3 = st.columns(3)
    
    # with col1:
        # st.metric("Total Candidates", f"{results['total_candidates']:,}")
    # with col2:
        # st.metric("Stage 1 Filtered", f"{results['stage1_count']:,}")
    # with col3:
        # avg_score = predictions_df['FinalScore'].mean()
        # st.metric("Average Confidence", f"{avg_score:.3f}")
    
    # # Peer analysis
    # if show_peer_analysis and len(predictions_df) > 0:
        # st.subheader("👥 Peer Adoption Analysis")
        
        # top_recommendation = predictions_df.iloc[0]
        # peer_insights = calculate_peer_insights(
            # results['user_id'], 
            # top_recommendation['EntitlementId']
        # )
        
        # if peer_insights:
            # col1, col2, col3, col4 = st.columns(4)
            
            # with col1:
                # rate = peer_insights['close_peers']['adoption_rate']
                # total = peer_insights['close_peers']['total']
                # st.metric("🤝 Close Peers", f"{rate:.1%}", f"{total} peers")
            
            # with col2:
                # rate = peer_insights['direct_team']['adoption_rate']
                # total = peer_insights['direct_team']['total']
                # st.metric("👥 Direct Team", f"{rate:.1%}", f"{total} members")
            
            # with col3:
                # rate = peer_insights['role_peers']['adoption_rate']
                # total = peer_insights['role_peers']['total']
                # st.metric("🏢 Role Peers", f"{rate:.1%}", f"{total} peers")
            
            # with col4:
                # rate = peer_insights['dept_peers']['adoption_rate'] 
                # total = peer_insights['dept_peers']['total']
                # st.metric("🏛️ Dept Peers", f"{rate:.1%}", f"{total} peers")

# def show_data_overview_page():
    # """Display data overview and statistics"""
    # st.header("📊 Data Overview")
    
    # if st.session_state.get('artifacts'):
        # artifacts = st.session_state.artifacts
        
        # # Dataset statistics
        # col1, col2, col3 = st.columns(3)
        
        # with col1:
            # st.subheader("👥 Users")
            # users_df = artifacts['graph_dfs']['users']
            # st.metric("Total Users", len(users_df))
            # st.metric("Active Users", len(users_df[users_df.get('IsActive', True)]))
        
        # with col2:
            # st.subheader("🔐 Entitlements")
            # entitlements_df = artifacts['graph_dfs']['entitlements']
            # st.metric("Total Entitlements", len(entitlements_df))
            # # Show unique systems if available
            # if 'id' in entitlements_df.columns:
                # systems = entitlements_df['id'].astype(str).str.split('_').str[0].nunique()
                # st.metric("Unique Systems", systems)
        
        # with col3:
            # st.subheader("🔗 Access Relationships")
            # entrecon_df = artifacts['graph_dfs']['entrecon']
            # st.metric("Total Relationships", len(entrecon_df))
            # st.metric("Unique User-Ent Pairs", len(entrecon_df.drop_duplicates(['UserId', 'EntitlementId'])))
        
        # # Data quality checks
        # st.subheader("🔍 Data Quality")
        
        # # Check for missing values, duplicates, etc.
        # quality_checks = []
        
        # # Users data quality
        # user_nulls = users_df.isnull().sum().sum()
        # quality_checks.append(("Users - Missing Values", user_nulls, "❌" if user_nulls > 0 else "✅"))
        
        # # Entitlements data quality  
        # ent_nulls = entitlements_df.isnull().sum().sum()
        # quality_checks.append(("Entitlements - Missing Values", ent_nulls, "❌" if ent_nulls > 0 else "✅"))
        
        # # Access relationships quality
        # entrecon_dupes = len(entrecon_df) - len(entrecon_df.drop_duplicates())
        # quality_checks.append(("Access - Duplicate Relationships", entrecon_dupes, "❌" if entrecon_dupes > 0 else "✅"))
        
        # # Display quality checks
        # for check_name, value, status in quality_checks:
            # col1, col2, col3 = st.columns([3, 1, 1])
            # with col1:
                # st.write(check_name)
            # with col2:
                # st.write(value)
            # with col3:
                # st.write(status)

# def show_analytics_page():
    # """Display analytics and insights"""
    # st.header("📈 Analytics & Insights")
    # st.info("📊 Advanced analytics features coming soon...")
    
    # if st.session_state.prediction_results is not None:
        # st.subheader("📊 Recent Prediction Analysis")
        
        # results_df = st.session_state.prediction_results
        
        # # Score distribution
        # st.write("**Score Distribution**")
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.hist(results_df['FinalScore'], bins=10, alpha=0.7)
        # ax.set_xlabel('Final Score')
        # ax.set_ylabel('Frequency')
        # ax.set_title('Distribution of Prediction Scores')
        # st.pyplot(fig)

# def show_explainability_page():
    # """Display model explainability features"""
    # st.header("🔍 Model Explainability")
    # st.info("🔍 SHAP explanations and feature importance analysis coming soon...")

# def show_configuration_page():
    # """Display configuration and settings"""
    # st.header("⚙️ Configuration")
    
    # st.subheader("🔧 Model Settings")
    
    # col1, col2 = st.columns(2)
    
    # with col1:
        # st.write("**Current Configuration:**")
        # st.code("""
        # Artifact Directory: artifacts/
        # Embedding Dimension: 64
        # Random State: 42
        # Initial Candidates: 100
        # """)
    
    # with col2:
        # st.write("**Model Performance:**")
        # st.code("""
        # Candidate Model AUC: 0.936
        # Reranker Model AUC: 0.999
        # Training Time: 25 minutes
        # """)

# # =============================================================================
# # MAIN APPLICATION LOGIC
# # =============================================================================

# def main():
    # """Main application logic"""
    
    # # Create sidebar
    # create_sidebar()
    
    # # Show debug panel
    # create_debug_panel()
    
    # # Route to appropriate page
    # page = st.session_state.get('page', 'Home')
    
    # if page == 'Home':
        # show_home_page()
    # elif page == 'Predictions':
        # show_predictions_page()
    # elif page == 'Data Overview':
        # show_data_overview_page()
    # elif page == 'Analytics':
        # show_analytics_page()
    # elif page == 'Explainability':
        # show_explainability_page()
    # elif page == 'Configuration':
        # show_configuration_page()
    # else:
        # show_home_page()
    
    # # Footer
    # st.markdown("---")
    # st.markdown("**IAM Access Prediction Engine** | Built with Streamlit + XGBoost + Neo4j + Node2Vec")

# # Run the main application
# if __name__ == "__main__":
    # main()
# else:
    # main()  # Streamlit runs the entire script
    
    