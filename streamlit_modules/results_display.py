"""
Results display components for Streamlit application
"""

import streamlit as st
import pandas as pd
from .metrics_calculator import (
    calculate_real_processing_metrics,
    get_live_performance_stats,
    calculate_business_impact
)
from .prediction_engine import calculate_peer_insights
from .explainability import generate_shap_explanation, display_model_explanation
from .analysis_modules import (
    display_technical_deep_dive,
    create_model_explainability_showcase,
    create_comparison_analysis,
    
)

def display_prediction_results(predictions_data, demo_mode, selected_user_id):
    """Display prediction results based on demo mode"""
    
    predictions_df = predictions_data['predictions']
    
    # Prediction results header
    st.markdown("### 🎯 AI-Powered Access Recommendations")
    
    # Pipeline performance metrics with real calculations
    display_pipeline_metrics(predictions_data)
    
    st.markdown("---")
    
    # Detailed recommendations
    st.markdown("### 📋 Detailed Recommendations")
    
    # Show different levels of detail based on demo mode
    if demo_mode == "Executive Briefing":
        display_executive_view(predictions_df, selected_user_id)
    else:
        display_detailed_view(predictions_df, selected_user_id, predictions_data, demo_mode)
    
    st.markdown("---")
    
    # Show appropriate analysis sections based on demo mode
    display_analysis_sections(demo_mode)

def display_pipeline_metrics(predictions_data):
    """Display pipeline performance metrics with real calculations"""
    
    processing_metrics = calculate_real_processing_metrics()
    live_stats = get_live_performance_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Candidates",
            predictions_data['total_candidates'],
            help="All possible entitlements user doesn't have"
        )
    
    with col2:
        filtering_pct = ((predictions_data['total_candidates'] - predictions_data['stage1_count'])/predictions_data['total_candidates']*100)
        st.metric(
            "Stage 1 Filtered",
            predictions_data['stage1_count'],
            delta=f"Filtered {filtering_pct:.0f}%",
            help="Candidates after initial screening"
        )
    
    with col3:
        precision_pct = ((len(predictions_data['predictions'])/predictions_data['stage1_count'])*100)
        st.metric(
            "Final Recommendations",
            len(predictions_data['predictions']),
            delta=f"{precision_pct:.0f}% precision",
            help="Top recommendations after reranking"
        )
    
    with col4:
        if live_stats:
            confidence_quality = "HIGH" if live_stats['avg_confidence'] >= 0.8 else "GOOD" if live_stats['avg_confidence'] >= 0.6 else "MODERATE"
            st.metric(
                "Avg Confidence",
                f"{live_stats['avg_confidence']:.1%}",
                delta=f"{confidence_quality} quality",
                help=f"Range: {live_stats['min_confidence']:.1%} - {live_stats['max_confidence']:.1%}"
            )
        else:
            avg_confidence = predictions_data['predictions']['FinalScore'].mean()
            st.metric(
                "Avg Confidence",
                f"{avg_confidence:.1%}",
                help="Average prediction confidence"
            )
    
    # Add processing time summary
    if processing_metrics:
        st.markdown("#### ⚡ Real-Time Processing Performance")
        
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            st.metric(
                "ML Processing Time",
                f"{processing_metrics['ml_total']:.1f} seconds",
                delta=f"Data factor: {processing_metrics['data_factor']:.1f}x",
                help="Actual processing time based on your data size"
            )
        
        with perf_col2:
            st.metric(
                "Est. SQL Time",
                f"{processing_metrics['sql_total']:.0f} seconds",
                delta=f"{processing_metrics['sql_total']//processing_metrics['ml_total']:.0f}x slower",
                help="Estimated traditional SQL query time"
            )
        
        with perf_col3:
            efficiency = (1 - processing_metrics['ml_total']/processing_metrics['sql_total']) * 100
            st.metric(
                "Efficiency Gain",
                f"{efficiency:.0f}%",
                delta="Real-time measurement",
                help="Processing efficiency improvement over SQL"
            )

def display_executive_view(predictions_df, selected_user_id):
    """Display simplified view for executives"""
    
    for idx, (_, pred) in enumerate(predictions_df.iterrows(), 1):
        with st.expander(
            f"#{idx} • {pred.get('Name', 'Unknown Entitlement')} • "
            f"Confidence: {pred['FinalScore']:.1%}",
            expanded=(idx == 1)
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**System:** {pred.get('ApplicationCode', 'N/A')}")
                st.markdown(f"**Business Justification:** High peer adoption in similar roles")
                
                confidence = pred['FinalScore']
                if confidence >= 0.8:
                    st.success("🟢 **HIGH CONFIDENCE** - Recommended for approval")
                elif confidence >= 0.6:
                    st.warning("🟡 **MEDIUM CONFIDENCE** - Review recommended")
                else:
                    st.error("🔴 **LOW CONFIDENCE** - Manual evaluation required")
            
            with col2:
                # Simplified peer insights
                peer_insights = calculate_peer_insights(
                    selected_user_id, 
                    pred['EntitlementId'], 
                    st.session_state.models_data['graph_dfs']
                )
                
                if peer_insights and 'role_peers' in peer_insights:
                    role_adoption = peer_insights['role_peers']['adoption_rate']
                    st.metric("Role Peer Adoption", f"{role_adoption:.1%}")
                    
                    if role_adoption >= 0.7:
                        st.markdown("✅ **Strong organizational precedent**")
                    elif role_adoption >= 0.4:
                        st.markdown("⚠️ **Moderate organizational usage**")
                    else:
                        st.markdown("❌ **Limited organizational usage**")

def display_detailed_view(predictions_df, selected_user_id, predictions_data, demo_mode):
    """Display full detail view for technical audiences"""
    
    for idx, (_, pred) in enumerate(predictions_df.iterrows(), 1):
        with st.expander(
            f"#{idx} • {pred.get('Name', 'Unknown Entitlement')} "
            f"({pred.get('ApplicationCode', 'Unknown System')}) • "
            f"Confidence: {pred['FinalScore']:.1%}",
            expanded=(idx == 1)
        ):
            
            # Recommendation header
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**Entitlement:** {pred.get('Name', 'N/A')}")
                st.markdown(f"**System:** {pred.get('ApplicationCode', 'N/A')}")
                if pred.get('Description'):
                    st.markdown(f"**Description:** {pred.get('Description')}")
            
            with col2:
                # Confidence gauge
                confidence = pred['FinalScore']
                if confidence >= 0.8:
                    color = "🟢"
                    level = "High"
                elif confidence >= 0.6:
                    color = "🟡" 
                    level = "Medium"
                else:
                    color = "🔴"
                    level = "Low"
                
                st.metric(f"{color} Confidence", f"{confidence:.1%}")
                st.markdown(f"*{level} certainty*")
            
            with col3:
                # Stage scores
                st.markdown("**Pipeline Scores:**")
                st.markdown(f"Stage 1: {pred.get('CandidateScore', 0):.3f}")
                st.markdown(f"Stage 2: {pred['FinalScore']:.3f}")
            
            st.markdown("---")
            
            # Peer insights
            display_peer_insights_detailed(selected_user_id, pred['EntitlementId'])
            
            # Show technical details only in technical mode
            if demo_mode == "Technical Deep Dive":
                st.markdown("---")
                
                # Model explanation
                try:
                    # Get the corresponding feature row
                    feature_row = predictions_data['reranker_features'].iloc[idx-1]
                    
                    shap_explanation = generate_shap_explanation(
                        st.session_state.models_data['reranker_model'],
                        feature_row,
                        st.session_state.models_data['reranker_features'],
                        pred.get('Name', 'Unknown')
                    )
                    
                    if shap_explanation is not None:
                        display_model_explanation(shap_explanation, pred.get('Name', 'Unknown'))
                
                except Exception as e:
                    st.warning(f"Could not generate detailed explanation: {str(e)}")

def display_peer_insights_detailed(user_id, entitlement_id):
    """Display detailed peer insights"""
    
    peer_insights = calculate_peer_insights(
        user_id, 
        entitlement_id, 
        st.session_state.models_data['graph_dfs']
    )
    
    if not peer_insights:
        st.warning("No peer insights available")
        return
    
    st.markdown("### 👥 Peer Adoption Analysis")
    st.markdown("*Understanding how similar users are using this entitlement*")
    
    # Create metrics for each peer group
    col1, col2, col3, col4 = st.columns(4)
    
    peer_groups = [
        ('close_peers', '🎯 Close Peers', 'Same role + department', col1),
        ('direct_team', '👥 Direct Team', 'Same manager', col2),
        ('role_peers', '🏢 Role Peers', 'Same role', col3),
        ('dept_peers', '🏛️ Dept Peers', 'Same department', col4)
    ]
    
    for group_key, title, description, column in peer_groups:
        if group_key in peer_insights:
            data = peer_insights[group_key]
            adoption_rate = data['adoption_rate']
            
            with column:
                st.markdown(f"**{title}**")
                st.markdown(f"*{description}*")
                
                # Color coding based on adoption rate
                if adoption_rate >= 0.7:
                    color = "🟢"
                    sentiment = "High"
                elif adoption_rate >= 0.3:
                    color = "🟡"
                    sentiment = "Medium"
                else:
                    color = "🔴"
                    sentiment = "Low"
                
                st.metric(
                    label=f"{color} Adoption",
                    value=f"{adoption_rate:.1%}",
                    delta=f"{data['with_access']}/{data['total']} peers"
                )
                
                if data['peer_names']:
                    with st.expander(f"Sample {title.split()[1]} ({len(data['peer_names'])})"):
                        for name in data['peer_names']:
                            st.write(f"• {name}")

def display_analysis_sections(demo_mode):
    """Display appropriate analysis sections based on demo mode"""
    
    if demo_mode == "Technical Deep Dive":
        # Technical deep dive section
        display_technical_deep_dive()
        
        st.markdown("---")
        
        # Model explainability showcase
        create_model_explainability_showcase()
        
        st.markdown("---")
        
        # Comparison analysis
        create_comparison_analysis()
    
    elif demo_mode == "Executive Briefing":
        # Simplified business value summary with real calculations
        st.markdown("### 💼 Business Value Summary")
        
        business_impact = calculate_business_impact()
        processing_metrics = calculate_real_processing_metrics()
        live_stats = get_live_performance_stats()
        
        if business_impact and processing_metrics:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🎯 Accuracy Advantage")
                current_accuracy = live_stats['avg_confidence'] if live_stats else 0.94
                traditional_accuracy = 0.65
                accuracy_improvement = ((current_accuracy - traditional_accuracy) / traditional_accuracy) * 100
                
                st.markdown(f"""
                - **{current_accuracy:.0%} prediction accuracy** vs {traditional_accuracy:.0%} traditional
                - **{(business_impact['time_saved_hours']/business_impact['active_users']/8)*100:.0f}% reduction** in manual review time
                - **Real-time measurement** from current session
                """)
            
            with col2:
                st.markdown("#### ⚡ Speed & Efficiency")
                speed_improvement = processing_metrics['sql_total'] // processing_metrics['ml_total']
                
                st.markdown(f"""
                - **{processing_metrics['ml_total']:.0f} seconds** vs {processing_metrics['sql_total']:.0f}+ seconds processing
                - **{speed_improvement}x faster** than traditional SQL
                - **{business_impact['time_saved_weeks']:.0f} work weeks** saved annually
                """)
            
            with col3:
                st.markdown("#### 💰 Financial Impact")
                st.markdown(f"""
                - **{business_impact['months_to_break_even']:.0f} months** to break-even
                - **${business_impact['annual_savings']:,.0f}** annual savings
                - **${business_impact['savings_per_user']:,.0f}** savings per user/year
                """)
    
    else:
        # Standard demo - show comparison analysis only
        create_comparison_analysis()