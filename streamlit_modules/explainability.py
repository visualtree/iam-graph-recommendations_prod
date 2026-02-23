"""
Model explainability components for Streamlit application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("⚠️ SHAP not available. Install with: pip install shap")

def generate_shap_explanation(model, features, feature_names, entitlement_name):
    """Generate SHAP explanation for prediction"""
    
    if not SHAP_AVAILABLE:
        return None
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features.values.reshape(1, -1))
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class for binary classification
        
        # Create DataFrame for easier handling
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'value': features.values,
            'shap_value': shap_values[0]
        })
        
        # Sort by absolute SHAP value
        shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
        shap_df = shap_df.sort_values('abs_shap', ascending=False)
        
        return shap_df
        
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {str(e)}")
        return None

def display_model_explanation(shap_df, entitlement_name):
    """Display model explanation with SHAP values"""
    
    if shap_df is None:
        return
    
    st.markdown("### 🔬 Model Decision Analysis")
    st.markdown(f"*Why the model recommended **{entitlement_name}***")
    
    # Get top features
    top_features = shap_df.head(10)
    
    # Create SHAP waterfall-style chart
    fig = go.Figure()
    
    colors = ['red' if x < 0 else 'green' for x in top_features['shap_value']]
    
    fig.add_trace(go.Bar(
        x=top_features['shap_value'],
        y=top_features['feature'],
        orientation='h',
        marker_color=colors,
        text=[f"{val:.3f}" for val in top_features['shap_value']],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Feature Impact on Prediction (SHAP Values)",
        xaxis_title="Impact on Prediction",
        yaxis_title="Features",
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature interpretation
    display_feature_interpretation(top_features)

def display_feature_interpretation(top_features):
    """Display feature interpretation with positive and negative factors"""
    
    st.markdown("#### 📊 Feature Interpretation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟢 Positive Factors (Supporting)**")
        positive_features = top_features[top_features['shap_value'] > 0].head(5)
        
        if not positive_features.empty:
            for _, row in positive_features.iterrows():
                feature_name = row['feature']
                impact = row['shap_value']
                value = row['value']
                
                # Simplify feature names for display
                display_name = simplify_feature_name(feature_name)
                
                st.markdown(f"• **{display_name}**: +{impact:.3f}")
                st.markdown(f"  *Value: {value:.3f}*")
        else:
            st.markdown("*No significant positive factors found*")
    
    with col2:
        st.markdown("**🔴 Negative Factors (Against)**")
        negative_features = top_features[top_features['shap_value'] < 0].head(5)
        
        if not negative_features.empty:
            for _, row in negative_features.iterrows():
                feature_name = row['feature']
                impact = row['shap_value']
                value = row['value']
                
                display_name = simplify_feature_name(feature_name)
                
                st.markdown(f"• **{display_name}**: {impact:.3f}")
                st.markdown(f"  *Value: {value:.3f}*")
        else:
            st.markdown("*No significant negative factors found*")

def simplify_feature_name(feature_name):
    """Simplify feature names for better display"""
    
    # Common feature name simplifications
    simplifications = {
        'embedding_cosine_similarity': 'Role Similarity Score',
        'embedding_euclidean_distance': 'Role Distance Score',
        'close_peer_adoption_rate': 'Close Peers Adoption',
        'direct_team_adoption_rate': 'Team Adoption Rate',
        'role_peer_adoption_rate': 'Role Peers Adoption',
        'dept_peer_adoption_rate': 'Department Adoption',
        'role_entitlement_count': 'Role Access Frequency',
        'manager_team_entitlement_count': 'Manager Team Access',
        'user_total_entitlements': 'User Access Count',
        'org_entitlement_count': 'Org Access Frequency'
    }
    
    # Check for exact matches first
    if feature_name in simplifications:
        return simplifications[feature_name]
    
    # Handle embedding features
    if feature_name.startswith('user_emb_') or feature_name.startswith('ent_emb_'):
        return "Graph Embedding Feature"
    
    # Handle peer features
    if 'peer' in feature_name.lower():
        return feature_name.replace('_', ' ').title()
    
    # Default: clean up underscores and capitalize
    return feature_name.replace('_', ' ').title()

def create_confidence_explanation(confidence_score, peer_insights=None):
    """Create human-readable explanation for confidence score"""
    
    explanations = []
    
    # Base confidence interpretation
    if confidence_score >= 0.9:
        explanations.append("🟢 **Very High Confidence** - Strong recommendation with minimal risk")
    elif confidence_score >= 0.8:
        explanations.append("🟢 **High Confidence** - Recommended for approval")
    elif confidence_score >= 0.7:
        explanations.append("🟡 **Good Confidence** - Likely appropriate with review")
    elif confidence_score >= 0.6:
        explanations.append("🟡 **Medium Confidence** - Requires careful evaluation")
    elif confidence_score >= 0.5:
        explanations.append("🔴 **Low Confidence** - High risk, manual review required")
    else:
        explanations.append("🔴 **Very Low Confidence** - Not recommended")
    
    # Add peer insights if available
    if peer_insights:
        high_adoption_groups = []
        low_adoption_groups = []
        
        for group_name, data in peer_insights.items():
            if data['total'] > 0:
                adoption_rate = data['adoption_rate']
                group_display = group_name.replace('_', ' ').title()
                
                if adoption_rate >= 0.7:
                    high_adoption_groups.append(f"{group_display} ({adoption_rate:.0%})")
                elif adoption_rate < 0.3:
                    low_adoption_groups.append(f"{group_display} ({adoption_rate:.0%})")
        
        if high_adoption_groups:
            explanations.append(f"✅ **Strong peer support** from: {', '.join(high_adoption_groups)}")
        
        if low_adoption_groups:
            explanations.append(f"⚠️ **Limited adoption** in: {', '.join(low_adoption_groups)}")
    
    return explanations

def display_prediction_reasoning(prediction_row, peer_insights, shap_df=None):
    """Display comprehensive reasoning for a prediction"""
    
    st.markdown("#### 🧠 AI Reasoning Summary")
    
    confidence = prediction_row['FinalScore']
    entitlement_name = prediction_row.get('Name', 'Unknown')
    
    # Get confidence explanation
    explanations = create_confidence_explanation(confidence, peer_insights)
    
    for explanation in explanations:
        st.markdown(explanation)
    
    # Risk assessment
    st.markdown("#### 🛡️ Risk Assessment")
    
    if confidence >= 0.8:
        risk_level = "LOW"
        risk_color = "🟢"
        risk_desc = "Well-established access pattern with strong peer support"
    elif confidence >= 0.6:
        risk_level = "MEDIUM"
        risk_color = "🟡"
        risk_desc = "Mixed signals require human judgment"
    else:
        risk_level = "HIGH"
        risk_color = "🔴"
        risk_desc = "Unusual access pattern, careful evaluation needed"
    
    st.markdown(f"{risk_color} **Risk Level: {risk_level}** - {risk_desc}")
    
    # Business justification
    st.markdown("#### 💼 Business Justification")
    
    if peer_insights:
        business_reasons = []
        
        for group_name, data in peer_insights.items():
            if data['total'] > 0 and data['adoption_rate'] >= 0.5:
                group_display = group_name.replace('_', ' ').title()
                business_reasons.append(
                    f"**{group_display}**: {data['adoption_rate']:.0%} adoption rate "
                    f"({data['with_access']}/{data['total']} users)"
                )
        
        if business_reasons:
            st.markdown("**Supporting evidence:**")
            for reason in business_reasons:
                st.markdown(f"• {reason}")
        else:
            st.markdown("⚠️ **Limited organizational precedent** - Consider business need carefully")
    
    # Technical details (if SHAP available)
    if shap_df is not None and not shap_df.empty:
        with st.expander("🔬 Technical Decision Factors"):
            top_factors = shap_df.head(5)
            
            st.markdown("**Top 5 Model Factors:**")
            for _, factor in top_factors.iterrows():
                impact_direction = "Supports" if factor['shap_value'] > 0 else "Opposes"
                feature_display = simplify_feature_name(factor['feature'])
                
                st.markdown(
                    f"• **{feature_display}**: {impact_direction} prediction "
                    f"(Impact: {factor['shap_value']:.3f})"
                )