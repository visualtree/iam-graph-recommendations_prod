"""
Real-time metrics calculations for Streamlit application
"""

import streamlit as st
import pandas as pd

def calculate_real_processing_metrics():
    """Calculate processing metrics based on actual data size"""
    if not st.session_state.models_loaded:
        return None
        
    graph_dfs = st.session_state.models_data['graph_dfs']
    
    # Real data metrics
    total_users = len(graph_dfs['users'])
    total_entitlements = len(graph_dfs['entitlements'])
    total_relationships = len(graph_dfs['entrecon'])
    
    # Calculate realistic processing times
    data_complexity_factor = max(1, total_relationships // 10000)
    
    # ML processing scales linearly
    ml_time = {
        'candidate_generation': 0.5 * data_complexity_factor,
        'feature_engineering': 1.0 * data_complexity_factor,
        'model_inference': 0.3,  # Model inference is constant
        'peer_analysis': 0.8 * data_complexity_factor,
        'post_processing': 0.2
    }
    
    # SQL processing scales exponentially
    sql_complexity = len([k for k in graph_dfs.keys() if isinstance(graph_dfs[k], pd.DataFrame)])
    sql_time = {
        'complex_joins': 15 * (sql_complexity ** 1.5),
        'peer_discovery': 25 * data_complexity_factor * 2,
        'aggregations': 20 * data_complexity_factor,
        'result_processing': 10 * data_complexity_factor
    }
    
    return {
        'ml_total': sum(ml_time.values()),
        'sql_total': min(300, sum(sql_time.values())),  # Cap at 5 minutes
        'data_factor': data_complexity_factor,
        'ml_breakdown': ml_time,
        'sql_breakdown': sql_time,
        'complexity_metrics': {
            'users': total_users,
            'entitlements': total_entitlements,
            'relationships': total_relationships,
            'sql_tables': sql_complexity
        }
    }

def get_live_performance_stats():
    """Get live performance statistics from current session"""
    if not st.session_state.current_predictions:
        return None
    
    predictions_df = st.session_state.current_predictions['predictions']
    
    return {
        'avg_confidence': predictions_df['FinalScore'].mean(),
        'max_confidence': predictions_df['FinalScore'].max(),
        'min_confidence': predictions_df['FinalScore'].min(),
        'confidence_std': predictions_df['FinalScore'].std(),
        'total_candidates': st.session_state.current_predictions['total_candidates'],
        'stage1_filtered': st.session_state.current_predictions['stage1_count'],
        'final_count': len(predictions_df),
        'high_confidence_count': len(predictions_df[predictions_df['FinalScore'] >= 0.8]),
        'medium_confidence_count': len(predictions_df[(predictions_df['FinalScore'] >= 0.6) & (predictions_df['FinalScore'] < 0.8)]),
        'low_confidence_count': len(predictions_df[predictions_df['FinalScore'] < 0.6]),
        'filtering_efficiency': (st.session_state.current_predictions['total_candidates'] - len(predictions_df)) / st.session_state.current_predictions['total_candidates'] * 100
    }

def calculate_business_impact():
    """Calculate real business impact based on actual organization size"""
    if not st.session_state.models_loaded:
        return None
        
    graph_dfs = st.session_state.models_data['graph_dfs']
    active_users = len(graph_dfs['users'][graph_dfs['users']['IsActive'] == True])
    
    # Business impact calculations
    current_manual_hours = active_users * 8  # 8 hours per user manual review
    with_ml_hours = active_users * 2  # 2 hours with ML assistance
    time_saved = current_manual_hours - with_ml_hours
    
    hourly_cost = 100  # Conservative hourly rate
    annual_savings = time_saved * hourly_cost
    
    # Implementation costs scale with organization size
    base_implementation_cost = 100000
    per_user_cost = 50
    total_implementation_cost = base_implementation_cost + (active_users * per_user_cost)
    
    # ROI calculations
    monthly_savings = annual_savings / 12
    months_to_break_even = total_implementation_cost / monthly_savings if monthly_savings > 0 else 999
    
    return {
        'active_users': active_users,
        'time_saved_hours': time_saved,
        'time_saved_weeks': time_saved // 40,
        'annual_savings': annual_savings,
        'implementation_cost': total_implementation_cost,
        'months_to_break_even': months_to_break_even,
        'year_1_roi': ((annual_savings - total_implementation_cost) / total_implementation_cost) * 100 if total_implementation_cost > 0 else 0,
        'cost_per_user': total_implementation_cost // active_users if active_users > 0 else 0,
        'savings_per_user': annual_savings // active_users if active_users > 0 else 0
    }

def get_data_complexity_metrics():
    """Get data complexity metrics for performance estimation"""
    if not st.session_state.models_loaded:
        return None
    
    graph_dfs = st.session_state.models_data['graph_dfs']
    
    # Calculate graph complexity
    total_users = len(graph_dfs['users'])
    active_users = len(graph_dfs['users'][graph_dfs['users']['IsActive'] == True])
    total_entitlements = len(graph_dfs['entitlements'])
    total_relationships = len(graph_dfs['entrecon'])
    
    # Calculate density and distribution metrics
    graph_density = total_relationships / (active_users * total_entitlements) if (active_users * total_entitlements) > 0 else 0
    avg_access_per_user = total_relationships / active_users if active_users > 0 else 0
    
    # Calculate organizational complexity
    org_count = len(graph_dfs['orgs']) if 'orgs' in graph_dfs else 0
    role_count = len(graph_dfs['designations']) if 'designations' in graph_dfs else 0
    system_count = len(graph_dfs['endpoints']) if 'endpoints' in graph_dfs else 0
    
    return {
        'total_users': total_users,
        'active_users': active_users,
        'total_entitlements': total_entitlements,
        'total_relationships': total_relationships,
        'graph_density': graph_density,
        'avg_access_per_user': avg_access_per_user,
        'organizational_complexity': {
            'organizations': org_count,
            'roles': role_count,
            'systems': system_count
        },
        'complexity_category': get_complexity_category(total_relationships, graph_density)
    }

def get_complexity_category(relationships, density):
    """Categorize data complexity for processing estimates"""
    if relationships < 10000:
        return "Small"
    elif relationships < 100000:
        return "Medium" if density < 0.1 else "Medium-Dense"
    elif relationships < 500000:
        return "Large" if density < 0.05 else "Large-Dense"
    else:
        return "Enterprise" if density < 0.02 else "Enterprise-Dense"