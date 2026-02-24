"""
Real-time metrics calculations for Streamlit application
"""

import streamlit as st
import pandas as pd

def _estimate_processing_metrics(graph_dfs):
    """Fallback estimate used when no measured pipeline timing is available."""
    total_users = len(graph_dfs["users"])
    total_entitlements = len(graph_dfs["entitlements"])
    total_relationships = len(graph_dfs["entrecon"])

    data_complexity_factor = max(1, total_relationships // 10000)

    ml_time = {
        "candidate_generation": 0.5 * data_complexity_factor,
        "feature_engineering": 1.0 * data_complexity_factor,
        "model_inference": 0.3,
        "peer_analysis": 0.8 * data_complexity_factor,
        "post_processing": 0.2,
    }

    sql_complexity = len([k for k in graph_dfs.keys() if isinstance(graph_dfs[k], pd.DataFrame)])
    sql_time = {
        "complex_joins": 15 * (sql_complexity ** 1.5),
        "peer_discovery": 25 * data_complexity_factor * 2,
        "aggregations": 20 * data_complexity_factor,
        "result_processing": 10 * data_complexity_factor,
    }

    return {
        "mode": "estimated",
        "ml_total": sum(ml_time.values()),
        "sql_total": min(300, sum(sql_time.values())),
        "data_factor": data_complexity_factor,
        "ml_breakdown": ml_time,
        "sql_breakdown": sql_time,
        "complexity_metrics": {
            "users": total_users,
            "entitlements": total_entitlements,
            "relationships": total_relationships,
            "sql_tables": sql_complexity,
        },
    }


def calculate_real_processing_metrics():
    """Return measured pipeline timing when available, otherwise estimated timing."""
    if not st.session_state.models_loaded:
        return None

    graph_dfs = st.session_state.models_data["graph_dfs"]
    estimated = _estimate_processing_metrics(graph_dfs)

    current = st.session_state.get("current_predictions")
    if not current:
        return estimated

    timings_ms = current.get("pipeline_timings_ms") or {}
    total_ms = current.get("pipeline_total_ms")

    if not timings_ms or total_ms is None:
        return estimated

    candidate_align_ms = timings_ms.get("candidate_align", 0.0)
    reranker_align_ms = timings_ms.get("reranker_align", 0.0)

    measured_ml_breakdown = {
        "user_context": (timings_ms.get("candidate_filter", 0.0) + timings_ms.get("candidate_matrix", 0.0)) / 1000.0,
        "candidate_features": timings_ms.get("candidate_features", 0.0) / 1000.0,
        "candidate_scoring": timings_ms.get("candidate_score_rank", 0.0) / 1000.0,
        "peer_analysis": timings_ms.get("reranker_features", 0.0) / 1000.0,
        "reranker_scoring": timings_ms.get("reranker_score_rank", 0.0) / 1000.0,
        "feature_alignment": (candidate_align_ms + reranker_align_ms) / 1000.0,
    }

    measured_component_ms = sum(timings_ms.values())
    residual_ms = max(0.0, float(total_ms) - measured_component_ms)
    if residual_ms > 0:
        measured_ml_breakdown["other"] = residual_ms / 1000.0

    return {
        **estimated,
        "mode": "measured",
        "ml_total": float(total_ms) / 1000.0,
        "ml_breakdown": measured_ml_breakdown,
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
