# """
# Prediction pipeline for Streamlit application
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# import traceback

# # Import ML pipeline modules
# from ml_pipeline import feature_engineering

# def run_prediction_pipeline(user_id, models_data, top_n=5, candidates=100):
    # """Run the full prediction pipeline with detailed tracking"""
    
    # progress_bar = st.progress(0)
    # status_text = st.empty()
    
    # try:
        # # Stage 1: Generate candidates
        # status_text.text("🔍 Stage 1: Generating candidate entitlements...")
        # progress_bar.progress(20)
        
        # graph_dfs = models_data['graph_dfs']
        # embeddings_df = models_data['embeddings_df']
        
        # # Get current entitlements
        # current_ents = set(graph_dfs['entrecon'][graph_dfs['entrecon']['UserId'] == user_id]['EntitlementId'])
        # all_ents = graph_dfs['entitlements']
        # candidate_ents_df = all_ents[~all_ents['id'].isin(current_ents)]
        
        # if candidate_ents_df.empty:
            # st.warning("No candidate entitlements found for this user.")
            # return None
        
        # # Create candidates DataFrame
        # candidates_df = pd.DataFrame({
            # 'UserId': [user_id] * len(candidate_ents_df),
            # 'EntitlementId': candidate_ents_df['id'].tolist()
        # })
        # candidates_df['UserId'] = candidates_df['UserId'].astype('int64')
        # candidates_df['EntitlementId'] = candidates_df['EntitlementId'].astype('string')
        
        # progress_bar.progress(40)
        
        # # Generate candidate features
        # status_text.text("🧮 Generating candidate features...")
        # X_cand, _, _ = feature_engineering.create_candidate_model_features(
            # candidates_df.copy(), embeddings_df
        # )
        
        # # Ensure feature alignment for candidate model
        # missing_cand_features = [f for f in models_data['candidate_features'] if f not in X_cand.columns]
        # if missing_cand_features:
            # for feat in missing_cand_features:
                # X_cand[feat] = 0
        
        # X_cand = X_cand[models_data['candidate_features']]
        
        # progress_bar.progress(60)
        
        # # Stage 1 predictions
        # status_text.text("🎯 Stage 1: Scoring candidates...")
        # pred_probs_cand = models_data['candidate_model'].predict_proba(X_cand)[:, 1]
        # candidates_df['CandidateScore'] = pred_probs_cand
        # top_candidates = candidates_df.sort_values('CandidateScore', ascending=False).head(candidates)
        
        # progress_bar.progress(80)
        
        # # Stage 2: Enhanced reranking
        # status_text.text("🔬 Stage 2: Enhanced reranking with peer analysis...")
        # X_rerank, _, _ = feature_engineering.create_enhanced_reranker_features(
            # top_candidates.copy(), embeddings_df, graph_dfs
        # )
        
        # # Ensure feature alignment for reranker model
        # missing_rerank_features = [f for f in models_data['reranker_features'] if f not in X_rerank.columns]
        # if missing_rerank_features:
            # for feat in missing_rerank_features:
                # X_rerank[feat] = 0
        
        # X_rerank = X_rerank[models_data['reranker_features']]
        
        # # Final predictions
        # pred_probs_rerank = models_data['reranker_model'].predict_proba(X_rerank)[:, 1]
        # top_candidates['FinalScore'] = pred_probs_rerank
        
        # final_recs = top_candidates.sort_values('FinalScore', ascending=False).head(top_n)
        
        # progress_bar.progress(100)
        # status_text.text("✅ Prediction pipeline completed!")
        
        # # Add entitlement details
        # final_recs = add_entitlement_details(final_recs, graph_dfs)
        
        # return {
            # 'predictions': final_recs,
            # 'candidate_features': X_cand,
            # 'reranker_features': X_rerank,
            # 'stage1_count': len(top_candidates),
            # 'total_candidates': len(candidate_ents_df)
        # }
        
    # except Exception as e:
        # st.error(f"❌ Prediction pipeline failed: {str(e)}")
        # st.text(traceback.format_exc())
        # return None

# def add_entitlement_details(predictions_df, graph_dfs):
    # """Add detailed entitlement information to predictions"""
    
    # # Add entitlement details
    # predictions_df = predictions_df.merge(
        # graph_dfs['entitlements'][['id', 'Name', 'Description']], 
        # left_on='EntitlementId', right_on='id', 
        # how='left', suffixes=('', '_ent')
    # )
    
    # # Add endpoint system info
    # predictions_df['EndpointSystemId'] = predictions_df['EntitlementId'].astype(str).str.split('_').str[0].astype('Int64')
    
    # if 'endpoints' in graph_dfs:
        # endpoint_cols = ['id', 'ApplicationCode']
        # if 'DisplayName' in graph_dfs['endpoints'].columns:
            # endpoint_cols.append('DisplayName')
        
        # predictions_df = predictions_df.merge(
            # graph_dfs['endpoints'][endpoint_cols], 
            # left_on='EndpointSystemId', right_on='id', 
            # how='left', suffixes=('', '_sys')
        # )
    
    # return predictions_df

# def calculate_peer_insights(user_id, entitlement_id, graph_dfs):
    # """Calculate detailed peer adoption insights"""
    
    # user_info = graph_dfs['users'][graph_dfs['users']['id'] == user_id]
    # if user_info.empty:
        # return None
    
    # user_data = user_info.iloc[0]
    # user_role = user_data.get('NBusinessRoleId')
    # user_org = user_data.get('NOrganisationId')
    # user_manager = user_data.get('ManagerId')
    
    # insights = {}
    
    # # Close peers (same role + same org)
    # if pd.notna(user_role) and pd.notna(user_org):
        # close_peers = graph_dfs['users'][
            # (graph_dfs['users']['NBusinessRoleId'] == user_role) & 
            # (graph_dfs['users']['NOrganisationId'] == user_org) &
            # (graph_dfs['users']['id'] != user_id) &
            # (graph_dfs['users']['IsActive'] == True)
        # ]
        
        # close_access = graph_dfs['entrecon'][
            # (graph_dfs['entrecon']['UserId'].isin(close_peers['id'])) &
            # (graph_dfs['entrecon']['EntitlementId'] == entitlement_id)
        # ]
        
        # insights['close_peers'] = {
            # 'total': len(close_peers),
            # 'with_access': len(close_access),
            # 'adoption_rate': len(close_access) / len(close_peers) if len(close_peers) > 0 else 0,
            # 'peer_names': close_peers['UserName'].tolist()[:5] if len(close_peers) > 0 else []
        # }
    
    # # Direct team (same manager)
    # if pd.notna(user_manager):
        # team_members = graph_dfs['users'][
            # (graph_dfs['users']['ManagerId'] == user_manager) & 
            # (graph_dfs['users']['id'] != user_id) &
            # (graph_dfs['users']['IsActive'] == True)
        # ]
        
        # team_access = graph_dfs['entrecon'][
            # (graph_dfs['entrecon']['UserId'].isin(team_members['id'])) &
            # (graph_dfs['entrecon']['EntitlementId'] == entitlement_id)
        # ]
        
        # insights['direct_team'] = {
            # 'total': len(team_members),
            # 'with_access': len(team_access),
            # 'adoption_rate': len(team_access) / len(team_members) if len(team_members) > 0 else 0,
            # 'peer_names': team_members['UserName'].tolist()[:5] if len(team_members) > 0 else []
        # }
    
    # # Role peers (same role, any department)
    # if pd.notna(user_role):
        # role_peers = graph_dfs['users'][
            # (graph_dfs['users']['NBusinessRoleId'] == user_role) & 
            # (graph_dfs['users']['id'] != user_id) &
            # (graph_dfs['users']['IsActive'] == True)
        # ]
        
        # role_access = graph_dfs['entrecon'][
            # (graph_dfs['entrecon']['UserId'].isin(role_peers['id'])) &
            # (graph_dfs['entrecon']['EntitlementId'] == entitlement_id)
        # ]
        
        # insights['role_peers'] = {
            # 'total': len(role_peers),
            # 'with_access': len(role_access),
            # 'adoption_rate': len(role_access) / len(role_peers) if len(role_peers) > 0 else 0,
            # 'peer_names': role_peers['UserName'].tolist()[:5] if len(role_peers) > 0 else []
        # }
    
    # # Department peers (same org, any role)
    # if pd.notna(user_org):
        # dept_peers = graph_dfs['users'][
            # (graph_dfs['users']['NOrganisationId'] == user_org) & 
            # (graph_dfs['users']['id'] != user_id) &
            # (graph_dfs['users']['IsActive'] == True)
        # ]
        
        # dept_access = graph_dfs['entrecon'][
            # (graph_dfs['entrecon']['UserId'].isin(dept_peers['id'])) &
            # (graph_dfs['entrecon']['EntitlementId'] == entitlement_id)
        # ]
        
        # insights['dept_peers'] = {
            # 'total': len(dept_peers),
            # 'with_access': len(dept_access),
            # 'adoption_rate': len(dept_access) / len(dept_peers) if len(dept_peers) > 0 else 0,
            # 'peer_names': dept_peers['UserName'].tolist()[:5] if len(dept_peers) > 0 else []
        # }
    
    # return insights
    
    
# streamlit_modules/prediction_engine.py - MINIMAL VERSION WITH NO DUPLICATION
# streamlit_modules/prediction_engine.py - Complete version using prediction_core
# streamlit_modules/prediction_engine.py - USE prediction_core.py

# streamlit_modules/prediction_engine.py - USE prediction_core.py

# streamlit_modules/prediction_engine.py - USE prediction_core.py

# streamlit_modules/prediction_engine.py - USE prediction_core.py

# streamlit_modules/prediction_engine.py - USE prediction_core.py

# streamlit_modules/prediction_engine.py - USE prediction_core.py

# streamlit_modules/prediction_engine.py - USE prediction_core.py
# streamlit_modules/prediction_engine.py - Complete working version
# streamlit_modules/prediction_engine.py - Complete working version

# streamlit_modules/prediction_engine.py - Complete working version

# streamlit_modules/prediction_engine.py - Complete working version

import streamlit as st
import pandas as pd
import sys
import os

# Add ML pipeline to path  
sys.path.append('ml_pipeline')

# Import from prediction_core.py and feature_engineering
from ml_pipeline.prediction_core import (
    _hard_fail_feature_alignment,
    PredictionArtifacts
)
from ml_pipeline import feature_engineering

def run_prediction_pipeline(user_id, models_data, top_n=5, candidates=100):
    """Main prediction pipeline function that uses shared hard-fail alignment"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Stage 1: Generate candidates
        status_text.text("🔍 Stage 1: Generating candidate entitlements...")
        progress_bar.progress(20)
        
        graph_dfs = models_data['graph_dfs']
        embeddings_df = models_data['embeddings_df']
        
        # Get current entitlements
        current_ents = set(graph_dfs['entrecon'][graph_dfs['entrecon']['UserId'] == user_id]['EntitlementId'])
        all_ents = graph_dfs['entitlements']
        candidate_ents_df = all_ents[~all_ents['id'].isin(current_ents)]
        
        if candidate_ents_df.empty:
            st.warning("No candidate entitlements found for this user.")
            return None
        
        # Create candidates DataFrame
        candidates_df = pd.DataFrame({
            'UserId': [user_id] * len(candidate_ents_df),
            'EntitlementId': candidate_ents_df['id'].tolist()
        })
        candidates_df['UserId'] = candidates_df['UserId'].astype('int64')
        candidates_df['EntitlementId'] = candidates_df['EntitlementId'].astype('string')
        
        progress_bar.progress(40)
        
        # Generate candidate features
        status_text.text("🧮 Generating candidate features...")
        X_cand, _, _ = feature_engineering.create_candidate_model_features(
            candidates_df.copy(), embeddings_df
        )
        
        # ✅ USE SHARED HARD-FAIL ALIGNMENT FROM prediction_core
        X_cand = _hard_fail_feature_alignment(X_cand, models_data['candidate_features'], "candidate")
        
        progress_bar.progress(60)
        
        # Stage 1 predictions
        status_text.text("🎯 Stage 1: Scoring candidates...")
        pred_probs_cand = models_data['candidate_model'].predict_proba(X_cand)[:, 1]
        candidates_df['CandidateScore'] = pred_probs_cand
        top_candidates = candidates_df.sort_values('CandidateScore', ascending=False).head(candidates)
        
        progress_bar.progress(80)
        
        # Stage 2: Enhanced reranking
        status_text.text("🔬 Stage 2: Enhanced reranking with peer analysis...")
        X_rerank, _, _ = feature_engineering.create_enhanced_reranker_features(
            top_candidates.copy(), embeddings_df, graph_dfs
        )
        
        # ✅ USE SHARED HARD-FAIL ALIGNMENT FROM prediction_core
        X_rerank = _hard_fail_feature_alignment(X_rerank, models_data['reranker_features'], "reranker")
        
        # Final predictions
        pred_probs_rerank = models_data['reranker_model'].predict_proba(X_rerank)[:, 1]
        top_candidates['FinalScore'] = pred_probs_rerank
        
        final_recs = top_candidates.sort_values('FinalScore', ascending=False).head(top_n)
        
        progress_bar.progress(100)
        status_text.text("✅ Prediction pipeline completed!")
        
        # Add entitlement details
        final_recs = add_entitlement_details_streamlit(final_recs, {'graph_dfs': graph_dfs})
        
        return {
            'predictions': final_recs,
            'candidate_features': X_cand,
            'reranker_features': X_rerank,
            'stage1_count': len(top_candidates),
            'total_candidates': len(candidate_ents_df)
        }
        
    except Exception as e:
        st.error(f"❌ Prediction pipeline failed: {str(e)}")
        progress_bar.progress(0)
        status_text.text("❌ Prediction failed")
        return None

def add_entitlement_details_streamlit(predictions_df, artifacts):
    """Add detailed entitlement information for Streamlit display"""
    
    graph_dfs = artifacts['graph_dfs']
    
    # Add entitlement details
    predictions_df = predictions_df.merge(
        graph_dfs['entitlements'][['id', 'Name', 'Description']], 
        left_on='EntitlementId', right_on='id', 
        how='left', suffixes=('', '_ent')
    )
    
    # Add endpoint system info
    predictions_df['EndpointSystemId'] = predictions_df['EntitlementId'].astype(str).str.split('_').str[0].astype('Int64')
    
    if 'endpoints' in graph_dfs:
        endpoint_cols = ['id', 'ApplicationCode']
        if 'DisplayName' in graph_dfs['endpoints'].columns:
            endpoint_cols.append('DisplayName')
        
        predictions_df = predictions_df.merge(
            graph_dfs['endpoints'][endpoint_cols], 
            left_on='EndpointSystemId', right_on='id', 
            how='left', suffixes=('', '_sys')
        )
    
    return predictions_df

# Replace the calculate_peer_insights function in prediction_engine.py

def calculate_peer_insights(user_id, entitlement_id, graph_dfs):
    """Calculate peer adoption insights with FIXED dropdowns showing only users with access"""
    
    try:
        insights = {}
        
        # Get user information
        user_info = graph_dfs['users'][graph_dfs['users']['id'] == user_id]
        if user_info.empty:
            return None
        
        user_role = user_info.iloc[0]['NBusinessRoleId']
        user_org = user_info.iloc[0]['NOrganisationId']
        user_manager = user_info.iloc[0]['ManagerId']
        
        # Close peers (same role + same department)
        if pd.notna(user_role) and pd.notna(user_org):
            close_peers = graph_dfs['users'][
                (graph_dfs['users']['NBusinessRoleId'] == user_role) & 
                (graph_dfs['users']['NOrganisationId'] == user_org) &
                (graph_dfs['users']['id'] != user_id) &
                (graph_dfs['users']['IsActive'] == True)
            ]
            
            close_access = graph_dfs['entrecon'][
                (graph_dfs['entrecon']['UserId'].isin(close_peers['id'])) &
                (graph_dfs['entrecon']['EntitlementId'] == entitlement_id)
            ]
            
            # FIXED: Only show users who have access
            close_users_with_access = close_peers[close_peers['id'].isin(close_access['UserId'])]
            
            insights['close_peers'] = {
                'total': len(close_peers),
                'with_access': len(close_access),
                'adoption_rate': len(close_access) / len(close_peers) if len(close_peers) > 0 else 0,
                'peer_names': close_users_with_access['UserName'].tolist()[:5]  # ✅ Only users with access
            }
        
        # Direct team (same manager)
        if pd.notna(user_manager):
            team_members = graph_dfs['users'][
                (graph_dfs['users']['ManagerId'] == user_manager) & 
                (graph_dfs['users']['id'] != user_id) &
                (graph_dfs['users']['IsActive'] == True)
            ]
            
            team_access = graph_dfs['entrecon'][
                (graph_dfs['entrecon']['UserId'].isin(team_members['id'])) &
                (graph_dfs['entrecon']['EntitlementId'] == entitlement_id)
            ]
            
            # FIXED: Only show users who have access
            team_users_with_access = team_members[team_members['id'].isin(team_access['UserId'])]
            
            insights['direct_team'] = {
                'total': len(team_members),
                'with_access': len(team_access),
                'adoption_rate': len(team_access) / len(team_members) if len(team_members) > 0 else 0,
                'peer_names': team_users_with_access['UserName'].tolist()[:5]  # ✅ Only users with access
            }
        
        # Role peers (same role, any department) - FIXED WITH DEBUG
        if pd.notna(user_role):
            role_peers = graph_dfs['users'][
                (graph_dfs['users']['NBusinessRoleId'] == user_role) & 
                (graph_dfs['users']['id'] != user_id) &
                (graph_dfs['users']['IsActive'] == True)
            ]
            
            role_access = graph_dfs['entrecon'][
                (graph_dfs['entrecon']['UserId'].isin(role_peers['id'])) &
                (graph_dfs['entrecon']['EntitlementId'] == entitlement_id)
            ]
            
            # Only show users who have access
            role_users_with_access = role_peers[role_peers['id'].isin(role_access['UserId'])]
            
            # DEBUG: Check org table structure
            org_lookup = graph_dfs.get('orgs', pd.DataFrame())
            print(f"🔍 DEBUG - Orgs table columns: {org_lookup.columns.tolist()}")
            print(f"🔍 DEBUG - Orgs table shape: {org_lookup.shape}")
            if not org_lookup.empty:
                print(f"🔍 DEBUG - Sample org data: {org_lookup.head(2).to_dict()}")
            
            # Try multiple possible column names for organization name
            org_names = {}
            if not org_lookup.empty and 'id' in org_lookup.columns:
                # Try different possible name columns
                #name_columns = ['DisplayName', 'Name', 'OrganisationName', 'OrgName', 'display_name', 'name']
                name_columns = ['Name']
                name_col = None
                
                for col in name_columns:
                    if col in org_lookup.columns:
                        name_col = col
                        print(f"✅ DEBUG - Using org name column: {col}")
                        break
                
                if name_col:
                    org_names = org_lookup.set_index('id')[name_col].to_dict()
                    print(f"🔍 DEBUG - Org lookup created with {len(org_names)} entries")
                else:
                    print(f"⚠️ DEBUG - No valid name column found in orgs table")
                    print(f"Available columns: {org_lookup.columns.tolist()}")
            
            # Create peer names with department info
            role_peer_names = []
            for _, user in role_users_with_access.head(5).iterrows():
                user_name = user['UserName']
                user_org_id = user.get('NOrganisationId')
                
                print(f"🔍 DEBUG - User: {user_name}, OrgId: {user_org_id}, Type: {type(user_org_id)}")
                
                if user_org_id and user_org_id in org_names:
                    dept_name = org_names[user_org_id]
                    role_peer_names.append(f"{user_name} ({dept_name})")
                    print(f"✅ DEBUG - Found dept: {dept_name}")
                else:
                    # Fallback: use org ID or just name
                    if user_org_id:
                        role_peer_names.append(f"{user_name} (Org {user_org_id})")
                        print(f"⚠️ DEBUG - Org not found, using ID: {user_org_id}")
                    else:
                        role_peer_names.append(f"{user_name}")
                        print(f"⚠️ DEBUG - No org ID for user")
            
            insights['role_peers'] = {
                'total': len(role_peers),
                'with_access': len(role_access),
                'adoption_rate': len(role_access) / len(role_peers) if len(role_peers) > 0 else 0,
                'peer_names': role_peer_names
            }
        
        # Department peers (same org, any role)
        if pd.notna(user_org):
            dept_peers = graph_dfs['users'][
                (graph_dfs['users']['NOrganisationId'] == user_org) & 
                (graph_dfs['users']['id'] != user_id) &
                (graph_dfs['users']['IsActive'] == True)
            ]
            
            dept_access = graph_dfs['entrecon'][
                (graph_dfs['entrecon']['UserId'].isin(dept_peers['id'])) &
                (graph_dfs['entrecon']['EntitlementId'] == entitlement_id)
            ]
            
            # FIXED: Only show users who have access
            dept_users_with_access = dept_peers[dept_peers['id'].isin(dept_access['UserId'])]
            
            insights['dept_peers'] = {
                'total': len(dept_peers),
                'with_access': len(dept_access),
                'adoption_rate': len(dept_access) / len(dept_peers) if len(dept_peers) > 0 else 0,
                'peer_names': dept_users_with_access['UserName'].tolist()[:5]  # ✅ Only users with access
            }
        
        return insights
        
    except Exception as e:
        print(f"Error calculating peer insights: {e}")
        return None

def format_predictions_for_streamlit(results):
    """Format prediction results for Streamlit display"""
    
    if not results or results['predictions'].empty:
        return None
    
    final_recs = results['predictions'].copy()
    artifacts = results.get('artifacts', {})
    
    # Add entitlement details if artifacts available
    if artifacts:
        final_recs = add_entitlement_details_streamlit(final_recs, artifacts)
    
    # Add original entitlement ID for display
    final_recs['OriginalEntitlementId'] = final_recs['EntitlementId'].astype(str).str.split('_').str[1]
    
    # Select columns for display
    display_cols = ['OriginalEntitlementId', 'Name', 'FinalScore']
    
    # Add optional columns if they exist
    if 'ApplicationCode' in final_recs.columns:
        display_cols.insert(-1, 'ApplicationCode')
    if 'DisplayName' in final_recs.columns:
        display_cols.insert(-1, 'DisplayName')
    
    # Rename columns for better display
    column_mapping = {
        'OriginalEntitlementId': 'Entitlement ID',
        'Name': 'Entitlement Name',
        'ApplicationCode': 'Application',
        'DisplayName': 'System Name',
        'FinalScore': 'Confidence Score'
    }
    
    display_df = final_recs[display_cols].copy()
    display_df = display_df.rename(columns=column_mapping)
    
    # Format confidence score as percentage
    if 'Confidence Score' in display_df.columns:
        display_df['Confidence Score'] = display_df['Confidence Score'].apply(lambda x: f"{x:.1%}")
    
    return display_df

def run_prediction_pipeline(user_id, models_data, top_n=5, candidates=100):
    """Main prediction pipeline function that uses shared hard-fail alignment"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Stage 1: Generate candidates
        status_text.text("🔍 Stage 1: Generating candidate entitlements...")
        progress_bar.progress(20)
        
        graph_dfs = models_data['graph_dfs']
        embeddings_df = models_data['embeddings_df']
        
        # Get current entitlements
        current_ents = set(graph_dfs['entrecon'][graph_dfs['entrecon']['UserId'] == user_id]['EntitlementId'])
        all_ents = graph_dfs['entitlements']
        candidate_ents_df = all_ents[~all_ents['id'].isin(current_ents)]
        
        if candidate_ents_df.empty:
            st.warning("No candidate entitlements found for this user.")
            return None
        
        # Create candidates DataFrame
        candidates_df = pd.DataFrame({
            'UserId': [user_id] * len(candidate_ents_df),
            'EntitlementId': candidate_ents_df['id'].tolist()
        })
        candidates_df['UserId'] = candidates_df['UserId'].astype('int64')
        candidates_df['EntitlementId'] = candidates_df['EntitlementId'].astype('string')
        
        progress_bar.progress(40)
        
        # Generate candidate features
        status_text.text("🧮 Generating candidate features...")
        X_cand, _, _ = feature_engineering.create_candidate_model_features(
            candidates_df.copy(), embeddings_df
        )
        
        # ✅ USE SHARED HARD-FAIL ALIGNMENT FROM prediction_core
        X_cand = _hard_fail_feature_alignment(X_cand, models_data['candidate_features'], "candidate")
        
        progress_bar.progress(60)
        
        # Stage 1 predictions
        status_text.text("🎯 Stage 1: Scoring candidates...")
        pred_probs_cand = models_data['candidate_model'].predict_proba(X_cand)[:, 1]
        candidates_df['CandidateScore'] = pred_probs_cand
        top_candidates = candidates_df.sort_values('CandidateScore', ascending=False).head(candidates)
        
        progress_bar.progress(80)
        
        # Stage 2: Enhanced reranking
        status_text.text("🔬 Stage 2: Enhanced reranking with peer analysis...")
        X_rerank, _, _ = feature_engineering.create_enhanced_reranker_features(
            top_candidates.copy(), embeddings_df, graph_dfs
        )
        
        # ✅ USE SHARED HARD-FAIL ALIGNMENT FROM prediction_core
        X_rerank = _hard_fail_feature_alignment(X_rerank, models_data['reranker_features'], "reranker")
        
        # Final predictions
        pred_probs_rerank = models_data['reranker_model'].predict_proba(X_rerank)[:, 1]
        top_candidates['FinalScore'] = pred_probs_rerank
        
        final_recs = top_candidates.sort_values('FinalScore', ascending=False).head(top_n)
        
        progress_bar.progress(100)
        status_text.text("✅ Prediction pipeline completed!")
        
        # Add entitlement details
        final_recs = add_entitlement_details_streamlit(final_recs, {'graph_dfs': graph_dfs})
        
        return {
            'predictions': final_recs,
            'candidate_features': X_cand,
            'reranker_features': X_rerank,
            'stage1_count': len(top_candidates),
            'total_candidates': len(candidate_ents_df)
        }
        
    except Exception as e:
        st.error(f"❌ Prediction pipeline failed: {str(e)}")
        progress_bar.progress(0)
        status_text.text("❌ Prediction failed")
        return None

def add_entitlement_details_streamlit(predictions_df, artifacts):
    """Add detailed entitlement information for Streamlit display"""
    
    graph_dfs = artifacts['graph_dfs']
    
    # Add entitlement details
    predictions_df = predictions_df.merge(
        graph_dfs['entitlements'][['id', 'Name', 'Description']], 
        left_on='EntitlementId', right_on='id', 
        how='left', suffixes=('', '_ent')
    )
    
    # Add endpoint system info
    predictions_df['EndpointSystemId'] = predictions_df['EntitlementId'].astype(str).str.split('_').str[0].astype('Int64')
    
    if 'endpoints' in graph_dfs:
        endpoint_cols = ['id', 'ApplicationCode']
        if 'DisplayName' in graph_dfs['endpoints'].columns:
            endpoint_cols.append('DisplayName')
        
        predictions_df = predictions_df.merge(
            graph_dfs['endpoints'][endpoint_cols], 
            left_on='EndpointSystemId', right_on='id', 
            how='left', suffixes=('', '_sys')
        )
    
    return predictions_df

def calculate_peer_insights_streamlit(user_id, entitlement_id):
    """
    Streamlit wrapper for peer insights calculation
    Uses the shared function from prediction_core
    """
    try:
        # ✅ Use shared function from prediction_core
        peer_insights = calculate_peer_insights(user_id, entitlement_id)
        
        if not peer_insights:
            return None
            
        # Format for Streamlit display
        return {
            'close_peers': {
                'adoption_rate': peer_insights['close_peers']['adoption_rate'],
                'total': peer_insights['close_peers']['total']
            },
            'direct_team': {
                'adoption_rate': peer_insights['direct_team']['adoption_rate'],
                'total': peer_insights['direct_team']['total']
            },
            'role_peers': {
                'adoption_rate': peer_insights['role_peers']['adoption_rate'],
                'total': peer_insights['role_peers']['total']
            },
            'dept_peers': {
                'adoption_rate': peer_insights['dept_peers']['adoption_rate'],
                'total': peer_insights['dept_peers']['total']
            }
        }
        
    except Exception as e:
        st.error(f"Error calculating peer insights: {e}")
        return None

def generate_shap_explanation_streamlit(results, display_in_streamlit=True):
    """
    Generate SHAP explanation using your existing approach
    """
    try:
        import shap
        
        final_recs = results['predictions']
        if final_recs.empty:
            return None
            
        # You'll need to adapt this based on how your existing SHAP generation works
        st.info("SHAP explanation generation - adapt this to your existing method")
        
        return None
        
    except ImportError:
        st.error("SHAP not available. Install with: pip install shap")
        return None
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {e}")
        return None

def format_predictions_for_streamlit(results):
    """Format prediction results for Streamlit display"""
    
    if not results or results['predictions'].empty:
        return None
    
    final_recs = results['predictions'].copy()
    artifacts = results['artifacts']
    
    # Add entitlement details
    final_recs = add_entitlement_details_streamlit(final_recs, artifacts)
    
    # Add original entitlement ID for display
    final_recs['OriginalEntitlementId'] = final_recs['EntitlementId'].astype(str).str.split('_').str[1]
    
    # Select columns for display
    display_cols = ['OriginalEntitlementId', 'Name', 'FinalScore']
    
    # Add optional columns if they exist
    if 'ApplicationCode' in final_recs.columns:
        display_cols.insert(-1, 'ApplicationCode')
    if 'DisplayName' in final_recs.columns:
        display_cols.insert(-1, 'DisplayName')
    
    # Rename columns for better display
    column_mapping = {
        'OriginalEntitlementId': 'Entitlement ID',
        'Name': 'Entitlement Name',
        'ApplicationCode': 'Application',
        'DisplayName': 'System Name',
        'FinalScore': 'Confidence Score'
    }
    
    display_df = final_recs[display_cols].copy()
    display_df = display_df.rename(columns=column_mapping)
    
    # Format confidence score as percentage
    if 'Confidence Score' in display_df.columns:
        display_df['Confidence Score'] = display_df['Confidence Score'].apply(lambda x: f"{x:.1%}")
    
    return display_df

