# # CLEANED predict.py - READY FOR NEW MODEL

# import os, joblib, pandas as pd, shap, warnings, argparse, numpy as np
# warnings.filterwarnings("ignore", category=UserWarning, module='shap')
# warnings.filterwarnings("ignore", category=FutureWarning)
# from . import config, feature_engineering
# # predict.py - simplified using shared functions
# from . import prediction_core

# def recommend_access_for_user(user_id, topN=5, initial_candidates=100):
    # # """Generates scalable and explainable recommendations using a two-stage model."""
    # # try:
        # # user_id = int(user_id)
    # # except (ValueError, TypeError):
        # # print(f"🚨 Error: User ID must be a valid integer. You provided '{user_id}'.")
        # # return

    # # print(f"====== GENERATING SCALABLE RECOMMENDATIONS FOR USER ID: {user_id} ======")
    
    # # print("🚀 Loading all production artifacts...")
    # # try:
        # # cand_model = joblib.load(os.path.join(config.ARTIFACT_DIR, 'candidate_model.joblib'))
        # # cand_features = joblib.load(os.path.join(config.ARTIFACT_DIR, 'candidate_model_features.joblib'))
        # # rerank_model = joblib.load(os.path.join(config.ARTIFACT_DIR, 'reranker_model.joblib'))
        # # rerank_features = joblib.load(os.path.join(config.ARTIFACT_DIR, 'reranker_model_features.joblib'))
        # # embeddings_df = pd.read_pickle(os.path.join(config.ARTIFACT_DIR, 'embeddings.pkl'))
        # # graph_dfs = {
            # # 'users': pd.read_pickle(os.path.join(config.ARTIFACT_DIR, 'users.pkl')),
            # # 'entitlements': pd.read_pickle(os.path.join(config.ARTIFACT_DIR, 'entitlements.pkl')),
            # # 'entrecon': pd.read_pickle(os.path.join(config.ARTIFACT_DIR, 'entrecon.pkl')),
            # # 'orgs': pd.read_pickle(os.path.join(config.ARTIFACT_DIR, 'orgs.pkl')),
            # # 'endpoints': pd.read_pickle(os.path.join(config.ARTIFACT_DIR, 'endpoints.pkl')),
            # # 'designations': pd.read_pickle(os.path.join(config.ARTIFACT_DIR, 'designations.pkl'))
        # # }
    # # except FileNotFoundError as e:
        # # print(f"🚨 Error: Artifact not found: {e}. Please run the training pipeline first.")
        # # return

    # # # Check what features the trained model expects
    # # all_features_expected = rerank_features
    # # peer_features_expected = [f for f in rerank_features if 'peer' in f]
    
    # # print(f"🔍 Model expects {len(rerank_features)} features total")
    # # if peer_features_expected:
        # # print(f"🔍 Including {len(peer_features_expected)} peer features: {peer_features_expected}")
    
    # # # TYPE SAFETY: Ensure loaded data has correct types
    # # print("🔧 Verifying data type consistency...")
    # # if 'users' in graph_dfs:
        # # print(f"   Users ID type: {graph_dfs['users']['id'].dtype}")
    # # if 'entitlements' in graph_dfs:
        # # print(f"   Entitlements ID type: {graph_dfs['entitlements']['id'].dtype}")
    # # if 'entrecon' in graph_dfs:
        # # print(f"   Entrecon types: UserId={graph_dfs['entrecon']['UserId'].dtype}, EntitlementId={graph_dfs['entrecon']['EntitlementId'].dtype}")

    # # user_details = graph_dfs['users'][graph_dfs['users']['id'] == user_id]
    # # if user_details.empty:
        # # print(f"🚨 Error: User ID {user_id} not found in the dataset.")
        # # return
    
    # # print("\n--- USER CONTEXT ---")
    # # user_series = user_details.iloc[0]
    # # org_id, desig_id, manager_id = user_series.get('NOrganisationId'), user_series.get('NBusinessRoleId'), user_series.get('ManagerId')
    
    # # # Handle nullable Int64 types properly
    # # org_name = 'N/A'
    # # if pd.notna(org_id):
        # # org_match = graph_dfs['orgs'][graph_dfs['orgs']['id'] == org_id]
        # # org_name = org_match['Name'].iloc[0] if not org_match.empty else 'N/A'
    
    # # desig_name = 'N/A'
    # # if pd.notna(desig_id):
        # # desig_match = graph_dfs['designations'][graph_dfs['designations']['id'] == desig_id]
        # # desig_name = desig_match['Code'].iloc[0] if not desig_match.empty else 'N/A'
    
    # # mgr_name = 'N/A'
    # # if pd.notna(manager_id):
        # # mgr_match = graph_dfs['users'][graph_dfs['users']['id'] == manager_id]
        # # mgr_name = mgr_match['UserName'].iloc[0] if not mgr_match.empty else 'N/A'
    
    # # print(f"  User Name:    {user_series.get('UserName', 'N/A')}")
    # # print(f"  Display Name: {user_series.get('DisplayName', 'N/A')}")
    # # print(f"  Organization: {org_name} (ID: {org_id})")
    # # print(f"  Designation:  {desig_name} (ID: {desig_id})")
    # # print(f"  Manager:      {mgr_name} (ID: {manager_id})")

    # # print(f"\n🚀 Stage 1: Generating top {initial_candidates} candidates with FAST model...")
    
    # # # Get current entitlements (using string entitlement IDs for consistency)
    # # current_ents = set(graph_dfs['entrecon'][graph_dfs['entrecon']['UserId'] == user_id]['EntitlementId'])
    # # all_ents = graph_dfs['entitlements']
    # # candidate_ents_df = all_ents[~all_ents['id'].isin(current_ents)]
    
    # # # TYPE SAFETY: Ensure candidate DataFrame has correct types
    # # candidates_df = pd.DataFrame({
        # # 'UserId': [user_id] * len(candidate_ents_df), 
        # # 'EntitlementId': candidate_ents_df['id'].tolist()
    # # })
    # # candidates_df['UserId'] = candidates_df['UserId'].astype('int64')
    # # candidates_df['EntitlementId'] = candidates_df['EntitlementId'].astype('string')
    
    # # if candidates_df.empty:
        # # print("🚨 No candidate entitlements found for this user.")
        # # return
    
    # # X_cand, _, _ = feature_engineering.create_candidate_model_features(candidates_df.copy(), embeddings_df)
    
    # # # Ensure feature alignment
    # # missing_cand_features = [f for f in cand_features if f not in X_cand.columns]
    # # if missing_cand_features:
        # # print(f"⚠️ Adding {len(missing_cand_features)} missing candidate features as zeros")
        # # for feat in missing_cand_features:
            # # X_cand[feat] = 0
    
    # # X_cand = X_cand[cand_features]
    # # pred_probs_cand = cand_model.predict_proba(X_cand)[:, 1]
    # # candidates_df['CandidateScore'] = pred_probs_cand
    # # top_candidates = candidates_df.sort_values('CandidateScore', ascending=False).head(initial_candidates)
    
    # # print(f"🚀 Stage 2: Re-ranking top {len(top_candidates)} candidates with ACCURATE model...")
    
    # # # Use enhanced feature engineering that includes ALL peer features
    # # X_rerank, _, available_features = feature_engineering.create_enhanced_reranker_features(
        # # top_candidates.copy(), embeddings_df, graph_dfs
    # # )
    
    
    # # X_rerank = _hard_fail_feature_alignment(X_rerank, rerank_features, model_name="reranker")
    
    # # # DEBUG: Check feature alignment
    # # print(f"🔍 Model expects these features: {len(rerank_features)} total")
    # # calculated_features = list(X_rerank.columns)
    # # print(f"🔍 We calculated these features: {len(calculated_features)} total")

    # # missing_in_model = set(calculated_features) - set(rerank_features)
    # # missing_in_calc = set(rerank_features) - set(calculated_features)

    # # if missing_in_model:
        # # print(f"⚠️ Calculated but not expected by model: {len(missing_in_model)} features")
    # # if missing_in_calc:
        # # print(f"⚠️ Expected by model but not calculated: {missing_in_calc}")
        # # # Add missing features as zeros
        # # for missing_feat in missing_in_calc:
            # # X_rerank[missing_feat] = 0
    
    # # # Ensure column order matches model expectations exactly
    # # X_rerank = X_rerank[rerank_features]
    
    # # print(f"✅ Final feature matrix shape: {X_rerank.shape}")
    
    # # pred_probs_rerank = rerank_model.predict_proba(X_rerank)[:, 1]
    # # top_candidates['FinalScore'] = pred_probs_rerank
    
    # # final_recs = top_candidates.sort_values('FinalScore', ascending=False).head(topN)
    
    # # # Display recommendations with proper error handling
    # # print("\n--- FINAL TOP RECOMMENDATIONS (from Re-Ranker Model) ---")
    
    
    # """Console interface using shared prediction core"""
    # print(f"====== GENERATING RECOMMENDATIONS FOR USER ID: {user_id} ======")
    
    # # Load artifacts using shared function
    # artifacts = prediction_core.load_prediction_artifacts()
    
    # # Run prediction using shared core
    # results = prediction_core.run_prediction_core(user_id, artifacts, topN, initial_candidates)
    
    # if not results:
        # print("❌ No recommendations found")
        # return
    
    # # Display results (console-specific formatting)
    # final_recs = results['predictions']
    # print("\n--- FINAL TOP RECOMMENDATIONS ---")
    
    
    
    # final_recs = final_recs.copy()
    # final_recs['EndpointSystemId'] = final_recs['EntitlementId'].astype(str).str.split('_').str[0].astype('Int64')
    # final_recs['OriginalEntitlementId'] = final_recs['EntitlementId'].astype(str).str.split('_').str[1]
    
    # # Add entitlement names and system info with error handling
    # try:
        # final_recs = final_recs.merge(
            # graph_dfs['entitlements'][['id', 'Name', 'Description']], 
            # left_on='EntitlementId', right_on='id', 
            # how='left', suffixes=('', '_ent')
        # )
        
        # # Use DisplayName from endpoints (as identified earlier)
        # endpoint_cols = ['id', 'ApplicationCode']
        # if 'DisplayName' in graph_dfs['endpoints'].columns:
            # endpoint_cols.append('DisplayName')
        
        # final_recs = final_recs.merge(
            # graph_dfs['endpoints'][endpoint_cols], 
            # left_on='EndpointSystemId', right_on='id', 
            # how='left', suffixes=('', '_sys')
        # )
        
        # # Display recommendations
        # display_cols = ['OriginalEntitlementId', 'Name', 'ApplicationCode', 'FinalScore']
        # if 'DisplayName' in final_recs.columns:
            # display_cols.insert(2, 'DisplayName')
        
        # display_df = final_recs[display_cols].copy()
        # column_names = ['EntitlementID', 'EntitlementName', 'ApplicationCode', 'Score']
        # if 'DisplayName' in final_recs.columns:
            # column_names.insert(2, 'SystemName')
        
        # display_df.columns = column_names
        # print(display_df.round(4))
        
    # except Exception as e:
        # print(f"⚠️ Error in recommendation display: {e}")
        # print("Basic recommendations:")
        # basic_display = final_recs[['EntitlementId', 'FinalScore']].copy()
        # basic_display.columns = ['EntitlementID', 'Score']
        # print(basic_display.round(4))
    
    # # Show ACTUAL peer insights (calculated fresh, not from model features)
    # if not final_recs.empty:
        # print(f"\n--- PEER ADOPTION INSIGHTS FOR TOP RECOMMENDATION ---")
        
        # try:
            # # Calculate actual peer features for this specific prediction
            # top_entitlement_id = final_recs.iloc[0]['EntitlementId']
            
            # test_df = pd.DataFrame({
                # 'UserId': [user_id],
                # 'EntitlementId': [top_entitlement_id]
            # })
            
            # actual_peer_features = feature_engineering.calculate_peer_adoption_features(test_df, graph_dfs)
            
            # if not actual_peer_features.empty:
                # peer_data = actual_peer_features.iloc[0]
                
                # print(f"📊 Close Peers (same role + dept): {peer_data.get('close_peer_adoption_rate', 0):.1%} adoption ({peer_data.get('close_peer_count', 0)} peers)")
                # print(f"👥 Direct Team (same manager): {peer_data.get('direct_team_adoption_rate', 0):.1%} adoption ({peer_data.get('direct_team_count', 0)} team members)")
                # print(f"🏢 Role Peers (same role): {peer_data.get('role_peer_adoption_rate', 0):.1%} adoption ({peer_data.get('role_peer_count', 0)} role peers)")
                # print(f"🏛️ Dept Peers (same dept): {peer_data.get('dept_peer_adoption_rate', 0):.1%} adoption ({peer_data.get('dept_peer_count', 0)} dept peers)")
            # else:
                # print("❌ Could not calculate peer features")
                
        # except Exception as e:
            # print(f"🚨 Error displaying peer insights: {e}")
    
    # # Generate SHAP explanation
    # if not final_recs.empty:
        # print("\n🚀 Generating SHAP explanation for the top recommendation...")
        # try:
            # top_rec_details = final_recs.iloc[0]
            # prediction_features_series = X_rerank.iloc[0]
            # prediction_features_reshaped = prediction_features_series.values.reshape(1, -1)
            
            # explainer = shap.TreeExplainer(rerank_model)
            # shap_values = explainer.shap_values(prediction_features_reshaped)
            
            # if isinstance(shap_values, list):
                # shap_values = shap_values[1]  # For binary classification, take positive class
            
            # shap.initjs()
            # force_plot = shap.force_plot(
                # explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
                # shap_values, 
                # prediction_features_series, 
                # feature_names=rerank_features
            # )
            
            # shap_html_path = f"shap_force_plot_user_{user_id}_ent_{top_rec_details['OriginalEntitlementId']}.html"
            # shap.save_html(shap_html_path, force_plot)
            # print(f"✅ SHAP force plot saved to '{shap_html_path}'.")
            
        # except Exception as e:
            # print(f"⚠️ Could not generate SHAP explanation: {e}")

    # print(f"\n✅ Recommendations complete for User {user_id}")
    
    
    # # # ---- HARD-FAIL FEATURE ALIGNMENT (drop-in) ----
# # def _hard_fail_feature_alignment(X_df, expected_features, model_name="reranker"):
    # # exp = list(expected_features)
    # # act = list(X_df.columns)

    # # missing_in_calc = [f for f in exp if f not in act]
    # # extra_in_calc   = [f for f in act if f not in exp]

    # # print(f"🔍 {model_name} expected features: {len(exp)}")
    # # print(f"🔍 {model_name} calculated features: {len(act)}")

    # # if missing_in_calc or extra_in_calc:
        # # raise RuntimeError(
            # # f"[{model_name.upper()} FEATURE MISMATCH]\n"
            # # f"  • Missing in calculated (expected but not found): {missing_in_calc}\n"
            # # f"  • Extra in calculated (not expected by model):   {extra_in_calc}\n"
            # # f"Resolve by cleaning artifacts & retraining so feature schemas match."
        # # )

    # # # Reorder strictly to the model’s expected order and coerce dtypes to numeric
    # # X_ordered = X_df[exp].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # # print(f"✅ {model_name} feature alignment perfect; using ordered shape: {X_ordered.shape}")
    # # return X_ordered
    # # # ------------------------------------------------


# # if __name__ == '__main__':
    # # parser = argparse.ArgumentParser(description="Generate scalable access recommendations for a given user.")
    # # parser.add_argument("user_id", help="The numeric ID of the user.")
    # # parser.add_argument("--top_n", type=int, default=5, help="Number of top recommendations to show.")
    # # parser.add_argument("--candidates", type=int, default=100, help="Number of initial candidates to generate.")
    # # args = parser.parse_args()
    # # recommend_access_for_user(args.user_id, args.top_n, args.candidates)
    
    
# predict.py - MINIMAL VERSION WITH NO DUPLICATION
# predict.py - Complete version using prediction_core
import argparse
import pandas as pd
from .prediction_core import (
    run_prediction_pipeline, 
    generate_shap_explanation,
    calculate_peer_insights
)

def display_recommendations(results):
    """Display recommendations for console"""
    final_recs = results['predictions']
    graph_dfs = results['artifacts']['graph_dfs']
    
    print("\n--- FINAL TOP RECOMMENDATIONS (from Re-Ranker Model) ---")
    
    final_recs = final_recs.copy()
    final_recs['EndpointSystemId'] = final_recs['EntitlementId'].astype(str).str.split('_').str[0].astype('Int64')
    final_recs['OriginalEntitlementId'] = final_recs['EntitlementId'].astype(str).str.split('_').str[1]
    
    # Add entitlement names and system info
    try:
        final_recs = final_recs.merge(
            graph_dfs['entitlements'][['id', 'Name', 'Description']], 
            left_on='EntitlementId', right_on='id', 
            how='left', suffixes=('', '_ent')
        )
        
        endpoint_cols = ['id', 'ApplicationCode']
        if 'DisplayName' in graph_dfs['endpoints'].columns:
            endpoint_cols.append('DisplayName')
        
        final_recs = final_recs.merge(
            graph_dfs['endpoints'][endpoint_cols], 
            left_on='EndpointSystemId', right_on='id', 
            how='left', suffixes=('', '_sys')
        )
        
        # Display recommendations
        display_cols = ['OriginalEntitlementId', 'Name', 'ApplicationCode', 'FinalScore']
        if 'DisplayName' in final_recs.columns:
            display_cols.insert(2, 'DisplayName')
        
        display_df = final_recs[display_cols].copy()
        column_names = ['EntitlementID', 'EntitlementName', 'ApplicationCode', 'Score']
        if 'DisplayName' in final_recs.columns:
            column_names.insert(2, 'SystemName')
        
        display_df.columns = column_names
        print(display_df.round(4))
        
    except Exception as e:
        print(f"⚠️ Error in recommendation display: {e}")

def display_peer_insights_console(results):
    """Display peer insights for console"""
    final_recs = results['predictions']
    user_id = results['user_id']
    
    if final_recs.empty:
        return
        
    print(f"\n--- PEER ADOPTION INSIGHTS FOR TOP RECOMMENDATION ---")
    
    try:
        top_entitlement_id = final_recs.iloc[0]['EntitlementId']
        # ✅ Use shared function from prediction_core
        peer_insights = calculate_peer_insights(user_id, top_entitlement_id)
        
        if peer_insights:
            print(f"📊 Close Peers: {peer_insights['close_peers']['adoption_rate']:.1%} adoption ({peer_insights['close_peers']['total']} peers)")
            print(f"👥 Direct Team: {peer_insights['direct_team']['adoption_rate']:.1%} adoption ({peer_insights['direct_team']['total']} team members)")
            print(f"🏢 Role Peers: {peer_insights['role_peers']['adoption_rate']:.1%} adoption ({peer_insights['role_peers']['total']} role peers)")
            print(f"🏛️ Dept Peers: {peer_insights['dept_peers']['adoption_rate']:.1%} adoption ({peer_insights['dept_peers']['total']} dept peers)")
        else:
            print("❌ Could not calculate peer features")
            
    except Exception as e:
        print(f"🚨 Error displaying peer insights: {e}")

def recommend_access_for_user(user_id, topN=5, initial_candidates=100):
    """Console interface using shared prediction core"""
    
    print(f"====== GENERATING SCALABLE RECOMMENDATIONS FOR USER ID: {user_id} ======")
    
    try:
        # ✅ Use shared prediction pipeline from prediction_core
        results = run_prediction_pipeline(user_id, topN, initial_candidates)
        
        if not results:
            print("❌ No recommendations found")
            return
        
        # ✅ Display results using console-specific formatting
        display_recommendations(results)
        display_peer_insights_console(results)
        
        # ✅ Use shared SHAP generation from prediction_core
        generate_shap_explanation(results)
        
        print(f"\n✅ Recommendations complete for User {user_id}")
        
    except Exception as e:
        print(f"🚨 Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate scalable access recommendations for a given user.")
    parser.add_argument("user_id", help="The numeric ID of the user.")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top recommendations to show.")
    parser.add_argument("--candidates", type=int, default=100, help="Number of initial candidates to generate.")
    args = parser.parse_args()
    recommend_access_for_user(args.user_id, args.top_n, args.candidates)