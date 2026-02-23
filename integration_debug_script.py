#!/usr/bin/env python3
"""
ML Pipeline Integration Debug Script
Works with your existing ml_pipeline to diagnose confidence issues
"""

import sys
import os
sys.path.append('.')  # Add current directory to path

from ml_pipeline import predict, feature_engineering, config
import pandas as pd
import numpy as np
import joblib

def debug_confidence_with_pipeline(user_id, topN=5, debug_mode=True):
    """
    Debug confidence issues using your actual ML pipeline with detailed logging
    """
    
    print("=" * 80)
    print(f"🔍 DEBUGGING CONFIDENCE FOR USER {user_id} WITH ACTUAL PIPELINE")
    print("=" * 80)
    
    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        print(f"❌ Error: User ID must be integer, got '{user_id}'")
        return
    
    # Step 1: Load all artifacts with validation
    print("\n📦 STEP 1: Loading Pipeline Artifacts")
    print("-" * 50)
    
    artifacts = load_pipeline_artifacts()
    if not artifacts:
        return
    
    # Step 2: Validate user exists
    print(f"\n👤 STEP 2: Validating User {user_id}")
    print("-" * 50)
    
    user_validation = validate_user(user_id, artifacts)
    if not user_validation['exists']:
        return
    
    # Step 3: Debug candidate generation
    print(f"\n🎯 STEP 3: Debugging Candidate Generation")
    print("-" * 50)
    
    candidate_debug = debug_candidate_stage(user_id, artifacts)
    
    # Step 4: Debug reranking  
    print(f"\n🔄 STEP 4: Debugging Reranking Stage")
    print("-" * 50)
    
    reranking_debug = debug_reranking_stage(user_id, artifacts, candidate_debug)
    
    # Step 5: Compare with expected values
    print(f"\n📊 STEP 5: Confidence Analysis")
    print("-" * 50)
    
    analyze_confidence_distribution(candidate_debug, reranking_debug)
    
    # Step 6: Generate specific recommendations
    print(f"\n💡 STEP 6: Specific Recommendations")
    print("-" * 50)
    
    generate_specific_recommendations(user_validation, candidate_debug, reranking_debug)

def load_pipeline_artifacts():
    """Load artifacts using your actual pipeline config"""
    
    try:
        print(f"📁 Loading from: {config.ARTIFACT_DIR}")
        
        artifacts = {
            'candidate_model': joblib.load(os.path.join(config.ARTIFACT_DIR, 'candidate_model.joblib')),
            'reranker_model': joblib.load(os.path.join(config.ARTIFACT_DIR, 'reranker_model.joblib')),
            'candidate_features': joblib.load(os.path.join(config.ARTIFACT_DIR, 'candidate_model_features.joblib')),
            'reranker_features': joblib.load(os.path.join(config.ARTIFACT_DIR, 'reranker_model_features.joblib')),
            'embeddings_df': pd.read_pickle(os.path.join(config.ARTIFACT_DIR, 'embeddings.pkl')),
            'graph_dfs': {
                'users': pd.read_pickle(os.path.join(config.ARTIFACT_DIR, 'users.pkl')),
                'entitlements': pd.read_pickle(os.path.join(config.ARTIFACT_DIR, 'entitlements.pkl')),
                'entrecon': pd.read_pickle(os.path.join(config.ARTIFACT_DIR, 'entrecon.pkl')),
                'orgs': pd.read_pickle(os.path.join(config.ARTIFACT_DIR, 'orgs.pkl')),
                'endpoints': pd.read_pickle(os.path.join(config.ARTIFACT_DIR, 'endpoints.pkl')),
                'designations': pd.read_pickle(os.path.join(config.ARTIFACT_DIR, 'designations.pkl'))
            }
        }
        
        print("✅ All artifacts loaded successfully")
        
        # Print artifact summary
        print(f"   📊 Data Summary:")
        print(f"      Users: {len(artifacts['graph_dfs']['users']):,}")
        print(f"      Entitlements: {len(artifacts['graph_dfs']['entitlements']):,}")
        print(f"      Access Records: {len(artifacts['graph_dfs']['entrecon']):,}")
        print(f"      Embeddings: {len(artifacts['embeddings_df']):,}")
        print(f"      Candidate Features: {len(artifacts['candidate_features'])}")
        print(f"      Reranker Features: {len(artifacts['reranker_features'])}")
        
        return artifacts
        
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        print("   Please ensure you've run: python -m ml_pipeline.train")
        return None

def validate_user(user_id, artifacts):
    """Validate user exists and get profile"""
    
    users_df = artifacts['graph_dfs']['users']
    entrecon_df = artifacts['graph_dfs']['entrecon']
    embeddings_df = artifacts['embeddings_df']
    
    # Check if user exists
    user_row = users_df[users_df['id'] == user_id]
    
    validation = {
        'exists': not user_row.empty,
        'user_id': user_id
    }
    
    if validation['exists']:
        user = user_row.iloc[0]
        
        validation.update({
            'role_id': user.get('NBusinessRoleId'),
            'org_id': user.get('NOrganisationId'),
            'manager_id': user.get('ManagerId'),
            'is_active': user.get('IsActive'),
        })
        
        # Get current access
        user_access = entrecon_df[entrecon_df['UserId'] == user_id]
        validation['current_access_count'] = len(user_access)
        validation['current_entitlements'] = user_access['EntitlementId'].tolist()
        
        # Check embedding
        user_embedding = embeddings_df[embeddings_df['originalId'] == user_id]
        validation['has_embedding'] = not user_embedding.empty
        
        print(f"✅ User {user_id} found:")
        print(f"   Role: {validation['role_id']}")
        print(f"   Organization: {validation['org_id']}")
        print(f"   Manager: {validation['manager_id']}")
        print(f"   Active: {validation['is_active']}")
        print(f"   Current Access: {validation['current_access_count']} entitlements")
        print(f"   Has Embedding: {validation['has_embedding']}")
        
        if not validation['has_embedding']:
            print(f"❌ CRITICAL: User {user_id} has no embedding!")
        
    else:
        print(f"❌ User {user_id} not found in users table!")
    
    return validation

def debug_candidate_stage(user_id, artifacts):
    """Debug the candidate generation stage"""
    
    # Get all entitlements user doesn't have
    entrecon_df = artifacts['graph_dfs']['entrecon']
    entitlements_df = artifacts['graph_dfs']['entitlements']
    
    user_current_access = set(entrecon_df[entrecon_df['UserId'] == user_id]['EntitlementId'])
    all_entitlements = set(entitlements_df['id'])
    candidate_entitlements = all_entitlements - user_current_access
    
    print(f"🎯 Candidate Pool Analysis:")
    print(f"   Total entitlements: {len(all_entitlements):,}")
    print(f"   User current access: {len(user_current_access):,}")
    print(f"   Candidate pool: {len(candidate_entitlements):,}")
    
    # Create candidate dataframe (first 10 for testing)
    test_candidates = list(candidate_entitlements)[:10]
    candidates_df = pd.DataFrame({
        'UserId': [user_id] * len(test_candidates),
        'EntitlementId': test_candidates
    })
    
    print(f"   Testing with {len(test_candidates)} candidates")
    
    # Test feature engineering
    print(f"\n🔧 Testing Candidate Feature Engineering:")
    
    try:
        X_cand, _, _ = feature_engineering.create_candidate_model_features(
            candidates_df.copy(), 
            artifacts['embeddings_df']
        )
        
        print(f"   ✅ Feature engineering successful")
        print(f"   Generated features: {X_cand.shape[1]}")
        print(f"   Expected features: {len(artifacts['candidate_features'])}")
        
        # Check feature alignment
        missing_features = set(artifacts['candidate_features']) - set(X_cand.columns)
        extra_features = set(X_cand.columns) - set(artifacts['candidate_features'])
        
        if missing_features:
            print(f"   ⚠️ Missing features: {missing_features}")
        if extra_features:
            print(f"   ⚠️ Extra features: {extra_features}")
        
        # Add missing features as zeros
        for feat in missing_features:
            X_cand[feat] = 0
        
        # Align features
        X_cand = X_cand[artifacts['candidate_features']]
        
        print(f"\n🎯 Testing Candidate Model Predictions:")
        
        # Get predictions
        pred_probs = artifacts['candidate_model'].predict_proba(X_cand)[:, 1]
        
        print(f"   Prediction range: [{np.min(pred_probs):.6f}, {np.max(pred_probs):.6f}]")
        print(f"   Mean prediction: {np.mean(pred_probs):.6f}")
        print(f"   Std prediction: {np.std(pred_probs):.6f}")
        
        # Show individual predictions
        for i, (ent_id, prob) in enumerate(zip(test_candidates[:5], pred_probs[:5])):
            print(f"   Candidate {i+1} (Ent {ent_id}): {prob:.6f}")
        
        # Check if any features have extreme values
        print(f"\n🔍 Feature Value Analysis:")
        feature_stats = X_cand.describe()
        
        # Find features with unusual values
        for col in X_cand.columns[:10]:  # Check first 10 features
            col_min, col_max = feature_stats.loc['min', col], feature_stats.loc['max', col]
            col_mean, col_std = feature_stats.loc['mean', col], feature_stats.loc['std', col]
            
            if abs(col_mean) > 1000 or col_std > 1000:
                print(f"   ⚠️ {col}: mean={col_mean:.3f}, std={col_std:.3f} (unusual scale)")
        
        return {
            'success': True,
            'feature_matrix': X_cand,
            'predictions': pred_probs,
            'candidates': test_candidates,
            'missing_features': missing_features,
            'stats': {
                'min_pred': np.min(pred_probs),
                'max_pred': np.max(pred_probs),
                'mean_pred': np.mean(pred_probs),
                'std_pred': np.std(pred_probs)
            }
        }
        
    except Exception as e:
        print(f"   ❌ Error in candidate stage: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def debug_reranking_stage(user_id, artifacts, candidate_debug):
    """Debug the reranking stage with peer features"""
    
    if not candidate_debug['success']:
        print("❌ Skipping reranking debug due to candidate stage failure")
        return {'success': False}
    
    # Take top candidates from candidate stage
    top_n = 5
    candidate_scores = candidate_debug['predictions']
    top_indices = np.argsort(candidate_scores)[-top_n:][::-1]
    
    top_candidates_df = pd.DataFrame({
        'UserId': [user_id] * top_n,
        'EntitlementId': [candidate_debug['candidates'][i] for i in top_indices],
        'CandidateScore': [candidate_scores[i] for i in top_indices]
    })
    
    print(f"🔄 Reranking Top {top_n} Candidates:")
    for i, row in top_candidates_df.iterrows():
        print(f"   {i+1}. Ent {row['EntitlementId']}: {row['CandidateScore']:.6f}")
    
    print(f"\n🔧 Testing Enhanced Feature Engineering:")
    
    try:
        # Test the enhanced reranker features
        X_rerank, _, available_features = feature_engineering.create_enhanced_reranker_features(
            top_candidates_df.copy(), 
            artifacts['embeddings_df'], 
            artifacts['graph_dfs']
        )
        
        print(f"   ✅ Enhanced feature engineering successful")
        print(f"   Generated features: {X_rerank.shape[1]}")
        print(f"   Expected features: {len(artifacts['reranker_features'])}")
        
        # Check for peer features specifically
        peer_features_generated = [f for f in X_rerank.columns if 'peer' in f.lower()]
        peer_features_expected = [f for f in artifacts['reranker_features'] if 'peer' in f.lower()]
        
        print(f"   Peer features generated: {len(peer_features_generated)}")
        print(f"   Peer features expected: {len(peer_features_expected)}")
        print(f"   Peer features: {peer_features_generated}")
        
        # Check feature alignment
        missing_features = set(artifacts['reranker_features']) - set(X_rerank.columns)
        extra_features = set(X_rerank.columns) - set(artifacts['reranker_features'])
        
        if missing_features:
            print(f"   ⚠️ Missing features: {missing_features}")
            # Add missing as zeros
            for feat in missing_features:
                X_rerank[feat] = 0
        
        if extra_features:
            print(f"   ⚠️ Extra features: {extra_features}")
        
        # Align features exactly
        X_rerank = X_rerank[artifacts['reranker_features']]
        
        print(f"\n🎯 Testing Reranker Model Predictions:")
        
        # Get reranker predictions
        rerank_probs = artifacts['reranker_model'].predict_proba(X_rerank)[:, 1]
        
        print(f"   Prediction range: [{np.min(rerank_probs):.6f}, {np.max(rerank_probs):.6f}]")
        print(f"   Mean prediction: {np.mean(rerank_probs):.6f}")
        print(f"   Std prediction: {np.std(rerank_probs):.6f}")
        
        # Show final rankings
        print(f"\n🏆 Final Rankings:")
        final_df = top_candidates_df.copy()
        final_df['FinalScore'] = rerank_probs
        final_df = final_df.sort_values('FinalScore', ascending=False)
        
        for i, row in final_df.iterrows():
            print(f"   {i+1}. Ent {row['EntitlementId']}: {row['FinalScore']:.6f} (was {row['CandidateScore']:.6f})")
        
        # Analyze peer features for top prediction
        print(f"\n👥 Peer Analysis for Top Prediction:")
        analyze_peer_features(user_id, final_df.iloc[0]['EntitlementId'], X_rerank.iloc[0], artifacts)
        
        return {
            'success': True,
            'feature_matrix': X_rerank,
            'predictions': rerank_probs,
            'final_rankings': final_df,
            'missing_features': missing_features,
            'peer_features': peer_features_generated,
            'stats': {
                'min_pred': np.min(rerank_probs),
                'max_pred': np.max(rerank_probs),
                'mean_pred': np.mean(rerank_probs),
                'std_pred': np.std(rerank_probs)
            }
        }
        
    except Exception as e:
        print(f"   ❌ Error in reranking stage: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def analyze_peer_features(user_id, entitlement_id, feature_row, artifacts):
    """Analyze peer features for a specific user-entitlement pair"""
    
    # Get user context
    users_df = artifacts['graph_dfs']['users']
    user_info = users_df[users_df['id'] == user_id].iloc[0]
    
    print(f"   User: {user_id}, Entitlement: {entitlement_id}")
    print(f"   User Role: {user_info.get('NBusinessRoleId')}")
    print(f"   User Org: {user_info.get('NOrganisationId')}")
    print(f"   User Manager: {user_info.get('ManagerId')}")
    
    # Extract peer features from the feature row
    peer_feature_names = [col for col in feature_row.index if 'peer' in col.lower()]
    
    for feat_name in peer_feature_names:
        feat_value = feature_row[feat_name]
        if 'adoption_rate' in feat_name:
            print(f"   {feat_name}: {feat_value:.3f} ({feat_value*100:.1f}%)")
        elif 'count' in feat_name:
            print(f"   {feat_name}: {feat_value:.0f}")
        else:
            print(f"   {feat_name}: {feat_value:.6f}")

def analyze_confidence_distribution(candidate_debug, reranking_debug):
    """Analyze confidence distribution and compare with expected ranges"""
    
    print("📊 CONFIDENCE DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    if candidate_debug['success']:
        cand_stats = candidate_debug['stats']
        print(f"🎯 Candidate Stage:")
        print(f"   Range: [{cand_stats['min_pred']:.6f}, {cand_stats['max_pred']:.6f}]")
        print(f"   Mean: {cand_stats['mean_pred']:.6f}")
        print(f"   Expected range: [0.01, 0.99] for healthy model")
        
        if cand_stats['max_pred'] < 0.01:
            print(f"   ❌ All candidate predictions extremely low!")
        elif cand_stats['min_pred'] > 0.99:
            print(f"   ❌ All candidate predictions extremely high!")
        else:
            print(f"   ✅ Candidate predictions in reasonable range")
    
    if reranking_debug['success']:
        rerank_stats = reranking_debug['stats']
        print(f"\n🔄 Reranking Stage:")
        print(f"   Range: [{rerank_stats['min_pred']:.6f}, {rerank_stats['max_pred']:.6f}]")
        print(f"   Mean: {rerank_stats['mean_pred']:.6f}")
        print(f"   Expected range: [0.01, 0.99] for healthy model")
        
        if rerank_stats['max_pred'] < 0.01:
            print(f"   ❌ All reranker predictions extremely low!")
            print(f"   🔍 This is your confidence issue!")
        elif rerank_stats['min_pred'] > 0.99:
            print(f"   ❌ All reranker predictions extremely high!")
        else:
            print(f"   ✅ Reranker predictions in reasonable range")
        
        # Compare stages
        if candidate_debug['success']:
            print(f"\n📈 Stage Comparison:")
            cand_mean = cand_stats['mean_pred']
            rerank_mean = rerank_stats['mean_pred']
            
            if rerank_mean < cand_mean * 0.1:
                print(f"   ❌ Reranker drastically reducing confidence!")
                print(f"   📉 {cand_mean:.6f} → {rerank_mean:.6f} (reduction factor: {cand_mean/rerank_mean:.1f}x)")
            else:
                print(f"   ✅ Normal confidence flow between stages")

def generate_specific_recommendations(user_validation, candidate_debug, reranking_debug):
    """Generate specific recommendations based on debug results"""
    
    print("💡 SPECIFIC RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = []
    critical_issues = []
    
    # User-level issues
    if not user_validation.get('has_embedding'):
        critical_issues.append("User has no embedding - predictions impossible")
        recommendations.append("Re-run embedding generation to include this user")
    
    if user_validation.get('current_access_count', 0) == 0:
        recommendations.append("User has no current access - consider cold-start strategy")
    
    # Candidate stage issues
    if candidate_debug.get('success'):
        if candidate_debug['stats']['max_pred'] < 0.001:
            critical_issues.append("Candidate model producing extremely low probabilities")
            recommendations.append("Check candidate model training - may need recalibration")
        
        if candidate_debug.get('missing_features'):
            critical_issues.append(f"Missing candidate features: {candidate_debug['missing_features']}")
            recommendations.append("Ensure feature engineering produces all expected features")
    
    # Reranking stage issues  
    if reranking_debug.get('success'):
        if reranking_debug['stats']['max_pred'] < 0.001:
            critical_issues.append("Reranker model producing extremely low probabilities")
            recommendations.append("Check reranker model training and calibration")
        
        if reranking_debug.get('missing_features'):
            critical_issues.append(f"Missing reranker features: {reranking_debug['missing_features']}")
            recommendations.append("Debug peer feature calculation")
        
        if len(reranking_debug.get('peer_features', [])) < 4:
            critical_issues.append("Not all peer features being generated")
            recommendations.append("Debug peer adoption rate calculations")
    
    # Print issues
    if critical_issues:
        print("🚨 CRITICAL ISSUES:")
        for i, issue in enumerate(critical_issues, 1):
            print(f"   {i}. {issue}")
    
    # Print recommendations
    if recommendations:
        print("\n🔧 IMMEDIATE ACTIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Advanced debugging
    print("\n🔬 ADVANCED DEBUGGING:")
    print("   1. Compare this user's features with training data distribution")
    print("   2. Check if model was trained on similar organizational structure")
    print("   3. Validate embedding similarity calculations")
    print("   4. Consider model recalibration with Platt scaling")
    print("   5. Test with different user IDs to isolate user-specific vs systemic issues")

def test_actual_prediction_pipeline(user_id):
    """Test the actual prediction pipeline to compare results"""
    
    print("\n🧪 TESTING ACTUAL PREDICTION PIPELINE")
    print("=" * 50)
    
    try:
        # Run your actual prediction function
        print(f"Running predict.recommend_access_for_user({user_id})...")
        
        # This would call your actual function - uncomment when ready:
        # result = predict.recommend_access_for_user(user_id, topN=5)
        # print(f"Actual pipeline results: {result}")
        
        print("   (Actual pipeline test commented out - uncomment to run)")
        
    except Exception as e:
        print(f"   ❌ Error running actual pipeline: {e}")

def inspect_model_internals(artifacts):
    """Inspect model internals for calibration issues"""
    
    print("\n🔬 MODEL INTERNALS INSPECTION")
    print("=" * 50)
    
    candidate_model = artifacts['candidate_model']
    reranker_model = artifacts['reranker_model']
    
    print("🎯 Candidate Model:")
    print(f"   Type: {type(candidate_model).__name__}")
    
    # Check if it's a tree-based model
    if hasattr(candidate_model, 'feature_importances_'):
        importances = candidate_model.feature_importances_
        top_features = np.argsort(importances)[-5:][::-1]
        feature_names = artifacts['candidate_features']
        
        print("   Top 5 Features:")
        for i, feat_idx in enumerate(top_features):
            print(f"     {i+1}. {feature_names[feat_idx]}: {importances[feat_idx]:.4f}")
    
    print("\n🔄 Reranker Model:")
    print(f"   Type: {type(reranker_model).__name__}")
    
    if hasattr(reranker_model, 'feature_importances_'):
        importances = reranker_model.feature_importances_
        top_features = np.argsort(importances)[-5:][::-1]
        feature_names = artifacts['reranker_features']
        
        print("   Top 5 Features:")
        for i, feat_idx in enumerate(top_features):
            print(f"     {i+1}. {feature_names[feat_idx]}: {importances[feat_idx]:.4f}")

def run_full_debug():
    """Run full debug workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug confidence issues with ML pipeline")
    parser.add_argument("user_id", type=int, help="User ID to debug")
    parser.add_argument("--topn", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--test-pipeline", action="store_true", help="Test actual prediction pipeline")
    parser.add_argument("--inspect-models", action="store_true", help="Inspect model internals")
    
    args = parser.parse_args()
    
    # Run main debug
    debug_confidence_with_pipeline(args.user_id, args.topn)
    
    # Optional additional tests
    if args.test_pipeline:
        test_actual_prediction_pipeline(args.user_id)
    
    if args.inspect_models:
        artifacts = load_pipeline_artifacts()
        if artifacts:
            inspect_model_internals(artifacts)

if __name__ == "__main__":
    run_full_debug()