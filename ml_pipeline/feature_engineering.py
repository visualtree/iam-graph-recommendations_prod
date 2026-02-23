# # ml_pipeline/feature_engineering.py (COMPLETE VERSION WITH ETL TYPE FIXES)

# import pandas as pd
# import numpy as np
# from . import config
# import time

# def _expand_embeddings(emb_df: pd.DataFrame, id_col_name: str, prefix: str) -> pd.DataFrame:
    # """
    # emb_df: DataFrame with columns ['originalId', 'embedding'] where 'embedding' is list[float]
    # Returns: DataFrame with [id_col_name, f'{prefix}0', f'{prefix}1', ...]
    # """
    # if emb_df is None or emb_df.empty:
        # return pd.DataFrame(columns=[id_col_name])

    # # Remove duplicates at the embedding level (safety check)
    # original_count = len(emb_df)
    # emb_df = emb_df.drop_duplicates(subset=['originalId'])
    # deduped_count = len(emb_df)
    # if original_count != deduped_count:
        # print(f"⚠️ Removed {original_count - deduped_count} duplicate embeddings in _expand_embeddings")

    # # convert each list to a float ndarray
    # emb_series = emb_df['embedding'].apply(lambda v: np.asarray(v, dtype=float))
    # emb_mat = np.vstack(emb_series.values)  # shape: (n, d)
    # cols = [f'{prefix}{i}' for i in range(emb_mat.shape[1])]

    # # Ensure clean index alignment
    # result_df = pd.concat(
        # [
            # emb_df[['originalId']].rename(columns={'originalId': id_col_name}).reset_index(drop=True),
            # pd.DataFrame(emb_mat, columns=cols).reset_index(drop=True)
        # ],
        # axis=1
    # )
    
    # return result_df

# def create_candidate_model_features(labeled_df: pd.DataFrame, embeddings_df: pd.DataFrame):
    # """Creates lightweight features for the candidate generation model (Stage 1) - TYPE SAFE VERSION."""
    # print("🚀 Creating candidate model features (optimized for speed)...")

    # # Check input data quality
    # original_labeled_count = len(labeled_df)
    # duplicate_pairs = labeled_df.duplicated(subset=['UserId', 'EntitlementId']).sum()
    # if duplicate_pairs > 0:
        # print(f"⚠️ WARNING: Found {duplicate_pairs} duplicate user-entitlement pairs in labeled_df")
        # labeled_df = labeled_df.drop_duplicates(subset=['UserId', 'EntitlementId']).reset_index(drop=True)
        # print(f"🔧 Cleaned labeled_df from {original_labeled_count} to {len(labeled_df)} rows")

    # # Deduplicate embeddings (safety check)
    # original_emb_count = len(embeddings_df)
    # embeddings_df = embeddings_df.drop_duplicates(subset=['originalId']).reset_index(drop=True)
    # deduped_emb_count = len(embeddings_df)
    # if original_emb_count != deduped_emb_count:
        # print(f"🔧 Removed {original_emb_count - deduped_emb_count} duplicate embeddings")

    # # pick just the rows we need
    # user_raw = embeddings_df[embeddings_df['originalId'].isin(labeled_df['UserId'])].copy()
    # ent_raw = embeddings_df[embeddings_df['originalId'].isin(labeled_df['EntitlementId'])].copy()

    # # expand list embeddings -> numeric columns
    # user_embeddings = _expand_embeddings(user_raw, 'UserId', 'user_emb_')
    # ent_embeddings = _expand_embeddings(ent_raw, 'EntitlementId', 'ent_emb_')

    # # Ensure labeled_df has clean index before merging
    # labeled_df = labeled_df.reset_index(drop=True)

    # # merge into labeled pairs
    # features_df = labeled_df.merge(user_embeddings, on='UserId', how='left')
    # features_df = features_df.merge(ent_embeddings, on='EntitlementId', how='left')

    # # Verify row count preservation
    # if len(features_df) != len(labeled_df):
        # print(f"🚨 ERROR: Row count changed from {len(labeled_df)} to {len(features_df)} during merging")
    
    # user_emb_cols = [c for c in features_df.columns if c.startswith('user_emb_')]
    # ent_emb_cols = [c for c in features_df.columns if c.startswith('ent_emb_')]

    # if user_emb_cols and ent_emb_cols:
        # user_emb_matrix = features_df[user_emb_cols].fillna(0.0).to_numpy(dtype=float)
        # ent_emb_matrix = features_df[ent_emb_cols].fillna(0.0).to_numpy(dtype=float)

        # # normalize (safe divide)
        # user_norms = np.linalg.norm(user_emb_matrix, axis=1, keepdims=True)
        # ent_norms = np.linalg.norm(ent_emb_matrix, axis=1, keepdims=True)
        # user_unit = np.divide(user_emb_matrix, user_norms, out=np.zeros_like(user_emb_matrix), where=(user_norms != 0))
        # ent_unit = np.divide(ent_emb_matrix, ent_norms, out=np.zeros_like(ent_emb_matrix), where=(ent_norms != 0))

        # # features
        # features_df['embedding_cosine_similarity'] = np.sum(user_unit * ent_unit, axis=1)
        # features_df['embedding_euclidean_distance'] = np.linalg.norm(user_emb_matrix - ent_emb_matrix, axis=1)

    # # EndpointSystemId from composite entitlement key "{endpoint}_{entId}"
    # if 'EntitlementId' in features_df.columns:
        # features_df['EndpointSystemId'] = (
            # features_df['EntitlementId'].astype(str).str.split('_').str[0].astype('Int64')
        # )

    # # final feature list
    # feature_cols = [
        # 'embedding_cosine_similarity',
        # 'embedding_euclidean_distance',
        # 'EndpointSystemId',
        # # keep a small slice of raw dims for signal
        # *user_emb_cols[:20],
        # *ent_emb_cols[:20],
    # ]
    # feature_cols = [c for c in feature_cols if c in features_df.columns]

    # X = features_df[feature_cols].fillna(0.0)
    # y = features_df['HasAccess'] if 'HasAccess' in features_df.columns else None

    # print(f"✅ Candidate features shape: {X.shape}")
    # return X, y, feature_cols


# def create_reranker_model_features(labeled_df, embeddings_df, graph_dfs):
    # """Creates comprehensive features for the reranker model (Stage 2) - TYPE SAFE VERSION."""
    # print("🚀 Creating reranker model features (comprehensive analysis)...")
    
    # # Convert embedding lists to numpy arrays if needed
    # if isinstance(embeddings_df['embedding'].iloc[0], list):
        # embeddings_df['embedding'] = embeddings_df['embedding'].apply(lambda x: np.array(x) if isinstance(x, list) else x)
    
    # # Start with candidate features
    # X_candidate, _, candidate_features = create_candidate_model_features(labeled_df.copy(), embeddings_df)
    
    # # CRITICAL: Verify row count alignment
    # if len(X_candidate) != len(labeled_df):
        # print(f"🚨 ERROR: Candidate features ({len(X_candidate)}) != labeled_df ({len(labeled_df)})")
        # min_rows = min(len(X_candidate), len(labeled_df))
        # X_candidate = X_candidate.iloc[:min_rows]
        # labeled_df = labeled_df.iloc[:min_rows]
        # print(f"🔧 Truncated both to {min_rows} rows for alignment")
    
    # # Ensure clean index alignment before concat
    # features_df = pd.concat([
        # labeled_df.reset_index(drop=True), 
        # X_candidate.reset_index(drop=True)
    # ], axis=1)
    
    # # Add user context features - Now type-safe due to ETL fixes
    # if 'users' in graph_dfs and len(graph_dfs['users']) > 0:
        # user_data = graph_dfs['users'].set_index('id')
        # features_df = features_df.merge(
            # user_data[['NOrganisationId', 'NBusinessRoleId', 'ManagerId']], 
            # left_on='UserId', right_index=True, how='left'
        # )
    
    # # Extract original entitlement ID from composite key for joining
    # features_df['OriginalEntitlementId'] = features_df['EntitlementId'].astype(str).str.split('_').str[1].astype('Int64')
    
    # # Add entitlement context features - Now type-safe due to ETL fixes
    # if 'entitlements' in graph_dfs and len(graph_dfs['entitlements']) > 0:
        # entitlement_data = graph_dfs['entitlements'].copy()
        # entitlement_data['OriginalEntitlementId'] = entitlement_data['id'].astype(str).str.split('_').str[1].astype('Int64')
        # entitlement_data['EndpointSystemId'] = entitlement_data['id'].astype(str).str.split('_').str[0].astype('int64')
        
        # features_df = features_df.merge(
            # entitlement_data[['OriginalEntitlementId', 'EndpointSystemId', 'Name']], 
            # on=['OriginalEntitlementId', 'EndpointSystemId'], how='left', suffixes=('', '_ent')
        # )
    
    # # Role-based features - Now type-safe due to ETL fixes
    # if 'entrecon' in graph_dfs and len(graph_dfs['entrecon']) > 0:
        # # Users in same role with this entitlement
        # role_ent_stats = graph_dfs['entrecon'].merge(
            # graph_dfs['users'][['id', 'NBusinessRoleId']], left_on='UserId', right_on='id'
        # ).groupby(['NBusinessRoleId', 'EntitlementId']).size().reset_index(name='role_entitlement_count')
        
        # features_df = features_df.merge(
            # role_ent_stats, 
            # left_on=['NBusinessRoleId', 'EntitlementId'], 
            # right_on=['NBusinessRoleId', 'EntitlementId'], how='left'
        # )
        # features_df['role_entitlement_count'] = features_df['role_entitlement_count'].fillna(0)
        
        # # Manager team features - Now type-safe due to ETL fixes
        # manager_ent_stats = graph_dfs['entrecon'].merge(
            # graph_dfs['users'][['id', 'ManagerId']], left_on='UserId', right_on='id'
        # ).groupby(['ManagerId', 'EntitlementId']).size().reset_index(name='manager_team_entitlement_count')
        
        # features_df = features_df.merge(
            # manager_ent_stats, 
            # left_on=['ManagerId', 'EntitlementId'], 
            # right_on=['ManagerId', 'EntitlementId'], how='left'
        # )
        # features_df['manager_team_entitlement_count'] = features_df['manager_team_entitlement_count'].fillna(0)
        
        # # User's current entitlement count
        # user_ent_counts = graph_dfs['entrecon'].groupby('UserId').size().reset_index(name='user_total_entitlements')
        # features_df = features_df.merge(user_ent_counts, on='UserId', how='left')
        # features_df['user_total_entitlements'] = features_df['user_total_entitlements'].fillna(0)
    
    # # Organization features - Now type-safe due to ETL fixes
    # if 'NOrganisationId' in features_df.columns and 'entrecon' in graph_dfs:
        # org_ent_stats = graph_dfs['entrecon'].merge(
            # graph_dfs['users'][['id', 'NOrganisationId']], left_on='UserId', right_on='id'
        # ).groupby(['NOrganisationId', 'EntitlementId']).size().reset_index(name='org_entitlement_count')
        
        # features_df = features_df.merge(
            # org_ent_stats, 
            # left_on=['NOrganisationId', 'EntitlementId'], 
            # right_on=['NOrganisationId', 'EntitlementId'], how='left'
        # )
        # features_df['org_entitlement_count'] = features_df['org_entitlement_count'].fillna(0)
    
    # # Select numerical features for the model
    # numerical_features = [
        # 'embedding_cosine_similarity', 'embedding_euclidean_distance',
        # 'role_entitlement_count', 'manager_team_entitlement_count', 
        # 'user_total_entitlements', 'org_entitlement_count',
        # 'EndpointSystemId', 'NOrganisationId', 'NBusinessRoleId'
    # ]
    
    # # Add embedding features from candidate model
    # numerical_features.extend([col for col in candidate_features if col.startswith(('user_emb_', 'ent_emb_'))])
    
    # # Select features that exist
    # existing_features = [col for col in numerical_features if col in features_df.columns]
    
    # X = features_df[existing_features].fillna(0)
    # y = features_df['HasAccess'] if 'HasAccess' in features_df.columns else None
    
    # print(f"✅ Reranker features shape: {X.shape}")
    # return X, y, existing_features


# def calculate_adoption_rate(peer_group_df, entitlement_id, entrecon_df):
    # """Helper function to calculate adoption rate for a peer group - TYPE SAFE VERSION"""
    
    # peer_count = len(peer_group_df)
    
    # if peer_count == 0:
        # return {
            # 'adoption_rate': 0,
            # 'peer_count': 0,
            # 'peers_with_access': 0
        # }
    
    # # TYPE SAFE: Now that ETL fixes data types, this should work properly
    # peer_ids = peer_group_df['id'].tolist()
    
    # # Ensure entitlement_id is string (matches entrecon data type from ETL)
    # entitlement_id_str = str(entitlement_id)
    
    # # Type-safe filtering - now works because ETL standardized types
    # peer_access = entrecon_df[
        # (entrecon_df['UserId'].isin(peer_ids)) &
        # (entrecon_df['EntitlementId'] == entitlement_id_str)
    # ]
    
    # peers_with_access = len(peer_access)
    # adoption_rate = peers_with_access / peer_count if peer_count > 0 else 0
    
    # return {
        # 'adoption_rate': adoption_rate,
        # 'peer_count': peer_count,
        # 'peers_with_access': peers_with_access
    # }


# def calculate_peer_adoption_features(labeled_df, graph_dfs):
    # """Calculate the 4 peer adoption rate features - TYPE SAFE VERSION"""
    # print("  📊 Calculating peer adoption rates...")
    
    # peer_features = []
    
    # for idx, row in labeled_df.iterrows():
        # user_id = row['UserId']
        # entitlement_id = row['EntitlementId']
        
        # # Get user context
        # user_info = graph_dfs['users'][graph_dfs['users']['id'] == user_id]
        # if user_info.empty:
            # # Handle missing user
            # peer_features.append({
                # 'UserId': user_id,
                # 'EntitlementId': entitlement_id,
                # 'close_peer_adoption_rate': 0,
                # 'direct_team_adoption_rate': 0,
                # 'role_peer_adoption_rate': 0,
                # 'dept_peer_adoption_rate': 0,
                # 'close_peer_count': 0,
                # 'direct_team_count': 0,
                # 'role_peer_count': 0,
                # 'dept_peer_count': 0
            # })
            # continue
            
        # user_role = user_info.iloc[0]['NBusinessRoleId']
        # user_org = user_info.iloc[0]['NOrganisationId'] 
        # user_manager = user_info.iloc[0]['ManagerId']
        
        # # 1. CLOSE PEERS (Same role + same department)
        # close_peers = graph_dfs['users'][
            # (graph_dfs['users']['NBusinessRoleId'] == user_role) & 
            # (graph_dfs['users']['NOrganisationId'] == user_org) &
            # (graph_dfs['users']['id'] != user_id) &
            # (graph_dfs['users']['IsActive'] == True)
        # ]
        # close_stats = calculate_adoption_rate(close_peers, entitlement_id, graph_dfs['entrecon'])
        
        # # 2. DIRECT TEAM (Same manager) - FIXED: Now works due to ETL type fixes
        # # if pd.notna(user_manager):
            # # direct_team = graph_dfs['users'][
                # # (graph_dfs['users']['ManagerId'] == user_manager) & 
                # # (graph_dfs['users']['id'] != user_id) &
                # # (graph_dfs['users']['IsActive'] == True)
            # # ]
            # # direct_team_stats = calculate_adoption_rate(direct_team, entitlement_id, graph_dfs['entrecon'])
        # # else:
            # # direct_team_stats = {'adoption_rate': 0, 'peer_count': 0, 'peers_with_access': 0}
            
        # # Calculate DIRECT TEAM adoption (the real calculation)
        # team_adoption_rate = 0
        # team_count = 0
        
        # if pd.notna(user_manager):
            # direct_team = graph_dfs['users'][
                # (graph_dfs['users']['ManagerId'] == user_manager) & 
                # (graph_dfs['users']['id'] != user_id) &
                # (graph_dfs['users']['IsActive'] == True)
            # ]
            
            # if len(direct_team) > 0:
                # team_ids = direct_team['id'].tolist()
                # team_access = graph_dfs['entrecon'][
                    # (graph_dfs['entrecon']['UserId'].isin(team_ids)) &
                    # (graph_dfs['entrecon']['EntitlementId'] == entitlement_id)
                # ]
                # team_adoption_rate = len(team_access) / len(direct_team)
                # team_count = len(direct_team)
        
        # # 3. ROLE PEERS (Same role, any department)
        # role_peers = graph_dfs['users'][
            # (graph_dfs['users']['NBusinessRoleId'] == user_role) & 
            # (graph_dfs['users']['id'] != user_id) &
            # (graph_dfs['users']['IsActive'] == True)
        # ]
        # role_stats = calculate_adoption_rate(role_peers, entitlement_id, graph_dfs['entrecon'])
        
        # # 4. DEPARTMENT PEERS (Same department, any role)
        # dept_peers = graph_dfs['users'][
            # (graph_dfs['users']['NOrganisationId'] == user_org) & 
            # (graph_dfs['users']['id'] != user_id) &
            # (graph_dfs['users']['IsActive'] == True)
        # ]
        # dept_stats = calculate_adoption_rate(dept_peers, entitlement_id, graph_dfs['entrecon'])
        
        # peer_features.append({
            # 'UserId': user_id,
            # 'EntitlementId': entitlement_id,
            # 'close_peer_adoption_rate': close_stats['adoption_rate'],
            # 'direct_team_adoption_rate': team_adoption_rate,
            # 'role_peer_adoption_rate': role_stats['adoption_rate'],
            # 'dept_peer_adoption_rate': dept_stats['adoption_rate'],
            # 'close_peer_count': close_stats['peer_count'],
            # 'direct_team_count': team_count,
            # 'role_peer_count': role_stats['peer_count'],
            # 'dept_peer_count': dept_stats['peer_count']
        # })
    
    # return pd.DataFrame(peer_features)


# def calculate_peer_adoption_features_optimized(labeled_df, graph_dfs):
    # """Calculate the 4 peer adoption rate features - OPTIMIZED VERSION (should take <1 minute)"""
    # print("  📊 Calculating peer adoption rates (optimized)...")
    
    # start_time = time.time()
    
    # # Pre-process data for faster lookups
    # users_df = graph_dfs['users'].set_index('id')
    # entrecon_df = graph_dfs['entrecon']
    
    # # Create lookup dictionaries for O(1) access
    # user_to_role = users_df['NBusinessRoleId'].to_dict()
    # user_to_org = users_df['NOrganisationId'].to_dict()
    # user_to_manager = users_df['ManagerId'].to_dict()
    
    # print(f"   🔧 Pre-processing completed in {time.time() - start_time:.2f}s")
    
    # # Pre-calculate peer groups (vectorized operations)
    # print("   🔧 Pre-calculating peer groups...")
    # peer_start = time.time()
    
    # # Group users by role, org, and manager for faster lookups
    # role_groups = users_df.groupby('NBusinessRoleId')['id'].apply(list).to_dict()
    # org_groups = users_df.groupby('NOrganisationId')['id'].apply(list).to_dict()
    # manager_groups = users_df.groupby('ManagerId')['id'].apply(list).to_dict()
    
    # # Pre-calculate entitlement adoption by groups
    # entrecon_counts = {}
    # for ent_id in labeled_df['EntitlementId'].unique():
        # users_with_ent = set(entrecon_df[entrecon_df['EntitlementId'] == ent_id]['UserId'])
        # entrecon_counts[ent_id] = users_with_ent
    
    # print(f"   🔧 Peer groups calculated in {time.time() - peer_start:.2f}s")
    
    # # Vectorized calculation for all labeled pairs
    # print("   🔧 Computing adoption rates...")
    # calc_start = time.time()
    
    # peer_features = []
    
    # # Process in batches for better performance
    # batch_size = 1000
    # total_rows = len(labeled_df)
    
    # for batch_start in range(0, total_rows, batch_size):
        # batch_end = min(batch_start + batch_size, total_rows)
        # batch_df = labeled_df.iloc[batch_start:batch_end]
        
        # batch_features = []
        
        # for _, row in batch_df.iterrows():
            # user_id = row['UserId']
            # entitlement_id = row['EntitlementId']
            
            # # Get user context (O(1) lookups)
            # user_role = user_to_role.get(user_id)
            # user_org = user_to_org.get(user_id)
            # user_manager = user_to_manager.get(user_id)
            
            # # Get users who have this entitlement (pre-calculated)
            # users_with_entitlement = entrecon_counts.get(entitlement_id, set())
            
            # # Calculate peer groups and adoption rates efficiently
            
            # # 1. Close peers (same role + same org)
            # close_peers = set()
            # if pd.notna(user_role) and pd.notna(user_org):
                # role_users = set(role_groups.get(user_role, []))
                # org_users = set(org_groups.get(user_org, []))
                # close_peers = (role_users & org_users) - {user_id}
            
            # close_with_access = close_peers & users_with_entitlement
            # close_adoption_rate = len(close_with_access) / len(close_peers) if close_peers else 0
            
            # # 2. Direct team (same manager)
            # direct_team = set()
            # if pd.notna(user_manager):
                # direct_team = set(manager_groups.get(user_manager, [])) - {user_id}
            
            # team_with_access = direct_team & users_with_entitlement
            # team_adoption_rate = len(team_with_access) / len(direct_team) if direct_team else 0
            
            # # 3. Role peers (same role)
            # role_peers = set()
            # if pd.notna(user_role):
                # role_peers = set(role_groups.get(user_role, [])) - {user_id}
            
            # role_with_access = role_peers & users_with_entitlement
            # role_adoption_rate = len(role_with_access) / len(role_peers) if role_peers else 0
            
            # # 4. Department peers (same org)
            # dept_peers = set()
            # if pd.notna(user_org):
                # dept_peers = set(org_groups.get(user_org, [])) - {user_id}
            
            # dept_with_access = dept_peers & users_with_entitlement
            # dept_adoption_rate = len(dept_with_access) / len(dept_peers) if dept_peers else 0
            
            # batch_features.append({
                # 'UserId': user_id,
                # 'EntitlementId': entitlement_id,
                # 'close_peer_adoption_rate': close_adoption_rate,
                # 'direct_team_adoption_rate': team_adoption_rate,
                # 'role_peer_adoption_rate': role_adoption_rate,
                # 'dept_peer_adoption_rate': dept_adoption_rate,
                # 'close_peer_count': len(close_peers),
                # 'direct_team_count': len(direct_team),
                # 'role_peer_count': len(role_peers),
                # 'dept_peer_count': len(dept_peers)
            # })
        
        # peer_features.extend(batch_features)
        
        # # Progress update
        # if batch_start % (batch_size * 10) == 0:
            # progress = (batch_end / total_rows) * 100
            # elapsed = time.time() - calc_start
            # print(f"   📊 Progress: {progress:.1f}% ({batch_end}/{total_rows}) - {elapsed:.1f}s elapsed")
    
    # print(f"   🔧 Adoption rates calculated in {time.time() - calc_start:.2f}s")
    
    # total_time = time.time() - start_time
    # print(f"✅ Peer adoption features completed in {total_time:.2f}s (was taking 15+ minutes)")
    
    # return pd.DataFrame(peer_features)

# def create_enhanced_reranker_features(labeled_df, embeddings_df, graph_dfs):
    # """Enhanced version of reranker features that includes peer adoption rates - TYPE SAFE VERSION"""
    # print("🚀 Creating enhanced reranker features with peer adoption rates...")
    
    # # Start with existing reranker features
    # X_original, y_original, original_features = create_reranker_model_features(labeled_df, embeddings_df, graph_dfs)
    
    # # Add peer adoption rate features
    # peer_features_df = calculate_peer_adoption_features(labeled_df, graph_dfs)
     # # Add peer adoption rate features - OPTIMIZED
    # #peer_features_df = calculate_peer_adoption_features_optimized(labeled_df, graph_dfs)  # ← Changed function call
    
    # #peer_features_df = calculate_peer_adoption_features_complete(labeled_df, graph_dfs) 
    
    
    # # Merge original features with peer features
    # #'direct_team_adoption_rate', 
    # features_df = pd.concat([
        # labeled_df.reset_index(drop=True),
        # X_original.reset_index(drop=True),
        # peer_features_df[['close_peer_adoption_rate','direct_team_adoption_rate',
                         # 'role_peer_adoption_rate', 'dept_peer_adoption_rate',
                         # 'close_peer_count', 'direct_team_count',
                         # 'role_peer_count', 'dept_peer_count']].reset_index(drop=True)
    # ], axis=1)
    
    # # Enhanced feature list
    # enhanced_features = original_features + [
        # 'close_peer_adoption_rate',     # 🎯 Same role + same dept
        # 'direct_team_adoption_rate',    # 👥 Same manager  
        # 'role_peer_adoption_rate',      # 🏢 Same role
        # 'dept_peer_adoption_rate',      # 🏛️ Same department
        # 'close_peer_count',
        # 'direct_team_count',
        # 'role_peer_count', 
        # 'dept_peer_count'
    # ]
    
    # # Select features that exist
    # existing_enhanced_features = [col for col in enhanced_features if col in features_df.columns]
    
    # X_enhanced = features_df[existing_enhanced_features].fillna(0)
    # y_enhanced = features_df['HasAccess'] if 'HasAccess' in features_df.columns else y_original
    
    # print(f"✅ Enhanced reranker features shape: {X_enhanced.shape}")
    # print(f"🆕 New peer features added: {[f for f in existing_enhanced_features if 'peer' in f]}")
    
    # return X_enhanced, y_enhanced, existing_enhanced_features
    
# # Add this debug version to your feature_engineering.py
# def calculate_peer_adoption_features_debug(labeled_df, graph_dfs):
    # """Debug version to find the team adoption bug"""
    # print("  📊 Calculating peer adoption rates (debug mode)...")
    
    # # For debugging, let's focus on just Jonathan's case
    # jonathan_rows = labeled_df[labeled_df['UserId'] == 348]
    # if jonathan_rows.empty:
        # print("❌ Jonathan (348) not found in labeled_df")
        # return pd.DataFrame()
    
    # # Get Jonathan's entitlement - use the first one for debugging
    # test_row = jonathan_rows.iloc[0]
    # user_id = test_row['UserId']
    # entitlement_id = test_row['EntitlementId']
    
    # print(f"🔍 DEBUG - User: {user_id}, Entitlement: {entitlement_id}")
    
    # # Check user context
    # users_df = graph_dfs['users']
    # user_info = users_df[users_df['id'] == user_id]
    
    # if user_info.empty:
        # print(f"❌ User {user_id} not found in users_df")
        # return pd.DataFrame()
    
    # user_manager = user_info.iloc[0]['ManagerId']
    # print(f"🔍 DEBUG - User's manager: {user_manager}")
    
    # # Check team members
    # if pd.notna(user_manager):
        # team_members = users_df[
            # (users_df['ManagerId'] == user_manager) & 
            # (users_df['id'] != user_id) &
            # (users_df['IsActive'] == True)
        # ]
        # print(f"🔍 DEBUG - Team members found: {len(team_members)}")
        # print(f"🔍 DEBUG - Sample team members: {team_members['UserName'].head(3).tolist()}")
        
        # # Check access to this entitlement
        # entrecon_df = graph_dfs['entrecon']
        # team_ids = team_members['id'].tolist()
        # team_access = entrecon_df[
            # (entrecon_df['UserId'].isin(team_ids)) &
            # (entrecon_df['EntitlementId'] == entitlement_id)
        # ]
        
        # print(f"🔍 DEBUG - Team members with access to {entitlement_id}: {len(team_access)}")
        # print(f"🔍 DEBUG - Expected adoption rate: {len(team_access) / len(team_members) * 100:.1f}%")
        
        # if len(team_access) > 0:
            # team_access_users = team_access['UserId'].tolist()
            # team_access_names = users_df[users_df['id'].isin(team_access_users)]['UserName'].tolist()
            # print(f"🔍 DEBUG - Team members with access: {team_access_names[:3]}")
    # else:
        # print(f"❌ User has no manager assigned")
    
    # # Return minimal result for now
    # return pd.DataFrame([{
        # 'UserId': user_id,
        # 'EntitlementId': entitlement_id,
        # 'direct_team_adoption_rate': 0.999,  # Dummy value to test
        # 'direct_team_count': 99,
        # 'close_peer_adoption_rate': 0,
        # 'role_peer_adoption_rate': 0,
        # 'dept_peer_adoption_rate': 0,
        # 'close_peer_count': 0,
        # 'role_peer_count': 0,
        # 'dept_peer_count': 0
    # }])


# def calculate_peer_adoption_features_fixed(labeled_df, graph_dfs):
    # """Fixed version that matches the trained model expectations"""
    # print("  📊 Calculating peer adoption rates (matching model expectations)...")
    
    # peer_features = []
    
    # for idx, row in labeled_df.iterrows():
        # user_id = row['UserId']
        # entitlement_id = row['EntitlementId']
        
        # # Get user context
        # user_info = graph_dfs['users'][graph_dfs['users']['id'] == user_id]
        # if user_info.empty:
            # peer_features.append({
                # 'UserId': user_id,
                # 'EntitlementId': entitlement_id,
                # 'close_peer_adoption_rate': 0,
                # 'role_peer_adoption_rate': 0,
                # 'dept_peer_adoption_rate': 0,
                # 'close_peer_count': 0,
                # 'role_peer_count': 0,
                # 'dept_peer_count': 0
            # })
            # continue
            
        # user_role = user_info.iloc[0]['NBusinessRoleId']
        # user_org = user_info.iloc[0]['NOrganisationId'] 
        # user_manager = user_info.iloc[0]['ManagerId']
        
        # # Calculate DIRECT TEAM adoption (the one we care about)
        # # But put it in the CLOSE_PEER field since that's what the model expects
        # team_adoption_rate = 0
        # team_count = 0
        
        # if pd.notna(user_manager):
            # direct_team = graph_dfs['users'][
                # (graph_dfs['users']['ManagerId'] == user_manager) & 
                # (graph_dfs['users']['id'] != user_id) &
                # (graph_dfs['users']['IsActive'] == True)
            # ]
            
            # if len(direct_team) > 0:
                # team_ids = direct_team['id'].tolist()
                # team_access = graph_dfs['entrecon'][
                    # (graph_dfs['entrecon']['UserId'].isin(team_ids)) &
                    # (graph_dfs['entrecon']['EntitlementId'] == entitlement_id)
                # ]
                # team_adoption_rate = len(team_access) / len(direct_team)
                # team_count = len(direct_team)
        
        # # HACK: Put the REAL team adoption in the close_peer field since that's what displays first
        # peer_features.append({
            # 'UserId': user_id,
            # 'EntitlementId': entitlement_id,
            # 'close_peer_adoption_rate': team_adoption_rate,  # REAL team data here
            # 'role_peer_adoption_rate': 0,
            # 'dept_peer_adoption_rate': 0,
            # 'close_peer_count': team_count,  # REAL team count here
            # 'role_peer_count': 0,
            # 'dept_peer_count': 0
        # })
    
    # return pd.DataFrame(peer_features)

# def calculate_peer_adoption_features_complete(labeled_df, graph_dfs):
    # """Complete version with all 8 peer features that the model expects"""
    # print("  📊 Calculating peer adoption rates (complete 8 features)...")
    
    # peer_features = []
    
    # for idx, row in labeled_df.iterrows():
        # user_id = row['UserId']
        # entitlement_id = row['EntitlementId']
        
        # # Get user context
        # user_info = graph_dfs['users'][graph_dfs['users']['id'] == user_id]
        # if user_info.empty:
            # peer_features.append({
                # 'UserId': user_id,
                # 'EntitlementId': entitlement_id,
                # 'close_peer_adoption_rate': 0,
                # 'direct_team_adoption_rate': 0,  # ADD THIS
                # 'role_peer_adoption_rate': 0,
                # 'dept_peer_adoption_rate': 0,
                # 'close_peer_count': 0,
                # 'direct_team_count': 0,  # ADD THIS
                # 'role_peer_count': 0,
                # 'dept_peer_count': 0
            # })
            # continue
            
        # user_role = user_info.iloc[0]['NBusinessRoleId']
        # user_org = user_info.iloc[0]['NOrganisationId'] 
        # user_manager = user_info.iloc[0]['ManagerId']
        
        # # Calculate DIRECT TEAM adoption (the real calculation)
        # team_adoption_rate = 0
        # team_count = 0
        
        # if pd.notna(user_manager):
            # direct_team = graph_dfs['users'][
                # (graph_dfs['users']['ManagerId'] == user_manager) & 
                # (graph_dfs['users']['id'] != user_id) &
                # (graph_dfs['users']['IsActive'] == True)
            # ]
            
            # if len(direct_team) > 0:
                # team_ids = direct_team['id'].tolist()
                # team_access = graph_dfs['entrecon'][
                    # (graph_dfs['entrecon']['UserId'].isin(team_ids)) &
                    # (graph_dfs['entrecon']['EntitlementId'] == entitlement_id)
                # ]
                # team_adoption_rate = len(team_access) / len(direct_team)
                # team_count = len(direct_team)
        # # print(f"  📊 team_adoption_rate={team_adoption_rate}")
        # # print(f"  📊 team_count={team_count}")
        # # Return ALL 8 features the model expects
        # # peer_features.append({
            # # 'UserId': user_id,
            # # 'EntitlementId': entitlement_id,
            # # 'close_peer_adoption_rate': 0,
            # # 'direct_team_adoption_rate': team_adoption_rate,  # REAL value here
            # # 'role_peer_adoption_rate': 0,
            # # 'dept_peer_adoption_rate': 0,
            # # 'close_peer_count': 0,
            # # 'direct_team_count': team_count,  # REAL value here
            # # 'role_peer_count': 0,
            # # 'dept_peer_count': 0
        # # })
        
        # peer_features.append({
            # 'UserId': user_id,
            # 'EntitlementId': entitlement_id,
            # 'close_peer_adoption_rate': close_stats['adoption_rate'],
            # 'direct_team_adoption_rate': team_adoption_rate,
            # 'role_peer_adoption_rate': role_stats['adoption_rate'],
            # 'dept_peer_adoption_rate': dept_stats['adoption_rate'],
            # 'close_peer_count': close_stats['peer_count'],
            # 'direct_team_count': team_count,
            # 'role_peer_count': role_stats['peer_count'],
            # 'dept_peer_count': dept_stats['peer_count']
        # })
    
    # return pd.DataFrame(peer_features)

# # def calculate_peer_adoption_features_fixed(labeled_df, graph_dfs):
    # # """Fixed version that matches the trained model expectations"""
    # # print("  📊 Calculating peer adoption rates (matching model expectations)...")
    
    # # peer_features = []
    
    # # for idx, row in labeled_df.iterrows():
        # # user_id = row['UserId']
        # # entitlement_id = row['EntitlementId']
        
        # # # Get user context
        # # user_info = graph_dfs['users'][graph_dfs['users']['id'] == user_id]
        # # if user_info.empty:
            # # peer_features.append({
                # # 'UserId': user_id,
                # # 'EntitlementId': entitlement_id,
                # # 'close_peer_adoption_rate': 0,
                # # 'role_peer_adoption_rate': 0,
                # # 'dept_peer_adoption_rate': 0,
                # # 'close_peer_count': 0,
                # # 'role_peer_count': 0,
                # # 'dept_peer_count': 0
            # # })
            # # continue
            
        # # user_role = user_info.iloc[0]['NBusinessRoleId']
        # # user_org = user_info.iloc[0]['NOrganisationId'] 
        # # user_manager = user_info.iloc[0]['ManagerId']
        
        # # # Calculate DIRECT TEAM adoption (the one we care about)
        # # # But put it in the CLOSE_PEER field since that's what the model expects
        # # team_adoption_rate = 0
        # # team_count = 0
        
        # # if pd.notna(user_manager):
            # # direct_team = graph_dfs['users'][
                # # (graph_dfs['users']['ManagerId'] == user_manager) & 
                # # (graph_dfs['users']['id'] != user_id) &
                # # (graph_dfs['users']['IsActive'] == True)
            # # ]
            
            # # if len(direct_team) > 0:
                # # team_ids = direct_team['id'].tolist()
                # # team_access = graph_dfs['entrecon'][
                    # # (graph_dfs['entrecon']['UserId'].isin(team_ids)) &
                    # # (graph_dfs['entrecon']['EntitlementId'] == entitlement_id)
                # # ]
                # # team_adoption_rate = len(team_access) / len(direct_team)
                # # team_count = len(direct_team)
        
        # # # HACK: Put the REAL team adoption in the close_peer field since that's what displays first
        # # peer_features.append({
            # # 'UserId': user_id,
            # # 'EntitlementId': entitlement_id,
            # # 'close_peer_adoption_rate': team_adoption_rate,  # REAL team data here
            # # 'role_peer_adoption_rate': 0,
            # # 'dept_peer_adoption_rate': 0,
            # # 'close_peer_count': team_count,  # REAL team count here
            # # 'role_peer_count': 0,
            # # 'dept_peer_count': 0
        # # })
    
    # # return pd.DataFrame(peer_features)

# def calculate_peer_adoption_features_final_debug(labeled_df, graph_dfs):
    # """Final debug to find the exact issue"""
    # print("  📊 Calculating peer adoption rates (final debug)...")
    
    # peer_features = []
    
    # # Focus on first few rows for debugging
    # debug_rows = labeled_df.head(5)
    
    # for idx, row in debug_rows.iterrows():
        # user_id = row['UserId']
        # entitlement_id = row['EntitlementId']
        
        # print(f"\n🔍 PROCESSING: User {user_id}, Entitlement {entitlement_id}")
        
        # # Get user context
        # user_info = graph_dfs['users'][graph_dfs['users']['id'] == user_id]
        # if user_info.empty:
            # print(f"❌ User {user_id} not found!")
            # continue
            
        # user_role = user_info.iloc[0]['NBusinessRoleId']
        # user_org = user_info.iloc[0]['NOrganisationId'] 
        # user_manager = user_info.iloc[0]['ManagerId']
        
        # print(f"   User context - Role: {user_role}, Org: {user_org}, Manager: {user_manager}")
        
        # # 2. DIRECT TEAM (Same manager)
        # team_adoption_rate = 0
        # team_count = 0
        
        # if pd.notna(user_manager):
            # direct_team = graph_dfs['users'][
                # (graph_dfs['users']['ManagerId'] == user_manager) & 
                # (graph_dfs['users']['id'] != user_id) &
                # (graph_dfs['users']['IsActive'] == True)
            # ]
            
            # print(f"   Team members found: {len(direct_team)}")
            
            # if len(direct_team) > 0:
                # team_ids = direct_team['id'].tolist()
                # print(f"   Team IDs: {team_ids[:3]}...")
                
                # team_access = graph_dfs['entrecon'][
                    # (graph_dfs['entrecon']['UserId'].isin(team_ids)) &
                    # (graph_dfs['entrecon']['EntitlementId'] == entitlement_id)
                # ]
                
                # print(f"   Team members with access to {entitlement_id}: {len(team_access)}")
                
                # if len(team_access) > 0:
                    # print(f"   Team access user IDs: {team_access['UserId'].tolist()}")
                
                # team_adoption_rate = len(team_access) / len(direct_team)
                # team_count = len(direct_team)
                
                # print(f"   ✅ Team adoption rate: {team_adoption_rate:.3f} ({len(team_access)}/{len(direct_team)})")
            # else:
                # print(f"   ⚠️ No team members found")
        # else:
            # print(f"   ⚠️ User has no manager")
        
        # # Store the result
        # peer_data = {
            # 'UserId': user_id,
            # 'EntitlementId': entitlement_id,
            # 'close_peer_adoption_rate': 0.123,  # Dummy non-zero for testing
            # 'direct_team_adoption_rate': team_adoption_rate,
            # 'role_peer_adoption_rate': 0.456,   # Dummy non-zero for testing
            # 'dept_peer_adoption_rate': 0.789,   # Dummy non-zero for testing
            # 'close_peer_count': 10,
            # 'direct_team_count': team_count,
            # 'role_peer_count': 20,
            # 'dept_peer_count': 30
        # }
        
        # peer_features.append(peer_data)
        # print(f"   📊 Stored: team_rate={team_adoption_rate:.3f}, team_count={team_count}")
    
    # # Fill remaining rows with zeros for now
    # remaining_rows = labeled_df.iloc[5:]
    # for idx, row in remaining_rows.iterrows():
        # peer_features.append({
            # 'UserId': row['UserId'],
            # 'EntitlementId': row['EntitlementId'],
            # 'close_peer_adoption_rate': 0,
            # 'direct_team_adoption_rate': 0,
            # 'role_peer_adoption_rate': 0,
            # 'dept_peer_adoption_rate': 0,
            # 'close_peer_count': 0,
            # 'direct_team_count': 0,
            # 'role_peer_count': 0,
            # 'dept_peer_count': 0
        # })
    
    # result_df = pd.DataFrame(peer_features)
    
    # print(f"\n📊 FINAL PEER FEATURES SUMMARY:")
    # print(f"   Total rows: {len(result_df)}")
    # print(f"   First 5 direct_team_adoption_rate values: {result_df['direct_team_adoption_rate'].head(5).tolist()}")
    # print(f"   First 5 direct_team_count values: {result_df['direct_team_count'].head(5).tolist()}")
    
    # return result_df



# CLEANED feature_engineering.py - READY FOR RETRAINING

import logging
import pandas as pd
import numpy as np
from . import config
import time

logger = logging.getLogger(__name__)

# def _expand_embeddings(emb_df: pd.DataFrame, id_col_name: str, prefix: str) -> pd.DataFrame:
    # """
    # emb_df: DataFrame with columns ['originalId', 'embedding'] where 'embedding' is list[float]
    # Returns: DataFrame with [id_col_name, f'{prefix}0', f'{prefix}1', ...]
    # """
    # if emb_df is None or emb_df.empty:
        # return pd.DataFrame(columns=[id_col_name])

    # # Remove duplicates at the embedding level (safety check)
    # original_count = len(emb_df)
    # emb_df = emb_df.drop_duplicates(subset=['originalId'])
    # deduped_count = len(emb_df)
    # if original_count != deduped_count:
        # print(f"⚠️ Removed {original_count - deduped_count} duplicate embeddings in _expand_embeddings")

    # # convert each list to a float ndarray
    # emb_series = emb_df['embedding'].apply(lambda v: np.asarray(v, dtype=float))
    # emb_mat = np.vstack(emb_series.values)  # shape: (n, d)
    # cols = [f'{prefix}{i}' for i in range(emb_mat.shape[1])]

    # # Ensure clean index alignment
    # result_df = pd.concat(
        # [
            # emb_df[['originalId']].rename(columns={'originalId': id_col_name}).reset_index(drop=True),
            # pd.DataFrame(emb_mat, columns=cols).reset_index(drop=True)
        # ],
        # axis=1
    # )
    
    # return result_df
    
    
# In your feature_engineering.py, modify _expand_embeddings() to add debug:
def _expand_embeddings(emb_df: pd.DataFrame, id_col_name: str, prefix: str) -> pd.DataFrame:
    if emb_df is None or emb_df.empty:
        return pd.DataFrame(columns=[id_col_name])



    # Remove duplicates
    original_count = len(emb_df)
    emb_df = emb_df.drop_duplicates(subset=['originalId'])
    deduped_count = len(emb_df)
    if original_count != deduped_count:
        logger.warning("Removed %d duplicate embeddings", original_count - deduped_count)

    # Convert each list to a float ndarray
    emb_series = emb_df['embedding'].apply(lambda v: np.asarray(v, dtype=float))
    emb_mat = np.vstack(emb_series.values)  # shape: (n, d)

    cols = [f'{prefix}{i}' for i in range(emb_mat.shape[1])]
    logger.debug("Embedding matrix shape: %s, columns: %s...", emb_mat.shape, cols[:5])

    # Ensure clean index alignment
    result_df = pd.concat([
        emb_df[['originalId']].rename(columns={'originalId': id_col_name}).reset_index(drop=True),
        pd.DataFrame(emb_mat, columns=cols).reset_index(drop=True)
    ], axis=1)

    logger.debug("Expanded embeddings result shape: %s", result_df.shape)
    return result_df

def create_candidate_model_features(labeled_df: pd.DataFrame, embeddings_df: pd.DataFrame):
    """Creates lightweight features for the candidate generation model (Stage 1) - TYPE SAFE VERSION."""
    logger.info("Creating candidate model features...")

    # Check input data quality
    original_labeled_count = len(labeled_df)
    duplicate_pairs = labeled_df.duplicated(subset=['UserId', 'EntitlementId']).sum()
    if duplicate_pairs > 0:
        logger.warning("Found %d duplicate user-entitlement pairs; deduplicating", duplicate_pairs)
        labeled_df = labeled_df.drop_duplicates(subset=['UserId', 'EntitlementId']).reset_index(drop=True)
        logger.info("Cleaned labeled_df from %d to %d rows", original_labeled_count, len(labeled_df))

    # Deduplicate embeddings (safety check)
    original_emb_count = len(embeddings_df)
    embeddings_df = embeddings_df.drop_duplicates(subset=['originalId']).reset_index(drop=True)
    deduped_emb_count = len(embeddings_df)
    if original_emb_count != deduped_emb_count:
        logger.info("Removed %d duplicate embeddings", original_emb_count - deduped_emb_count)

    # pick just the rows we need
    user_raw = embeddings_df[embeddings_df['originalId'].isin(labeled_df['UserId'])].copy()
    ent_raw = embeddings_df[embeddings_df['originalId'].isin(labeled_df['EntitlementId'])].copy()
    
    logger.debug(
        "User embedding lookup: need %d unique IDs, found %d embeddings",
        len(labeled_df['UserId'].unique()), len(user_raw),
    )
    logger.debug(
        "Entitlement embedding lookup: need %d unique IDs, found %d embeddings",
        len(labeled_df['EntitlementId'].unique()), len(ent_raw),
    )

    # expand list embeddings -> numeric columns
    user_embeddings = _expand_embeddings(user_raw, 'UserId', 'user_emb_')
    ent_embeddings = _expand_embeddings(ent_raw, 'EntitlementId', 'ent_emb_')

    # Ensure labeled_df has clean index before merging
    labeled_df = labeled_df.reset_index(drop=True)

    # merge into labeled pairs
    features_df = labeled_df.merge(user_embeddings, on='UserId', how='left')
    features_df = features_df.merge(ent_embeddings, on='EntitlementId', how='left')

    # Verify row count preservation
    if len(features_df) != len(labeled_df):
        logger.error("Row count changed from %d to %d during embedding merge", len(labeled_df), len(features_df))
    
    user_emb_cols = [c for c in features_df.columns if c.startswith('user_emb_')]
    ent_emb_cols = [c for c in features_df.columns if c.startswith('ent_emb_')]

    if user_emb_cols and ent_emb_cols:
        user_emb_matrix = features_df[user_emb_cols].fillna(0.0).to_numpy(dtype=float)
        ent_emb_matrix = features_df[ent_emb_cols].fillna(0.0).to_numpy(dtype=float)

        # normalize (safe divide)
        user_norms = np.linalg.norm(user_emb_matrix, axis=1, keepdims=True)
        ent_norms = np.linalg.norm(ent_emb_matrix, axis=1, keepdims=True)
        user_unit = np.divide(user_emb_matrix, user_norms, out=np.zeros_like(user_emb_matrix), where=(user_norms != 0))
        ent_unit = np.divide(ent_emb_matrix, ent_norms, out=np.zeros_like(ent_emb_matrix), where=(ent_norms != 0))

        # features
        features_df['embedding_cosine_similarity'] = np.sum(user_unit * ent_unit, axis=1)
        features_df['embedding_euclidean_distance'] = np.linalg.norm(user_emb_matrix - ent_emb_matrix, axis=1)

    # EndpointSystemId from composite entitlement key "{endpoint}_{entId}"
    if 'EntitlementId' in features_df.columns:
        features_df['EndpointSystemId'] = (
            features_df['EntitlementId'].astype(str).str.split('_').str[0].astype('Int64')
        )

    # final feature list
    feature_cols = [
        'embedding_cosine_similarity',
        'embedding_euclidean_distance',
        'EndpointSystemId',
        # keep a small slice of raw dims for signal
        *user_emb_cols[:20],
        *ent_emb_cols[:20],
    ]
    feature_cols = [c for c in feature_cols if c in features_df.columns]

    X = features_df[feature_cols].fillna(0.0)
    y = features_df['HasAccess'] if 'HasAccess' in features_df.columns else None

    logger.info("Candidate features shape: %s", X.shape)
    return X, y, feature_cols

def create_reranker_model_features(labeled_df, embeddings_df, graph_dfs):
    """Creates comprehensive features for the reranker model (Stage 2) - TYPE SAFE VERSION."""
    logger.info("Creating reranker model features...")
    
    # Convert embedding lists to numpy arrays if needed
    if isinstance(embeddings_df['embedding'].iloc[0], list):
        embeddings_df['embedding'] = embeddings_df['embedding'].apply(lambda x: np.array(x) if isinstance(x, list) else x)
    
    # Start with candidate features
    X_candidate, _, candidate_features = create_candidate_model_features(labeled_df.copy(), embeddings_df)
    
    # CRITICAL: Verify row count alignment
    if len(X_candidate) != len(labeled_df):
        logger.error("Row count mismatch: candidate features=%d, labeled_df=%d", len(X_candidate), len(labeled_df))
        min_rows = min(len(X_candidate), len(labeled_df))
        X_candidate = X_candidate.iloc[:min_rows]
        labeled_df = labeled_df.iloc[:min_rows]
        logger.warning("Truncated both to %d rows for alignment", min_rows)
    
    # Ensure clean index alignment before concat
    features_df = pd.concat([
        labeled_df.reset_index(drop=True), 
        X_candidate.reset_index(drop=True)
    ], axis=1)
    
    # Add user context features
    if 'users' in graph_dfs and len(graph_dfs['users']) > 0:
        user_data = graph_dfs['users'].set_index('id')
        features_df = features_df.merge(
            user_data[['NOrganisationId', 'NBusinessRoleId', 'ManagerId']], 
            left_on='UserId', right_index=True, how='left'
        )
    
    # Extract original entitlement ID from composite key for joining
    features_df['OriginalEntitlementId'] = features_df['EntitlementId'].astype(str).str.split('_').str[1].astype('Int64')
    
    # Add entitlement context features
    if 'entitlements' in graph_dfs and len(graph_dfs['entitlements']) > 0:
        entitlement_data = graph_dfs['entitlements'].copy()
        entitlement_data['OriginalEntitlementId'] = entitlement_data['id'].astype(str).str.split('_').str[1].astype('Int64')
        entitlement_data['EndpointSystemId'] = entitlement_data['id'].astype(str).str.split('_').str[0].astype('int64')
        
        features_df = features_df.merge(
            entitlement_data[['OriginalEntitlementId', 'EndpointSystemId', 'Name']], 
            on=['OriginalEntitlementId', 'EndpointSystemId'], how='left', suffixes=('', '_ent')
        )
    
    # Role-based features
    if 'entrecon' in graph_dfs and len(graph_dfs['entrecon']) > 0:
        # Users in same role with this entitlement
        role_ent_stats = graph_dfs['entrecon'].merge(
            graph_dfs['users'][['id', 'NBusinessRoleId']], left_on='UserId', right_on='id'
        ).groupby(['NBusinessRoleId', 'EntitlementId']).size().reset_index(name='role_entitlement_count')
        
        features_df = features_df.merge(
            role_ent_stats, 
            left_on=['NBusinessRoleId', 'EntitlementId'], 
            right_on=['NBusinessRoleId', 'EntitlementId'], how='left'
        )
        features_df['role_entitlement_count'] = features_df['role_entitlement_count'].fillna(0)
        
        # Manager team features
        manager_ent_stats = graph_dfs['entrecon'].merge(
            graph_dfs['users'][['id', 'ManagerId']], left_on='UserId', right_on='id'
        ).groupby(['ManagerId', 'EntitlementId']).size().reset_index(name='manager_team_entitlement_count')
        
        features_df = features_df.merge(
            manager_ent_stats, 
            left_on=['ManagerId', 'EntitlementId'], 
            right_on=['ManagerId', 'EntitlementId'], how='left'
        )
        features_df['manager_team_entitlement_count'] = features_df['manager_team_entitlement_count'].fillna(0)
        
        # User's current entitlement count
        user_ent_counts = graph_dfs['entrecon'].groupby('UserId').size().reset_index(name='user_total_entitlements')
        features_df = features_df.merge(user_ent_counts, on='UserId', how='left')
        features_df['user_total_entitlements'] = features_df['user_total_entitlements'].fillna(0)
    
    # Organization features
    if 'NOrganisationId' in features_df.columns and 'entrecon' in graph_dfs:
        org_ent_stats = graph_dfs['entrecon'].merge(
            graph_dfs['users'][['id', 'NOrganisationId']], left_on='UserId', right_on='id'
        ).groupby(['NOrganisationId', 'EntitlementId']).size().reset_index(name='org_entitlement_count')
        
        features_df = features_df.merge(
            org_ent_stats, 
            left_on=['NOrganisationId', 'EntitlementId'], 
            right_on=['NOrganisationId', 'EntitlementId'], how='left'
        )
        features_df['org_entitlement_count'] = features_df['org_entitlement_count'].fillna(0)
    
    # Select numerical features for the model
    numerical_features = [
        'embedding_cosine_similarity', 'embedding_euclidean_distance',
        'role_entitlement_count', 'manager_team_entitlement_count', 
        'user_total_entitlements', 'org_entitlement_count',
        'EndpointSystemId', 'NOrganisationId', 'NBusinessRoleId'
    ]
    
    # Add embedding features from candidate model
    numerical_features.extend([col for col in candidate_features if col.startswith(('user_emb_', 'ent_emb_'))])
    
    # Select features that exist
    existing_features = [col for col in numerical_features if col in features_df.columns]
    
    X = features_df[existing_features].fillna(0)
    y = features_df['HasAccess'] if 'HasAccess' in features_df.columns else None
    
    logger.info("Reranker features shape: %s", X.shape)
    return X, y, existing_features

def calculate_adoption_rate(peer_group_df, entitlement_id, entrecon_df):
    """Helper function to calculate adoption rate for a peer group - TYPE SAFE VERSION"""
    
    peer_count = len(peer_group_df)
    
    if peer_count == 0:
        return {
            'adoption_rate': 0,
            'peer_count': 0,
            'peers_with_access': 0
        }
    
    # TYPE SAFE: Now that ETL fixes data types, this should work properly
    peer_ids = peer_group_df['id'].tolist()
    
    # Ensure entitlement_id is string (matches entrecon data type from ETL)
    entitlement_id_str = str(entitlement_id)
    
    # Type-safe filtering - now works because ETL standardized types
    peer_access = entrecon_df[
        (entrecon_df['UserId'].isin(peer_ids)) &
        (entrecon_df['EntitlementId'] == entitlement_id_str)
    ]
    
    peers_with_access = len(peer_access)
    adoption_rate = peers_with_access / peer_count if peer_count > 0 else 0
    
    return {
        'adoption_rate': adoption_rate,
        'peer_count': peer_count,
        'peers_with_access': peers_with_access
    }

def build_peer_lookup_cache(graph_dfs):
    """
    Build immutable lookup maps for fast peer-adoption feature calculation.
    Intended to run once at artifact load and reused across prediction requests.
    """
    users_df = graph_dfs.get("users")
    entrecon_df = graph_dfs.get("entrecon")
    if users_df is None or entrecon_df is None or users_df.empty or entrecon_df.empty:
        return {
            "active_user_ids": set(),
            "user_to_role": {},
            "user_to_org": {},
            "user_to_manager": {},
            "role_to_users": {},
            "org_to_users": {},
            "manager_to_users": {},
            "entitlement_to_users": {},
        }

    users = users_df.copy()
    if "IsActive" in users.columns:
        active_mask = users["IsActive"].fillna(False).astype(bool)
        users = users[active_mask].copy()

    users["id"] = pd.to_numeric(users["id"], errors="coerce")
    users = users.dropna(subset=["id"]).copy()
    users["id"] = users["id"].astype("int64")

    active_user_ids = set(users["id"].tolist())

    user_to_role = {}
    user_to_org = {}
    user_to_manager = {}
    role_to_users = {}
    org_to_users = {}
    manager_to_users = {}

    for _, row in users.iterrows():
        uid = int(row["id"])
        role = row.get("NBusinessRoleId")
        org = row.get("NOrganisationId")
        manager = row.get("ManagerId")

        user_to_role[uid] = role
        user_to_org[uid] = org
        user_to_manager[uid] = manager

        if pd.notna(role):
            role_to_users.setdefault(role, set()).add(uid)
        if pd.notna(org):
            org_to_users.setdefault(org, set()).add(uid)
        if pd.notna(manager):
            manager_to_users.setdefault(manager, set()).add(uid)

    entitlement_to_users = {}
    entre = entrecon_df.copy()
    entre["UserId"] = pd.to_numeric(entre["UserId"], errors="coerce")
    entre = entre.dropna(subset=["UserId"]).copy()
    entre["UserId"] = entre["UserId"].astype("int64")
    entre = entre[entre["UserId"].isin(active_user_ids)].copy()
    entre["EntitlementId"] = entre["EntitlementId"].astype(str)

    for ent_id, grp in entre.groupby("EntitlementId"):
        entitlement_to_users[ent_id] = set(grp["UserId"].tolist())

    return {
        "active_user_ids": active_user_ids,
        "user_to_role": user_to_role,
        "user_to_org": user_to_org,
        "user_to_manager": user_to_manager,
        "role_to_users": role_to_users,
        "org_to_users": org_to_users,
        "manager_to_users": manager_to_users,
        "entitlement_to_users": entitlement_to_users,
    }


def _adoption(peer_set, entitled_users):
    peer_count = len(peer_set)
    if peer_count == 0:
        return 0.0, 0
    return len(peer_set & entitled_users) / peer_count, peer_count


def calculate_peer_adoption_features(labeled_df, graph_dfs, peer_lookup_cache=None):
    """Calculate 8 peer adoption features using cached lookup maps + set operations."""
    logger.info("Calculating peer adoption rates (8 features)...")

    cache = peer_lookup_cache or graph_dfs.get("_peer_lookup") or build_peer_lookup_cache(graph_dfs)
    empty = set()

    active_user_ids = cache["active_user_ids"]
    user_to_role = cache["user_to_role"]
    user_to_org = cache["user_to_org"]
    user_to_manager = cache["user_to_manager"]
    role_to_users = cache["role_to_users"]
    org_to_users = cache["org_to_users"]
    manager_to_users = cache["manager_to_users"]
    entitlement_to_users = cache["entitlement_to_users"]

    peer_features = []

    for _, row in labeled_df.iterrows():
        try:
            user_id = int(row["UserId"])
        except Exception:
            user_id = row["UserId"]
        entitlement_id = str(row["EntitlementId"])

        if user_id not in active_user_ids:
            peer_features.append({
                "UserId": user_id,
                "EntitlementId": entitlement_id,
                "close_peer_adoption_rate": 0.0,
                "direct_team_adoption_rate": 0.0,
                "role_peer_adoption_rate": 0.0,
                "dept_peer_adoption_rate": 0.0,
                "close_peer_count": 0,
                "direct_team_count": 0,
                "role_peer_count": 0,
                "dept_peer_count": 0,
            })
            continue

        role = user_to_role.get(user_id)
        org = user_to_org.get(user_id)
        manager = user_to_manager.get(user_id)
        entitled_users = entitlement_to_users.get(entitlement_id, empty)

        role_users = role_to_users.get(role, empty) if pd.notna(role) else empty
        org_users = org_to_users.get(org, empty) if pd.notna(org) else empty
        manager_users = manager_to_users.get(manager, empty) if pd.notna(manager) else empty

        close_peers = (role_users & org_users) - {user_id}
        direct_team = manager_users - {user_id}
        role_peers = role_users - {user_id}
        dept_peers = org_users - {user_id}

        close_rate, close_count = _adoption(close_peers, entitled_users)
        team_rate, team_count = _adoption(direct_team, entitled_users)
        role_rate, role_count = _adoption(role_peers, entitled_users)
        dept_rate, dept_count = _adoption(dept_peers, entitled_users)

        peer_features.append({
            "UserId": user_id,
            "EntitlementId": entitlement_id,
            "close_peer_adoption_rate": close_rate,
            "direct_team_adoption_rate": team_rate,
            "role_peer_adoption_rate": role_rate,
            "dept_peer_adoption_rate": dept_rate,
            "close_peer_count": close_count,
            "direct_team_count": team_count,
            "role_peer_count": role_count,
            "dept_peer_count": dept_count,
        })

    return pd.DataFrame(peer_features)

# MINIMAL CHANGE to your existing create_enhanced_reranker_features function
# In your feature_engineering.py, just change this one part:

# def create_enhanced_reranker_features(labeled_df, embeddings_df, graph_dfs):
    # """Enhanced version of reranker features that includes peer adoption rates - TYPE SAFE VERSION"""
    # print("🚀 Creating enhanced reranker features with peer adoption rates...")
    
    # # Start with existing reranker features
    # X_original, y_original, original_features = create_reranker_model_features(labeled_df, embeddings_df, graph_dfs)
    
    # # Add peer adoption rate features
    # peer_features_df = calculate_peer_adoption_features(labeled_df, graph_dfs)
    
    # # ✅ ONLY CHANGE: Include ALL 8 peer features instead of 6
    # # OLD CODE was:
    # # peer_features_df[['close_peer_adoption_rate','direct_team_adoption_rate',
    # #                  'role_peer_adoption_rate', 'dept_peer_adoption_rate',
    # #                  'close_peer_count', 'direct_team_count',
    # #                  'role_peer_count', 'dept_peer_count']]
    
    # # But then the enhanced_features list was missing direct_team features
    # # NEW CODE:
    # all_peer_feature_columns = [
        # 'close_peer_adoption_rate',
        # 'direct_team_adoption_rate',     # ✅ Make sure this is included
        # 'role_peer_adoption_rate', 
        # 'dept_peer_adoption_rate',
        # 'close_peer_count',
        # 'direct_team_count',             # ✅ Make sure this is included
        # 'role_peer_count', 
        # 'dept_peer_count'
    # ]
    
    # # Merge original features with peer features
    # features_df = pd.concat([
        # labeled_df.reset_index(drop=True),
        # X_original.reset_index(drop=True),
        # peer_features_df[all_peer_feature_columns].reset_index(drop=True)
    # ], axis=1)
    
    # # ✅ FIXED: Enhanced feature list with ALL peer features
    # enhanced_features = original_features + all_peer_feature_columns
    
    # # Select features that exist
    # existing_enhanced_features = [col for col in enhanced_features if col in features_df.columns]
    
    # X_enhanced = features_df[existing_enhanced_features].fillna(0)
    # y_enhanced = features_df['HasAccess'] if 'HasAccess' in features_df.columns else y_original
    
    # print(f"✅ Enhanced reranker features shape: {X_enhanced.shape}")
    # print(f"🆕 ALL peer features added: {all_peer_feature_columns}")
    
    # return X_enhanced, y_enhanced, existing_enhanced_features
    
    
def create_enhanced_reranker_features(labeled_df, embeddings_df, graph_dfs, peer_lookup_cache=None):
    """Enhanced reranker features with ALL 8 peer features, schema-safe & type-safe."""
    total_start = time.perf_counter()
    logger.info("Creating enhanced reranker features with peer adoption rates...")

    # 1) Base reranker features
    t0 = time.perf_counter()
    X_original, y_original, original_features = create_reranker_model_features(
        labeled_df, embeddings_df, graph_dfs
    )
    base_reranker_ms = (time.perf_counter() - t0) * 1000

    # 2) Peer features (must return ALL 8)
    t0 = time.perf_counter()
    peer_features_df = calculate_peer_adoption_features(
        labeled_df,
        graph_dfs,
        peer_lookup_cache=peer_lookup_cache,
    )
    peer_features_ms = (time.perf_counter() - t0) * 1000

    EXPECTED_PEER_COLS = [
        "close_peer_adoption_rate",
        "direct_team_adoption_rate",
        "role_peer_adoption_rate",
        "dept_peer_adoption_rate",
        "close_peer_count",
        "direct_team_count",
        "role_peer_count",
        "dept_peer_count",
    ]
    # Ensure all 8 columns exist and are numeric (rates=float, counts=int)
    t0 = time.perf_counter()
    for col in EXPECTED_PEER_COLS:
        if col not in peer_features_df.columns:
            peer_features_df[col] = 0
    rate_cols = [
        "close_peer_adoption_rate",
        "direct_team_adoption_rate",
        "role_peer_adoption_rate",
        "dept_peer_adoption_rate",
    ]
    count_cols = [
        "close_peer_count",
        "direct_team_count",
        "role_peer_count",
        "dept_peer_count",
    ]
    peer_features_df[rate_cols] = (
        peer_features_df[rate_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )
    peer_features_df[count_cols] = (
        peer_features_df[count_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("int64")
    )
    peer_coerce_ms = (time.perf_counter() - t0) * 1000

    # 3) Merge original + peer features alongside labels/ids for safe alignment
    t0 = time.perf_counter()
    features_df = pd.concat(
        [
            labeled_df.reset_index(drop=True),
            X_original.reset_index(drop=True),
            peer_features_df[EXPECTED_PEER_COLS].reset_index(drop=True),
        ],
        axis=1,
    )
    merge_ms = (time.perf_counter() - t0) * 1000

    # 4) Build final ordered feature list and enforce presence
    t0 = time.perf_counter()
    enhanced_features = list(original_features) + EXPECTED_PEER_COLS
    missing_final = [c for c in enhanced_features if c not in features_df.columns]
    if missing_final:
        raise RuntimeError(f"Missing expected enhanced features: {missing_final}")

    # 5) Select, coerce numerics, fill NaNs, and sanity-check names
    X_enhanced = features_df[enhanced_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_enhanced = features_df["HasAccess"] if "HasAccess" in features_df.columns else y_original

    bad_names = [c for c in X_enhanced.columns if c is None or str(c).strip() == ""]
    if bad_names:
        raise RuntimeError(f"Blank/None feature names in enhanced matrix: {bad_names}")
    dups = X_enhanced.columns[X_enhanced.columns.duplicated()].tolist()
    if dups:
        raise RuntimeError(f"Duplicate feature names in enhanced matrix: {dups}")
    finalize_ms = (time.perf_counter() - t0) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000

    logger.info("Enhanced reranker features shape: %s", X_enhanced.shape)
    logger.info(
        "Enhanced reranker timing | rows=%d total_ms=%.2f base_reranker_ms=%.2f "
        "peer_features_ms=%.2f peer_coerce_ms=%.2f merge_ms=%.2f finalize_ms=%.2f",
        len(labeled_df),
        total_ms,
        base_reranker_ms,
        peer_features_ms,
        peer_coerce_ms,
        merge_ms,
        finalize_ms,
    )
    logger.debug("Peer features added: %s", EXPECTED_PEER_COLS)
    return X_enhanced, y_enhanced, enhanced_features
    
    
def calculate_peer_adoption_features_optimized(labeled_df, graph_dfs):
    """Calculate the 4 peer adoption rate features - OPTIMIZED VERSION (should take <1 minute)"""
    logger.info("Calculating peer adoption rates (optimized)...")
    
    start_time = time.time()
    
    # Pre-process data for faster lookups
    users_df = graph_dfs['users'].set_index('id')
    entrecon_df = graph_dfs['entrecon']
    
    # Create lookup dictionaries for O(1) access
    user_to_role = users_df['NBusinessRoleId'].to_dict()
    user_to_org = users_df['NOrganisationId'].to_dict()
    user_to_manager = users_df['ManagerId'].to_dict()
    
    logger.debug("Pre-processing completed in %.2fs", time.time() - start_time)

    # Pre-calculate peer groups (vectorized operations)
    peer_start = time.time()
    
    # Group users by role, org, and manager for faster lookups
    role_groups = users_df.groupby('NBusinessRoleId')['id'].apply(list).to_dict()
    org_groups = users_df.groupby('NOrganisationId')['id'].apply(list).to_dict()
    manager_groups = users_df.groupby('ManagerId')['id'].apply(list).to_dict()
    
    # Pre-calculate entitlement adoption by groups
    entrecon_counts = {}
    for ent_id in labeled_df['EntitlementId'].unique():
        users_with_ent = set(entrecon_df[entrecon_df['EntitlementId'] == ent_id]['UserId'])
        entrecon_counts[ent_id] = users_with_ent
    
    logger.debug("Peer groups calculated in %.2fs", time.time() - peer_start)

    # Vectorized calculation for all labeled pairs
    calc_start = time.time()
    
    peer_features = []
    
    # Process in batches for better performance
    batch_size = 1000
    total_rows = len(labeled_df)
    
    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch_df = labeled_df.iloc[batch_start:batch_end]
        
        batch_features = []
        
        for _, row in batch_df.iterrows():
            user_id = row['UserId']
            entitlement_id = row['EntitlementId']
            
            # Get user context (O(1) lookups)
            user_role = user_to_role.get(user_id)
            user_org = user_to_org.get(user_id)
            user_manager = user_to_manager.get(user_id)
            
            # Get users who have this entitlement (pre-calculated)
            users_with_entitlement = entrecon_counts.get(entitlement_id, set())
            
            # Calculate peer groups and adoption rates efficiently
            
            # 1. Close peers (same role + same org)
            close_peers = set()
            if pd.notna(user_role) and pd.notna(user_org):
                role_users = set(role_groups.get(user_role, []))
                org_users = set(org_groups.get(user_org, []))
                close_peers = (role_users & org_users) - {user_id}
            
            close_with_access = close_peers & users_with_entitlement
            close_adoption_rate = len(close_with_access) / len(close_peers) if close_peers else 0
            
            # 2. Direct team (same manager)
            direct_team = set()
            if pd.notna(user_manager):
                direct_team = set(manager_groups.get(user_manager, [])) - {user_id}
            
            team_with_access = direct_team & users_with_entitlement
            team_adoption_rate = len(team_with_access) / len(direct_team) if direct_team else 0
            
            # 3. Role peers (same role)
            role_peers = set()
            if pd.notna(user_role):
                role_peers = set(role_groups.get(user_role, [])) - {user_id}
            
            role_with_access = role_peers & users_with_entitlement
            role_adoption_rate = len(role_with_access) / len(role_peers) if role_peers else 0
            
            # 4. Department peers (same org)
            dept_peers = set()
            if pd.notna(user_org):
                dept_peers = set(org_groups.get(user_org, [])) - {user_id}
            
            dept_with_access = dept_peers & users_with_entitlement
            dept_adoption_rate = len(dept_with_access) / len(dept_peers) if dept_peers else 0
            
            batch_features.append({
                'UserId': user_id,
                'EntitlementId': entitlement_id,
                'close_peer_adoption_rate': close_adoption_rate,
                'direct_team_adoption_rate': team_adoption_rate,
                'role_peer_adoption_rate': role_adoption_rate,
                'dept_peer_adoption_rate': dept_adoption_rate,
                'close_peer_count': len(close_peers),
                'direct_team_count': len(direct_team),
                'role_peer_count': len(role_peers),
                'dept_peer_count': len(dept_peers)
            })
        
        peer_features.extend(batch_features)
        
        # Progress update
        if batch_start % (batch_size * 10) == 0:
            progress = (batch_end / total_rows) * 100
            elapsed = time.time() - calc_start
            logger.debug("Progress: %.1f%% (%d/%d) - %.1fs elapsed", progress, batch_end, total_rows, elapsed)

    logger.debug("Adoption rates calculated in %.2fs", time.time() - calc_start)

    total_time = time.time() - start_time
    logger.info("Peer adoption features completed in %.2fs", total_time)
    
    return pd.DataFrame(peer_features)
