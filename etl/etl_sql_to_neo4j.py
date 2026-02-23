# etl/etl_sql_to_neo4j.py (UPDATED WITH TYPE FIXES AT EXACT LOCATIONS)

import pandas as pd
from sqlalchemy import create_engine
from neo4j import GraphDatabase
from urllib.parse import quote_plus
import time

# --- 1. CONFIGURATION ---
SQL_DB_HOST, SQL_DB_USER, SQL_DB_PASS, SQL_DB_NAME = 'localhost', 'sa', '@bcd1234', 'F_IACM_Demo'
NEO4J_URI, NEO4J_USER, NEO4J_PASS, NEO4J_DATABASE = "bolt://localhost:7687", "neo4j", "@bcd1234", "neo4j"

# --- 2. CONNECTIONS & HELPERS ---
def get_sql_engine():
    """Connects to the SQL Server database."""
    print("Connecting to SQL Server...")
    conn_str = f"mssql+pyodbc://{SQL_DB_USER}:{quote_plus(SQL_DB_PASS)}@{SQL_DB_HOST}/{SQL_DB_NAME}?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
    return create_engine(conn_str)

def get_neo4j_driver():
    """Connects to the Neo4j database."""
    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    driver.verify_connectivity()
    return driver

# ADD THIS NEW FUNCTION RIGHT HERE (after get_neo4j_driver)
def fix_data_types_after_load(all_users, orgs, designations, unified_entitlements, endpoints, accounts, entrecon):
    """Fix data types immediately after SQL load - ROOT CAUSE FIX"""
    print("🔧 Applying comprehensive type fixes at ETL stage...")
    
    # Users DataFrame - Fix the problematic float64 columns
    print("   📊 Fixing Users DataFrame types...")
    print(f"   🔍 Users columns: {list(all_users.columns)}")
    
    # Handle both 'id' and 'Id' columns (your data likely has 'Id')
    user_id_col = 'Id' if 'Id' in all_users.columns else 'id'
    if user_id_col in all_users.columns:
        all_users['id'] = all_users[user_id_col].astype('int64')  # Standardize to lowercase 'id'
        if user_id_col == 'Id':  # Remove the original 'Id' column
            all_users = all_users.drop('Id', axis=1)
        print(f"      ✅ {user_id_col} → id: int64")
    
    # Convert problematic float64 columns to nullable Int64
    float_to_int_cols = ['ManagerId', 'NOrganisationId', 'NBusinessRoleId', 
                        'EndpointSystemId', 'EmployeeTypeId', 'HRMSUserId', 
                        'InvalidLoginAttempt', 'teamSize']
    
    for col in float_to_int_cols:
        if col in all_users.columns:
            all_users[col] = all_users[col].astype('Int64')  # Nullable integer (capital I)
            print(f"      ✅ {col}: float64 → Int64")
    
    # String columns for better performance
    string_cols = ['UserName', 'DisplayName', 'EmailId', 'FirstName', 'LastName', 'KnownAs']
    for col in string_cols:
        if col in all_users.columns:
            all_users[col] = all_users[col].astype('string')
    
    # Organizations DataFrame
    print("   📊 Fixing Organizations DataFrame types...")
    org_id_col = 'Id' if 'Id' in orgs.columns else 'id'
    if org_id_col in orgs.columns:
        orgs['id'] = orgs[org_id_col].astype('int64')
        if org_id_col == 'Id':
            orgs = orgs.drop('Id', axis=1)
    
    if 'ParentOrgId' in orgs.columns:
        orgs['ParentOrgId'] = orgs['ParentOrgId'].astype('Int64')
    if 'ScopeId' in orgs.columns:
        orgs['ScopeId'] = orgs['ScopeId'].astype('Int64')
    
    # Designations DataFrame
    print("   📊 Fixing Designations DataFrame types...")
    desig_id_col = 'Id' if 'Id' in designations.columns else 'id'
    if desig_id_col in designations.columns:
        designations['id'] = designations[desig_id_col].astype('int64')
        if desig_id_col == 'Id':
            designations = designations.drop('Id', axis=1)
    
    # Entitlements DataFrame - CRITICAL: Force string IDs
    print("   📊 Fixing Entitlements DataFrame types...")
    if 'id' in unified_entitlements.columns:
        unified_entitlements['id'] = unified_entitlements['id'].astype('string')
    if 'composite_id' in unified_entitlements.columns:
        unified_entitlements['composite_id'] = unified_entitlements['composite_id'].astype('string')
    if 'EndpointSystemId' in unified_entitlements.columns:
        unified_entitlements['EndpointSystemId'] = unified_entitlements['EndpointSystemId'].astype('int64')
    
    # Endpoints DataFrame
    print("   📊 Fixing Endpoints DataFrame types...")
    endpoint_id_col = 'Id' if 'Id' in endpoints.columns else 'id'
    if endpoint_id_col in endpoints.columns:
        endpoints['id'] = endpoints[endpoint_id_col].astype('int64')
        if endpoint_id_col == 'Id':
            endpoints = endpoints.drop('Id', axis=1)
    
    if 'OwnerUserId' in endpoints.columns:
        endpoints['OwnerUserId'] = endpoints['OwnerUserId'].astype('Int64')
    if 'ServiceAccountId' in endpoints.columns:
        endpoints['ServiceAccountId'] = endpoints['ServiceAccountId'].astype('Int64')
    
    # Accounts DataFrame
    print("   📊 Fixing Accounts DataFrame types...")
    account_id_col = 'Id' if 'Id' in accounts.columns else 'id'
    if account_id_col in accounts.columns:
        accounts['id'] = accounts[account_id_col].astype('int64')
        if account_id_col == 'Id':
            accounts = accounts.drop('Id', axis=1)
    
    if 'UserId' in accounts.columns:
        accounts['UserId'] = accounts['UserId'].astype('Int64')
    if 'EndpointSystemId' in accounts.columns:
        accounts['EndpointSystemId'] = accounts['EndpointSystemId'].astype('Int64')
    
    # Entrecon DataFrame - MOST CRITICAL for ML
    print("   📊 Fixing Entrecon DataFrame types...")
    if 'UserId' in entrecon.columns:
        entrecon['UserId'] = entrecon['UserId'].astype('int64')
    if 'EntitlementId' in entrecon.columns:
        entrecon['EntitlementId'] = entrecon['EntitlementId'].astype('string')
    
    print("✅ All data types fixed at ETL stage!")
    return all_users, orgs, designations, unified_entitlements, endpoints, accounts, entrecon
    
    
def clean_dataframe_for_neo4j(df):
    """Cleans DataFrame for Neo4j import by handling dates and NaNs."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime('%Y-%m-%dT%H:%M:%S').replace({pd.NaT: None})
    return df.where(pd.notnull(df), None)

def load_nodes(driver, df, label, id_column):
    """Dynamically loads all columns of a DataFrame as node properties."""
    if df.empty or id_column not in df.columns or df[id_column].isnull().all():
        print(f"⚠️ No valid nodes to load for label :{label}.")
        return
    print(f"Loading {len(df)} nodes with label :{label}...")
    df = clean_dataframe_for_neo4j(df.copy())
    query = f"UNWIND $rows AS row MERGE (n:{label} {{id: row.{id_column}}}) SET n += row"
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(query, rows=df.to_dict('records'))
    print(f"✅ Finished loading :{label} nodes.")

def load_relationships(driver, df, from_label, from_id, to_label, to_id, rel_type, rel_properties=[], force_direction=None):
    """Dynamically loads relationships with explicit direction control."""
    if df.empty:
        print(f"⚠️ No relationships to load for type [:{rel_type}].")
        return
    print(f"Loading {len(df)} potential relationships of type [:{rel_type}]...")
    df = clean_dataframe_for_neo4j(df.copy())
    
    # Handle NULL values for User relationships
    if from_label == 'User' and df[from_id].isnull().any():
        null_from_rows = df[df[from_id].isnull()].copy()
        if len(null_from_rows) > 0:
            print(f"  🔧 Handling {len(null_from_rows)} relationships with NULL {from_label} IDs...")
            df[from_id] = df[from_id].astype('object')
            for idx, row in null_from_rows.iterrows():
                df.loc[idx, from_id] = 'ORPHAN_USER'
    
    if to_label == 'User' and df[to_id].isnull().any():
        null_to_rows = df[df[to_id].isnull()].copy()
        if len(null_to_rows) > 0:
            print(f"  🔧 Handling {len(null_to_rows)} relationships with NULL {to_label} IDs...")
            df[to_id] = df[to_id].astype('object')
            for idx, row in null_to_rows.iterrows():
                df.loc[idx, to_id] = 'ORPHAN_USER'
    
    clean_df_dict = df.dropna(subset=[from_id, to_id]).to_dict('records')
    if not clean_df_dict:
        print(f"⚠️ No valid relationships to create for [:{rel_type}] after processing NULLs.")
        return
    
    set_clause = f"SET r += apoc.map.submap(row, $props_to_set)" if rel_properties else ""
    
    # FIXED: Explicit direction control
    if force_direction == 'reverse':
        relationship_pattern = f"(from)<-[r:{rel_type}]-(to)"
    elif force_direction == 'forward':
        relationship_pattern = f"(from)-[r:{rel_type}]->(to)"
    else:
        # Default logic (but improved)
        if rel_type in ['IS_GRANTED_TO', 'REPORTS_TO']:
            relationship_pattern = f"(from)<-[r:{rel_type}]-(to)"
        else:
            relationship_pattern = f"(from)-[r:{rel_type}]->(to)"
    
    query = f"""
    UNWIND $rows AS row
    MATCH (from:{from_label} {{id: row.{from_id}}})
    MATCH (to:{to_label} {{id: row.{to_id}}})
    MERGE {relationship_pattern}
    {set_clause}
    """
    
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(query, rows=clean_df_dict, props_to_set=rel_properties)
    print(f"✅ Finished loading [:{rel_type}] relationships.")

def create_constraints(driver):
    """Creates uniqueness constraints for all primary node labels."""
    print("Creating database constraints...")
    with driver.session(database=NEO4J_DATABASE) as session:
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:User) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entitlement) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Organization) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Designation) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:EndpointSystem) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Account) REQUIRE n.id IS UNIQUE"
        ]
        for constraint in constraints: session.run(constraint)
    print("✅ Constraints created.")

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    start_time = time.time()
    sql_engine = get_sql_engine()
    neo4j_driver = get_neo4j_driver()
    
    print("\n--- Phase 1: Reading ALL Data (Complete Replica) ---")
    create_constraints(neo4j_driver)
    
    # LOAD ALL DATA - NO FILTERING AT ETL STAGE
    all_users = pd.read_sql("SELECT * FROM IACM_User", sql_engine)
    orgs = pd.read_sql("SELECT * FROM IACM_NOrganisation", sql_engine)
    designations = pd.read_sql("SELECT * FROM IACM_NDesignation", sql_engine)
    generic_entitlements = pd.read_sql("SELECT * FROM IACM_Entitlement", sql_engine)
    ad_groups = pd.read_sql("SELECT * FROM ILM_ADS_GroupInfo", sql_engine)
    endpoints = pd.read_sql("SELECT * FROM IACM_EndpointSystem", sql_engine)
    accounts = pd.read_sql("SELECT * FROM IACM_AccountReconcilation", sql_engine)
    entrecon = pd.read_sql("SELECT * FROM IACM_EntitlementReconcilation", sql_engine)
    
    # Use ALL users (active + inactive) for complete replica
    users = all_users.copy()
    print(f"📊 Loading ALL users: {len(users)} total users ({len(all_users[all_users['IsActive'] == 1])} active, {len(all_users[all_users['IsActive'] == 0])} inactive)")
    
    print("\n--- Phase 2: Unifying Entitlements + Creating Missing Ones ---")
    
    generic_entitlements['composite_id'] = generic_entitlements['EndpointSystemId'].astype('Int64').astype(str) + "_" + generic_entitlements['EndpointEntitlementId'].astype('Int64').astype(str)
    ad_groups['composite_id'] = ad_groups['EndPointSystemId'].astype('Int64').astype(str) + "_" + ad_groups['Id'].astype('Int64').astype(str)
    
    gen_ent_subset = generic_entitlements[['composite_id', 'Name', 'Description', 'EndpointSystemId']].copy()
    ad_group_subset = ad_groups[['composite_id', 'name', 'description', 'EndPointSystemId']].rename(columns={'name': 'Name', 'description': 'Description', 'EndPointSystemId': 'EndpointSystemId'}).copy()
    unified_entitlements = pd.concat([gen_ent_subset, ad_group_subset], ignore_index=True).drop_duplicates(subset=['composite_id'])
    
    entrecon['composite_id'] = entrecon['EndpointSystemId'].astype('Int64').astype(str) + "_" + entrecon['EntitlementId'].astype('Int64').astype(str)
    
    # Create phantom entitlements for missing ones
    print("🔍 Identifying missing entitlements referenced in reconciliation...")
    existing_entitlement_ids = set(unified_entitlements['composite_id'].dropna())
    referenced_entitlement_ids = set(entrecon['composite_id'].dropna())
    missing_entitlement_ids = referenced_entitlement_ids - existing_entitlement_ids
    
    print(f"📊 Entitlement analysis:")
    print(f"   Existing entitlements: {len(existing_entitlement_ids)}")
    print(f"   Referenced in reconciliation: {len(referenced_entitlement_ids)}")
    print(f"   Missing entitlements: {len(missing_entitlement_ids)}")
    
    if missing_entitlement_ids:
        print(f"🔧 Creating phantom entitlement nodes for {len(missing_entitlement_ids)} missing entitlements...")
        phantom_entitlements = []
        for missing_id in missing_entitlement_ids:
            endpoint_id, ent_id = missing_id.split('_')
            phantom_entitlements.append({
                'composite_id': missing_id,
                'Name': f'Phantom_Entitlement_{ent_id}',
                'Description': f'Auto-created for missing entitlement {ent_id} on system {endpoint_id}',
                'EndpointSystemId': int(endpoint_id)
            })
        
        phantom_df = pd.DataFrame(phantom_entitlements)
        unified_entitlements = pd.concat([unified_entitlements, phantom_df], ignore_index=True)
        print(f"✅ Total entitlements after adding phantoms: {len(unified_entitlements)}")
    
    # HANDLE NULL USER IDs
    null_user_records = entrecon[entrecon['UserId'].isnull()]
    if len(null_user_records) > 0:
        print(f"🔧 Found {len(null_user_records)} records with NULL user IDs - creating phantom users...")
        
        phantom_users = []
        for idx, record in null_user_records.iterrows():
            phantom_user_id = -1000 - idx
            phantom_users.append({
                'Id': phantom_user_id,
                'UserName': f'PHANTOM_USER_{phantom_user_id}',
                'DisplayName': f'Auto-created for NULL user ID (Record {idx})',
                'IsActive': False,
                'NOrganisationId': None,
                'NBusinessRoleId': None,
                'ManagerId': None
            })
        
        phantom_users_df = pd.DataFrame(phantom_users)
        users = pd.concat([users, phantom_users_df], ignore_index=True)
        
        # Update NULL user IDs to phantom users
        for idx, (orig_idx, record) in enumerate(null_user_records.iterrows()):
            entrecon.loc[orig_idx, 'UserId'] = -1000 - idx
        
        print(f"✅ Created {len(phantom_users)} phantom users for NULL user IDs")
    
    # ADD ID COLUMN FOR UNIFIED ENTITLEMENTS
    unified_entitlements['id'] = unified_entitlements['composite_id']
    
    # WITH THIS CORRECTED VERSION:
    entrecon_rels = pd.DataFrame({
        'UserId': entrecon['UserId'].copy(),
        'EntitlementId': entrecon['composite_id'].copy()
    })
    
    # 🚀 CRITICAL: APPLY TYPE FIXES RIGHT HERE (after data processing, before filtering)
    users, orgs, designations, unified_entitlements, endpoints, accounts, entrecon_rels = fix_data_types_after_load(
        users, orgs, designations, unified_entitlements, endpoints, accounts, entrecon_rels
    )
    
    
    # SAFETY: Reset all indices to prevent duplicate index issues
    users = users.reset_index(drop=True)
    orgs = orgs.reset_index(drop=True)
    designations = designations.reset_index(drop=True)
    unified_entitlements = unified_entitlements.reset_index(drop=True)
    endpoints = endpoints.reset_index(drop=True)
    accounts = accounts.reset_index(drop=True)
    entrecon_rels = entrecon_rels.reset_index(drop=True)


    
    print("\n🚀 Pre-filtering relationship data ONLY for referential integrity...")
    
    # IMPORTANT: Now use the properly typed data for filtering
    valid_user_ids = set(users['id'].dropna())
    valid_entitlement_ids = set(unified_entitlements['id'].dropna())
    valid_account_ids = set(accounts['id'].dropna())
    
    # entrecon_rels = entrecon_rels[
        # (entrecon_rels['UserId'].isin(valid_user_ids) | entrecon_rels['UserId'].isna()) &
        # entrecon_rels['EntitlementId'].isin(valid_entitlement_ids)
    # ].copy()
    
    # More explicit filtering that handles nullable Int64 properly
    # user_filter = entrecon_rels['UserId'].isin(valid_user_ids) | entrecon_rels['UserId'].isna()
    # entitlement_filter = entrecon_rels['EntitlementId'].isin(valid_entitlement_ids)

    # entrecon_rels = entrecon_rels[user_filter & entitlement_filter].copy()
    
    
    
    print("\n🔍 DEBUGGING entrecon_rels DataFrame...")

    # Check the DataFrame structure
    print(f"entrecon_rels shape: {entrecon_rels.shape}")
    print(f"entrecon_rels columns: {list(entrecon_rels.columns)}")
    print(f"entrecon_rels dtypes:\n{entrecon_rels.dtypes}")

    # Check for duplicate indices
    duplicate_indices = entrecon_rels.index.duplicated().sum()
    print(f"Duplicate indices: {duplicate_indices}")

    # Check for duplicate rows
    duplicate_rows = entrecon_rels.duplicated().sum()
    print(f"Duplicate rows: {duplicate_rows}")

    # COMPLETE FIX: Clean the DataFrame thoroughly
    print("🔧 Applying comprehensive DataFrame cleanup...")

    # 1. Reset index completely
    entrecon_rels = entrecon_rels.reset_index(drop=True)

    # 2. Remove any duplicate rows
    entrecon_rels = entrecon_rels.drop_duplicates().reset_index(drop=True)

    # 3. Ensure proper data types (force conversion)
    print("🔧 Re-applying data types to entrecon_rels...")
    entrecon_rels['UserId'] = pd.to_numeric(entrecon_rels['UserId'], errors='coerce').astype('int64')
    entrecon_rels['EntitlementId'] = entrecon_rels['EntitlementId'].astype('string')

    # 4. Remove rows with invalid data
    print("🔧 Removing rows with invalid data...")
    initial_count = len(entrecon_rels)
    entrecon_rels = entrecon_rels.dropna(subset=['UserId', 'EntitlementId']).reset_index(drop=True)
    final_count = len(entrecon_rels)
    print(f"Removed {initial_count - final_count} rows with null values")

    print("\n🚀 SIMPLIFIED: Using all clean relationship data (no complex filtering)...")
    print(f"📊 Final entrecon_rels: {len(entrecon_rels)} records")
    
    print(f"📊 Filtered `entrecon` from {len(entrecon)} down to {len(entrecon_rels)} records.")
    
    

    print("\n--- Phase 3: Loading ALL Graph Nodes (Complete Replica + Phantoms) ---")
    
    # Create ORPHAN_USER node
    orphan_user = pd.DataFrame([{
        'id': 'ORPHAN_USER',
        'UserName': 'ORPHAN_USER', 
        'DisplayName': 'Placeholder for NULL user references',
        'IsActive': False
    }])
    users = pd.concat([users, orphan_user], ignore_index=True)
    
    load_nodes(neo4j_driver, users, 'User', 'id')
    load_nodes(neo4j_driver, orgs, 'Organization', 'id')
    load_nodes(neo4j_driver, designations, 'Designation', 'id')
    load_nodes(neo4j_driver, endpoints, 'EndpointSystem', 'id')
    load_nodes(neo4j_driver, accounts, 'Account', 'id')
    load_nodes(neo4j_driver, unified_entitlements, 'Entitlement', 'id')

    print("\n--- Phase 4: Loading ALL Graph Relationships (CORRECT DIRECTIONS) ---")
    load_relationships(neo4j_driver, users, 'User', 'id', 'Organization', 'NOrganisationId', 'BELONGS_TO', force_direction='forward')
    load_relationships(neo4j_driver, users, 'User', 'id', 'Designation', 'NBusinessRoleId', 'HAS_DESIGNATION', force_direction='forward')
    load_relationships(neo4j_driver, users, 'User', 'id', 'User', 'ManagerId', 'REPORTS_TO', force_direction='forward')
    load_relationships(neo4j_driver, accounts, 'User', 'UserId', 'Account', 'id', 'OWNS_ACCOUNT', force_direction='forward')
    load_relationships(neo4j_driver, endpoints, 'User', 'OwnerUserId', 'EndpointSystem', 'id', 'OWNS_SYSTEM', force_direction='forward')
    load_relationships(neo4j_driver, accounts, 'Account', 'id', 'EndpointSystem', 'EndpointSystemId', 'ON_SYSTEM', force_direction='forward')
    load_relationships(neo4j_driver, unified_entitlements, 'Entitlement', 'id', 'EndpointSystem', 'EndpointSystemId', 'PART_OF_SYSTEM', force_direction='forward')
    
    # FIXED: Correct direction for access relationships
    has_access_props = [c for c in entrecon_rels.columns if c not in ['UserId', 'EntitlementId']]
    load_relationships(neo4j_driver, entrecon_rels, 'User', 'UserId', 'Entitlement', 'EntitlementId', 'HAS_ACCESS_TO', has_access_props, force_direction='forward')

    print("\n--- Phase 5: Pre-calculating and storing aggregate statistics ---")
    with neo4j_driver.session(database=NEO4J_DATABASE) as session:
        print("🚀 Calculating Manager Team Sizes...")
        session.run("MATCH (mgr:User)<-[:REPORTS_TO]-(emp:User) WITH mgr, count(emp) AS teamSize SET mgr.teamSize = teamSize")
        print("🚀 Calculating Role Group Sizes...")
        session.run("MATCH (u:User)-[:BELONGS_TO]->(o:Organization) MATCH (u)-[:HAS_DESIGNATION]->(d:Designation) WITH o, d, count(u) as roleGroupSize MERGE (o)-[r:HAS_ROLE_GROUP_SIZE {designationId: d.id}]->(d) SET r.count = roleGroupSize")
        print("🚀 Calculating Manager-Entitlement Counts...")
        session.run("MATCH (u:User)-[:REPORTS_TO]->(mgr:User) MATCH (u)-[:HAS_ACCESS_TO]->(e:Entitlement) WITH mgr, e, count(u) AS entCount MERGE (mgr)-[r:TEAM_HAS_ENTITLEMENT]->(e) SET r.count = entCount")
        print("🚀 Calculating Role-Entitlement Counts...")
        session.run("MATCH (u:User)-[:BELONGS_TO]->(o:Organization) MATCH (u)-[:HAS_DESIGNATION]->(d:Designation) MATCH (u)-[:HAS_ACCESS_TO]->(e:Entitlement) WITH o, d, e, count(u) as entCount MERGE (o)-[r:ROLE_GROUP_HAS_ENTITLEMENT {designationId: d.id}]->(e) SET r.count = entCount")

    # Final statistics
    with neo4j_driver.session(database=NEO4J_DATABASE) as session:
        print("\n📊 Final Database Statistics:")
        result = session.run("MATCH (u:User) RETURN u.IsActive as status, count(u) as count ORDER BY count DESC")
        for record in result:
            status = "Active" if record['status'] else "Inactive"
            print(f"   {status} Users: {record['count']}")
        
        # FIXED: Check relationships in correct direction
        result = session.run("MATCH (u:User)-[:HAS_ACCESS_TO]->() RETURN u.IsActive as status, count(DISTINCT u) as usersWithAccess ORDER BY usersWithAccess DESC")
        for record in result:
            status = "Active" if record['status'] else "Inactive" 
            print(f"   {status} Users with Access: {record['usersWithAccess']}")
        
        result = session.run("MATCH (u:User)-[:HAS_ACCESS_TO]->() RETURN count(*) as totalAccess")
        total_access = result.single()['totalAccess']
        print(f"   Total HAS_ACCESS_TO relationships: {total_access}")
        
        # Show phantom statistics
        result = session.run("MATCH (u:User) WHERE u.UserName CONTAINS 'PHANTOM' RETURN count(u) as phantomUsers")
        phantom_users_count = result.single()['phantomUsers']
        print(f"   Phantom Users Created: {phantom_users_count}")
        
        result = session.run("MATCH (e:Entitlement) WHERE e.Name CONTAINS 'Phantom' RETURN count(e) as phantomEnts")
        phantom_ents_count = result.single()['phantomEnts']
        print(f"   Phantom Entitlements Created: {phantom_ents_count}")

    neo4j_driver.close()
    print(f"\n🎉 ETL with PROPER TYPE FIXES completed in {time.time() - start_time:.2f} seconds.")
    print("📋 All data now has correct types from the source!")