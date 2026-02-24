# etl_diagnostic.py - Diagnose why records are being filtered out

import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from ml_pipeline import config

def get_sql_engine():
    config.require_sql_config()
    conn_str = (
        f"mssql+pyodbc://{config.SQL_DB_USER}:{quote_plus(config.SQL_DB_PASS)}"
        f"@{config.SQL_DB_HOST}/{config.SQL_DB_NAME}"
        "?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
    )
    return create_engine(conn_str)

def diagnose_etl_filtering():
    print("=== ETL FILTERING DIAGNOSTIC ===\n")
    
    sql_engine = get_sql_engine()
    
    # Load the same data as ETL
    print("1. Loading data from SQL...")
    all_users = pd.read_sql("SELECT * FROM IACM_User", sql_engine)
    generic_entitlements = pd.read_sql("SELECT * FROM IACM_Entitlement", sql_engine)
    ad_groups = pd.read_sql("SELECT * FROM ILM_ADS_GroupInfo", sql_engine)
    accounts = pd.read_sql("SELECT * FROM IACM_AccountReconcilation", sql_engine)
    entrecon = pd.read_sql("SELECT * FROM IACM_EntitlementReconcilation", sql_engine)
    
    print(f"   Total users: {len(all_users)}")
    print(f"   Total entrecon records: {len(entrecon)}")
    print(f"   Total accounts: {len(accounts)}")
    print(f"   Generic entitlements: {len(generic_entitlements)}")
    print(f"   AD groups: {len(ad_groups)}")
    
    # Replicate the ETL logic
    print("\n2. Replicating ETL composite key creation...")
    generic_entitlements['composite_id'] = generic_entitlements['EndpointSystemId'].astype('Int64').astype(str) + "_" + generic_entitlements['EndpointEntitlementId'].astype('Int64').astype(str)
    ad_groups['composite_id'] = ad_groups['EndPointSystemId'].astype('Int64').astype(str) + "_" + ad_groups['Id'].astype('Int64').astype(str)
    
    gen_ent_subset = generic_entitlements[['composite_id', 'Name', 'Description', 'EndpointSystemId']].copy()
    ad_group_subset = ad_groups[['composite_id', 'name', 'description', 'EndPointSystemId']].rename(columns={'name': 'Name', 'description': 'Description', 'EndPointSystemId': 'EndpointSystemId'}).copy()
    unified_entitlements = pd.concat([gen_ent_subset, ad_group_subset], ignore_index=True).drop_duplicates(subset=['composite_id'])
    
    entrecon['composite_id'] = entrecon['EndpointSystemId'].astype('Int64').astype(str) + "_" + entrecon['EntitlementId'].astype('Int64').astype(str)
    
    print(f"   Unified entitlements created: {len(unified_entitlements)}")
    print(f"   Entrecon with composite keys: {len(entrecon)}")
    
    # Check what happens during filtering
    print("\n3. Analyzing referential integrity filtering...")
    
    valid_user_ids = set(all_users['Id'].dropna())
    valid_entitlement_ids = set(unified_entitlements['composite_id'].dropna()) 
    valid_account_ids = set(accounts['Id'].dropna())
    
    print(f"   Valid user IDs: {len(valid_user_ids)}")
    print(f"   Valid entitlement composite IDs: {len(valid_entitlement_ids)}")
    print(f"   Valid account IDs: {len(valid_account_ids)}")
    
    # Check each filter condition separately
    user_filter_pass = entrecon['UserId'].isin(valid_user_ids)
    ent_filter_pass = entrecon['composite_id'].isin(valid_entitlement_ids)
    account_filter_pass = entrecon['AccountReconcilationId'].isin(valid_account_ids)
    
    print(f"\n4. Filter Analysis:")
    print(f"   Records passing user filter: {user_filter_pass.sum()} / {len(entrecon)}")
    print(f"   Records passing entitlement filter: {ent_filter_pass.sum()} / {len(entrecon)}")
    print(f"   Records passing account filter: {account_filter_pass.sum()} / {len(entrecon)}")
    
    # Final filter
    all_filters_pass = user_filter_pass & ent_filter_pass & account_filter_pass
    print(f"   Records passing ALL filters: {all_filters_pass.sum()} / {len(entrecon)}")
    print(f"   Records filtered OUT: {len(entrecon) - all_filters_pass.sum()}")
    
    # Analyze what's being filtered out
    print(f"\n5. What's being filtered out:")
    
    # User ID issues
    user_issues = entrecon[~user_filter_pass]
    if len(user_issues) > 0:
        print(f"   User ID issues: {len(user_issues)} records")
        print(f"   Sample invalid user IDs: {user_issues['UserId'].unique()[:10]}")
        print(f"   Are these user IDs NULL? {user_issues['UserId'].isnull().sum()}")
    
    # Entitlement issues  
    ent_issues = entrecon[~ent_filter_pass]
    if len(ent_issues) > 0:
        print(f"   Entitlement ID issues: {len(ent_issues)} records")
        print(f"   Sample invalid composite IDs: {ent_issues['composite_id'].unique()[:10]}")
        print(f"   Are composite IDs NULL? {ent_issues['composite_id'].isnull().sum()}")
        
        # Check if the individual components exist
        print(f"   Sample EndpointSystemIds in failing records: {ent_issues['EndpointSystemId'].unique()[:10]}")
        print(f"   Sample EntitlementIds in failing records: {ent_issues['EntitlementId'].unique()[:10]}")
    
    # Account issues
    account_issues = entrecon[~account_filter_pass]
    if len(account_issues) > 0:
        print(f"   Account ID issues: {len(account_issues)} records")
        print(f"   Sample invalid account IDs: {account_issues['AccountReconcilationId'].unique()[:10]}")
        print(f"   Are account IDs NULL? {account_issues['AccountReconcilationId'].isnull().sum()}")
    
    # Check for potential data type issues
    print(f"\n6. Data Type Analysis:")
    print(f"   EntitlementId data types in entrecon: {entrecon['EntitlementId'].dtype}")
    print(f"   EndpointSystemId data types in entrecon: {entrecon['EndpointSystemId'].dtype}")
    print(f"   Generic entitlement EndpointEntitlementId types: {generic_entitlements['EndpointEntitlementId'].dtype}")
    print(f"   AD group Id types: {ad_groups['Id'].dtype}")
    
    # Active vs inactive user analysis
    print(f"\n7. Active vs Inactive User Analysis:")
    active_users = all_users[all_users['IsActive'] == 1]
    inactive_users = all_users[all_users['IsActive'] == 0]
    
    entrecon_filtered = entrecon[all_filters_pass]
    active_user_entrecon = entrecon_filtered[entrecon_filtered['UserId'].isin(active_users['Id'])]
    inactive_user_entrecon = entrecon_filtered[entrecon_filtered['UserId'].isin(inactive_users['Id'])]
    
    print(f"   Active users: {len(active_users)}")
    print(f"   Inactive users: {len(inactive_users)}")
    print(f"   Entrecon records for active users: {len(active_user_entrecon)}")
    print(f"   Entrecon records for inactive users: {len(inactive_user_entrecon)}")

if __name__ == "__main__":
    diagnose_etl_filtering()
