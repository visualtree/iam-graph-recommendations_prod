# ml_pipeline/data_loader.py (UPDATED FOR ETL TYPE FIXES)

import pandas as pd
from neo4j import GraphDatabase
from . import config

def get_neo4j_driver():
    """Creates and verifies a Neo4j database driver instance."""
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))
    driver.verify_connectivity()
    return driver

def get_nodes_as_df(driver, node_label, where_clause=""):
    """Fetches nodes with an optional WHERE clause for filtering."""
    print(f"Fetching :{node_label} nodes from Neo4j...")
    
    # Filter out only system users, but keep phantom entitlements since they're the only ones users have
    if node_label == 'User' and not where_clause:
        where_clause = "WHERE NOT u.UserName CONTAINS 'PHANTOM' AND NOT u.UserName = 'ORPHAN_USER'"
    
    query = f"MATCH (n:{node_label}) {where_clause} RETURN n"
    with driver.session(database=config.NEO4J_DATABASE) as session:
        results = session.run(query).data()
    if not results:
        print(f"⚠️ No nodes found for label ':{node_label}' with filter '{where_clause}'.")
        return pd.DataFrame({'id': []})
    
    df = pd.DataFrame([r['n'] for r in results])
    
    # IMPORTANT: Apply type standardization after loading from Neo4j
    df = standardize_loaded_dataframe_types(df, node_label)
    
    return df

def standardize_loaded_dataframe_types(df, node_label):
    """Standardize data types for DataFrames loaded from Neo4j to match ETL types"""
    
    if df.empty:
        return df
    
    # Users DataFrame
    if node_label == 'User':
        # Core ID - ensure int64
        if 'id' in df.columns:
            df['id'] = df['id'].astype('int64')
        
        # Foreign key columns - ensure nullable Int64 (matches ETL)
        nullable_int_cols = ['ManagerId', 'NOrganisationId', 'NBusinessRoleId', 
                            'EndpointSystemId', 'EmployeeTypeId', 'HRMSUserId']
        for col in nullable_int_cols:
            if col in df.columns:
                # Handle Neo4j nulls properly
                df[col] = df[col].where(pd.notna(df[col])).astype('Int64')
        
        # String columns
        string_cols = ['UserName', 'DisplayName', 'EmailId', 'FirstName', 'LastName']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype('string')
        
        # Boolean columns
        if 'IsActive' in df.columns:
            df['IsActive'] = df['IsActive'].astype('bool')
    
    # Entitlements DataFrame
    elif node_label == 'Entitlement':
        # CRITICAL: Entitlement IDs must be string (matches ETL)
        if 'id' in df.columns:
            df['id'] = df['id'].astype('string')
        
        # EndpointSystemId should be int64
        if 'EndpointSystemId' in df.columns:
            df['EndpointSystemId'] = df['EndpointSystemId'].astype('int64')
        
        # String columns
        string_cols = ['Name', 'Description']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype('string')
    
    # Organizations DataFrame
    elif node_label == 'Organization':
        if 'id' in df.columns:
            df['id'] = df['id'].astype('int64')
        
        nullable_int_cols = ['ParentOrgId', 'ScopeId']
        for col in nullable_int_cols:
            if col in df.columns:
                df[col] = df[col].where(pd.notna(df[col])).astype('Int64')
    
    # Designations DataFrame
    elif node_label == 'Designation':
        if 'id' in df.columns:
            df['id'] = df['id'].astype('int64')
    
    # Endpoints DataFrame
    elif node_label == 'EndpointSystem':
        if 'id' in df.columns:
            df['id'] = df['id'].astype('int64')
        
        nullable_int_cols = ['OwnerUserId', 'ServiceAccountId']
        for col in nullable_int_cols:
            if col in df.columns:
                df[col] = df[col].where(pd.notna(df[col])).astype('Int64')
    
    # Accounts DataFrame
    elif node_label == 'Account':
        if 'id' in df.columns:
            df['id'] = df['id'].astype('int64')
        
        nullable_int_cols = ['UserId', 'EndpointSystemId']
        for col in nullable_int_cols:
            if col in df.columns:
                df[col] = df[col].where(pd.notna(df[col])).astype('Int64')
    
    return df

def get_all_graph_data():
    """Connects to Neo4j and pulls only the ACTIVE, REAL data needed for the ML pipeline."""
    driver = get_neo4j_driver()
    print("\n🚀 Fetching ACTIVE user data from the complete Neo4j graph replica...")
    
    # Filter for active users only (excludes phantom users)
    active_user_filter = "WHERE n.IsActive = true AND NOT n.UserName CONTAINS 'PHANTOM' AND NOT n.UserName = 'ORPHAN_USER'"
    
    graph_dfs = {
        'users': get_nodes_as_df(driver, 'User', active_user_filter),
        'entitlements': get_nodes_as_df(driver, 'Entitlement'),  # Includes phantom entitlements
        'orgs': get_nodes_as_df(driver, 'Organization'),
        'endpoints': get_nodes_as_df(driver, 'EndpointSystem'),
        'designations': get_nodes_as_df(driver, 'Designation'),
        'accounts': get_nodes_as_df(driver, 'Account')
    }

    print("Fetching active user access relationships for training labels...")
    with driver.session(database=config.NEO4J_DATABASE) as session:
        # Relationships now go in correct direction: User -> Entitlement
        query = """
        MATCH (u:User)-[:HAS_ACCESS_TO]->(e:Entitlement)
        WHERE u.IsActive = true 
        AND NOT u.UserName CONTAINS 'PHANTOM' 
        AND NOT u.UserName = 'ORPHAN_USER'
        RETURN u.id AS UserId, e.id AS EntitlementId
        """
        results = session.run(query).data()
        entrecon_df = pd.DataFrame(results).drop_duplicates() if results else pd.DataFrame(columns=['UserId', 'EntitlementId'])
        
        # CRITICAL: Apply type standardization to relationship data
        if not entrecon_df.empty:
            entrecon_df['UserId'] = entrecon_df['UserId'].astype('int64')
            entrecon_df['EntitlementId'] = entrecon_df['EntitlementId'].astype('string')
        
        graph_dfs['entrecon'] = entrecon_df

    driver.close()
    
    # Debug output to verify we're getting the right data
    print(f"✅ ML Training data fetched from Neo4j (with proper types):")
    for key, df in graph_dfs.items():
        if not df.empty and 'id' in df.columns:
            id_type = df['id'].dtype
            print(f"   {key}: {len(df)} rows (id type: {id_type})")
        else:
            print(f"   {key}: {len(df)} rows")
    
    # Specifically check if we have training relationships
    if len(graph_dfs['entrecon']) == 0:
        print("❌ ERROR: No HAS_ACCESS_TO relationships found for active users!")
        print("   This suggests either:")
        print("   1. No active users have entitlements, or")
        print("   2. All entitlements are phantom entitlements")
        print("   3. The ETL didn't create relationships properly")
        
        # Debug query to see what we have
        with driver.session(database=config.NEO4J_DATABASE) as session:
            result = session.run("MATCH (u:User)-[:HAS_ACCESS_TO]->(e:Entitlement) RETURN count(*) as total").single()
            print(f"   Total HAS_ACCESS_TO relationships in database: {result['total']}")
            
            result = session.run("MATCH (u:User) WHERE u.IsActive = true RETURN count(u) as activeUsers").single()
            print(f"   Total active users: {result['activeUsers']}")
            
            result = session.run("MATCH (e:Entitlement) WHERE NOT e.Name CONTAINS 'Phantom' RETURN count(e) as realEnts").single()
            print(f"   Total real (non-phantom) entitlements: {result['realEnts']}")
    else:
        print(f"✅ Found {len(graph_dfs['entrecon'])} access relationships for ML training")
    
    return graph_dfs

def get_neo4j_embeddings():
    """Connects to Neo4j and runs GDS Node2Vec on the entire projected graph."""
    driver = get_neo4j_driver()
    graph_name = 'iam-graph-projection'
    
    # Drop any existing projection first
    with driver.session(database=config.NEO4J_DATABASE) as session:
        try:
            session.run(f"CALL gds.graph.drop('{graph_name}')")
        except:
            pass  # Graph might not exist
    
    project_query = f"CALL gds.graph.project('{graph_name}', '*', '*')"
    
    # FIXED: Add DISTINCT to prevent duplicates and proper node ID handling
    stream_query = f"""
    CALL gds.node2vec.stream('{graph_name}', {{
        embeddingDimension: {config.EMBEDDING_DIMENSION},
        walkLength: 40,
        walksPerNode: 15
    }})
    YIELD nodeId, embedding
    WITH DISTINCT nodeId, embedding
    RETURN gds.util.asNode(nodeId).id AS originalId, embedding
    """
    
    drop_query = f"CALL gds.graph.drop('{graph_name}')"
    
    with driver.session(database=config.NEO4J_DATABASE) as session:
        print(f"\n🚀 Creating GDS graph projection of the COMPLETE graph...")
        session.run(project_query)
        
        print(f"🚀 Streaming Node2Vec embeddings with DEDUPLICATION...")
        results = session.run(stream_query).data()
        
        print(f"🚀 Dropping GDS graph projection '{graph_name}'...")
        session.run(drop_query)
    
    driver.close()
    
    if not results:
        raise RuntimeError("Neo4j GDS did not return any embeddings.")
        
    embeddings_df = pd.DataFrame(results)
    
    # CRITICAL: Additional deduplication at DataFrame level
    print(f"📊 Raw embeddings received: {len(embeddings_df)}")
    embeddings_df = embeddings_df.drop_duplicates(subset=['originalId'])
    print(f"📊 After deduplication: {len(embeddings_df)}")
    
    # Verify no duplicates remain
    duplicates = embeddings_df['originalId'].duplicated().sum()
    if duplicates > 0:
        print(f"⚠️ WARNING: Still {duplicates} duplicates found, removing...")
        embeddings_df = embeddings_df.drop_duplicates(subset=['originalId'])
    
    # IMPORTANT: Ensure embedding IDs have consistent types
    # Convert to string for entitlement IDs, int for user IDs
    print("🔧 Standardizing embedding ID types...")
    
    # Check which IDs are in the embeddings
    sample_ids = embeddings_df['originalId'].head(10).tolist()
    print(f"📊 Sample embedding IDs: {sample_ids}")
    
    # Convert composite entitlement IDs to string, numeric user IDs to int
    def standardize_embedding_id(eid):
        """FIXED: Proper embedding ID standardization without float conversion issues"""
        if pd.isna(eid):
            return eid
        
        eid_str = str(eid)
        
        if '_' in eid_str:  
            # Composite entitlement ID - keep as string
            return eid_str
        else:  
            # Numeric user/org/designation ID - convert to int
            try:
                # ✅ FIXED: Direct int conversion without float step
                if '.' in eid_str:
                    # If it's a float string, convert to int directly
                    return int(float(eid_str))
                else:
                    return int(eid_str)
            except (ValueError, OverflowError):
                # If conversion fails, return as string
                return eid_str

    embeddings_df['originalId'] = embeddings_df['originalId'].apply(standardize_embedding_id)
    
    print(f"✅ Final unique embeddings: {len(embeddings_df)}")
    return embeddings_df