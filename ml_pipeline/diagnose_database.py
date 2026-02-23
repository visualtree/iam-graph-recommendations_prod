# # diagnostic_script.py
# import pandas as pd
# from neo4j import GraphDatabase

# # Configuration (update with your credentials)
# NEO4J_URI = "bolt://localhost:7687"
# NEO4J_USER = "neo4j"
# NEO4J_PASS = "@bcd1234"
# NEO4J_DATABASE = "neo4j"

# def get_neo4j_driver():
    # driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    # driver.verify_connectivity()
    # return driver

# def diagnose_database():
    # driver = get_neo4j_driver()
    
    # with driver.session(database=NEO4J_DATABASE) as session:
        # print("=== DIAGNOSTIC REPORT ===\n")
        
        # # 1. Check total node counts
        # print("1. Node Counts:")
        # result = session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as count ORDER BY count DESC")
        # for record in result:
            # print(f"   {record['label']}: {record['count']}")
        
        # # 2. Check User nodes and their IsActive values
        # print("\n2. User Node IsActive Analysis:")
        # result = session.run("MATCH (u:User) RETURN u.IsActive as isActive, count(u) as count ORDER BY count DESC")
        # for record in result:
            # print(f"   IsActive = {record['isActive']} (type: {type(record['isActive']).__name__}): {record['count']} users")
        
        # # 3. Sample User properties
        # print("\n3. Sample User Properties:")
        # result = session.run("MATCH (u:User) RETURN u LIMIT 3")
        # for i, record in enumerate(result, 1):
            # user = record['u']
            # print(f"   User {i}: ID={user.get('Id')}, IsActive={user.get('IsActive')} (type: {type(user.get('IsActive')).__name__})")
        
        # # 4. Check relationship counts
        # print("\n4. Relationship Counts:")
        # result = session.run("MATCH ()-[r]->() RETURN type(r) as relType, count(r) as count ORDER BY count DESC")
        # for record in result:
            # print(f"   {record['relType']}: {record['count']}")
        
        # # 5. Check HAS_ACCESS_TO relationships specifically
        # print("\n5. HAS_ACCESS_TO Relationship Analysis:")
        # result = session.run("MATCH (u:User)-[:HAS_ACCESS_TO]->(e:Entitlement) RETURN count(*) as total")
        # total_access = result.single()['total']
        # print(f"   Total HAS_ACCESS_TO relationships: {total_access}")
        
        # if total_access > 0:
            # # Check IsActive values for users with access
            # result = session.run("""
                # MATCH (u:User)-[:HAS_ACCESS_TO]->(e:Entitlement) 
                # RETURN u.IsActive as isActive, count(*) as accessCount 
                # ORDER BY accessCount DESC
            # """)
            # print("   IsActive distribution for users with access:")
            # for record in result:
                # print(f"     IsActive = {record['isActive']}: {record['accessCount']} access relationships")
        
        # # 6. Test different filter approaches
        # print("\n6. Testing Filter Approaches:")
        # filters = [
            # ("u.IsActive = 1", "Direct integer comparison"),
            # ("u.IsActive = true", "Direct boolean comparison"), 
            # ("toString(u.IsActive) = '1'", "String conversion to '1'"),
            # ("toString(u.IsActive) = 'true'", "String conversion to 'true'"),
            # ("u.IsActive IN [1, true, '1', 'true']", "Multiple type match")
        # ]
        
        # for filter_expr, description in filters:
            # result = session.run(f"MATCH (u:User) WHERE {filter_expr} RETURN count(u) as count")
            # count = result.single()['count']
            # print(f"   {description}: {count} users")
            
            # if count > 0:
                # # Test with relationships
                # result = session.run(f"""
                    # MATCH (u:User)-[:HAS_ACCESS_TO]->(e:Entitlement) 
                    # WHERE {filter_expr} 
                    # RETURN count(*) as accessCount
                # """)
                # access_count = result.single()['accessCount']
                # print(f"     -> {access_count} HAS_ACCESS_TO relationships")
    
    # driver.close()
    # print("\n=== END DIAGNOSTIC ===")

# if __name__ == "__main__":
    # diagnose_database()
    
# diagnose_data.py (STANDALONE DIAGNOSTIC SCRIPT)
# diagnose_data.py (STANDALONE DIAGNOSTIC SCRIPT - Corrected Import)

import pandas as pd
from neo4j import GraphDatabase
# VVV THIS IS THE CORRECTED IMPORT VVV
from ml_pipeline import config # This will work when run as a module

def run_diagnostic():
    print("====== RUNNING FINAL DATA DIAGNOSTIC ======")
    
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))
    driver.verify_connectivity()
    print("✅ Neo4j connection successful.")

    with driver.session(database=config.NEO4J_DATABASE) as session:
        # --- Step 1: Check for Active Users ---
        print("\n--- Step 1: Checking for Active Users ---")
        active_user_query = "MATCH (u:User) WHERE toString(u.IsActive) = '1' OR toString(u.IsActive) = 'true' RETURN count(u) AS activeUserCount"
        result = session.run(active_user_query).single()
        active_user_count = result['activeUserCount']
        print(f"Found {active_user_count} User nodes with IsActive = true or '1'.")
        
        # --- Step 2: Check for ANY HAS_ACCESS_TO relationships ---
        print("\n--- Step 2: Checking for ANY HAS_ACCESS_TO relationships (unfiltered) ---")
        total_rels_query = "MATCH ()-[r:HAS_ACCESS_TO]->() RETURN count(r) AS totalRels"
        result = session.run(total_rels_query).single()
        total_rels_count = result['totalRels']
        print(f"Found {total_rels_count} total HAS_ACCESS_TO relationships in the graph.")
        
        # --- Step 3: The CRITICAL TEST ---
        print("\n--- Step 3: Checking for HAS_ACCESS_TO relationships connected to ACTIVE users ---")
        active_rels_query = """
        MATCH (u:User)-[:HAS_ACCESS_TO]->(e:Entitlement)
        WHERE toString(u.IsActive) = '1' OR toString(u.IsActive) = 'true'
        RETURN count(u) AS activeRelCount
        """
        result = session.run(active_rels_query).single()
        active_rels_count = result['activeRelCount']
        print(f"Found {active_rels_count} HAS_ACCESS_TO relationships connected to an Active User.")

    driver.close()
    
    print("\n\n====== DIAGNOSIS COMPLETE ======")
    if active_user_count > 0 and total_rels_count > 0 and active_rels_count == 0:
        print("💥💥💥 ROOT CAUSE CONFIRMED 💥💥💥")
        print("The data exists, but there is NO OVERLAP between the set of 'Active Users' and the set of 'Users with Access'.")
        print("This means all existing access relationships belong to users who are marked as Inactive in the graph.")
        print("To fix this, the training data (positive examples) MUST be gathered from ALL users, not just active ones.")
    elif total_rels_count == 0:
        print("❌ CRITICAL FAILURE: The ETL script did not create any HAS_ACCESS_TO relationships. The ETL script is the problem.")
    else:
        print("❓ An unknown issue exists that this diagnostic did not catch.")

if __name__ == '__main__':
    run_diagnostic()