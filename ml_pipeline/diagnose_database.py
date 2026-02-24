"""Standalone Neo4j data diagnostic for training-readiness checks."""

from neo4j import GraphDatabase

from ml_pipeline import config


def run_diagnostic() -> None:
    print("====== RUNNING DATA DIAGNOSTIC ======")

    config.require_neo4j_config()
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASS),
    )
    driver.verify_connectivity()
    print("Neo4j connection successful.")

    with driver.session(database=config.NEO4J_DATABASE) as session:
        print("\n--- Step 1: Active users ---")
        result = session.run(
            "MATCH (u:User) "
            "WHERE toString(u.IsActive) = '1' OR toString(u.IsActive) = 'true' "
            "RETURN count(u) AS activeUserCount"
        ).single()
        active_user_count = result["activeUserCount"]
        print(f"Active users: {active_user_count}")

        print("\n--- Step 2: Total HAS_ACCESS_TO relationships ---")
        result = session.run(
            "MATCH ()-[r:HAS_ACCESS_TO]->() RETURN count(r) AS totalRels"
        ).single()
        total_rels_count = result["totalRels"]
        print(f"Total HAS_ACCESS_TO: {total_rels_count}")

        print("\n--- Step 3: Active users with HAS_ACCESS_TO ---")
        result = session.run(
            "MATCH (u:User)-[:HAS_ACCESS_TO]->(:Entitlement) "
            "WHERE toString(u.IsActive) = '1' OR toString(u.IsActive) = 'true' "
            "RETURN count(u) AS activeRelCount"
        ).single()
        active_rels_count = result["activeRelCount"]
        print(f"HAS_ACCESS_TO from active users: {active_rels_count}")

    driver.close()

    print("\n====== DIAGNOSIS COMPLETE ======")
    if active_user_count > 0 and total_rels_count > 0 and active_rels_count == 0:
        print(
            "Root cause likely: no overlap between active users and users who have "
            "HAS_ACCESS_TO relationships."
        )
    elif total_rels_count == 0:
        print("Critical: ETL produced no HAS_ACCESS_TO relationships.")
    else:
        print("No critical overlap issue detected by this diagnostic.")


if __name__ == "__main__":
    run_diagnostic()
