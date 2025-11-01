import os
from neo4j import GraphDatabase, basic_auth

def test_neo4j_connection():
    print("Testing Neo4j connection...")
    
    # Get connection details from environment variables
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "12345678")
    database = os.getenv("NEO4J_DATABASE", "arascandb")
    
    print(f"Connecting to: {uri}")
    print(f"Database: {database}")
    
    try:
        print("\nTrying to connect without database name...")
        # First try without database name (will use default)
        driver = GraphDatabase.driver(
            uri,
            auth=basic_auth(user, password)
        )
        with driver.session() as session:
            result = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions[0] as version")
            record = result.single()
            print(f"✅ Connected to Neo4j {record['name']} v{record['version']}")
            
            # Now try to list databases
            try:
                result = session.run("SHOW DATABASES")
                print("\nAvailable databases:")
                for record in result:
                    print(f"- {record['name']} (default: {record.get('default', False)})")
                return True
            except Exception as e:
                print(f"\nℹ️ Could not list databases: {str(e)}")
                return True
                
    except Exception as e:
        print(f"❌ Connection without database name failed: {str(e)}")
        
    # If we get here, try with the specified database
    print(f"\nTrying to connect with database: {database}")
    try:
        driver = GraphDatabase.driver(
            uri,
            auth=basic_auth(user, password),
            database=database
        )
        
        # Run a simple query to verify the connection
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful' AS message")
            record = result.single()
            print(f"✅ {record['message']}")
            
            # List all databases (if permissions allow)
            try:
                result = session.run("SHOW DATABASES")
                print("\nAvailable databases:")
                for record in result:
                    print(f"- {record['name']} (default: {record.get('default', False)})")
            except Exception as e:
                print(f"\nℹ️ Could not list databases: {str(e)}")
                
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False
    finally:
        if 'driver' in locals():
            driver.close()
    
    return True

if __name__ == "__main__":
    test_neo4j_connection()
