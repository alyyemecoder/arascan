# app/database/neo4j_connection.py
import os
from neo4j import GraphDatabase, basic_auth

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Create driver with database name
driver = GraphDatabase.driver(
    NEO4J_URI, 
    auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD),
    database=NEO4J_DATABASE
)

def close_driver():
    driver.close()
