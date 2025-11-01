# File: app/services/neo4j_service.py
from neo4j import GraphDatabase, exceptions
import os
import logging
from typing import List, Dict, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", os.getenv("NEO4J_PASS", "12345678"))  # Support both NEO4J_PASS and NEO4J_PASSWORD
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Initialize driver
driver = None

def init_driver():
    """Initialize the Neo4j driver."""
    global driver
    if driver is not None:
        return driver
        
    try:
        # Convert neo4j:// to bolt:// if needed
        uri = NEO4J_URI
        if uri.startswith('neo4j://'):
            uri = uri.replace('neo4j://', 'bolt://')
            
        driver = GraphDatabase.driver(
            uri,
            auth=(NEO4J_USER, NEO4J_PASS),
            max_connection_lifetime=30,  # 30 seconds
            max_connection_pool_size=50,
            connection_timeout=15  # 15 seconds
        )
        
        # Test the connection
        with driver.session(database=NEO4J_DATABASE) as session:
            session.run("RETURN 1").single()
            
        logger.info("Successfully connected to Neo4j database")
        return driver
        
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j driver: {e}")
        driver = None
        return None

# Initialize on import
init_driver()

def get_letter_node(letter_name: str) -> Optional[Dict]:
    """Get a letter node with its properties."""
    if not driver:
        logger.warning("No database connection available")
        return None

    query = """
    MATCH (l:Letter {char: $char})
    RETURN l { 
        .char, 
        .name,
        .examples,
        .audio,
        .description,
        .pronunciation
    } AS node
    """
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.execute_read(
                lambda tx: tx.run(query, char=letter_name).single()
            )
            return result["node"] if result else None
    except Exception as e:
        logger.error(f"Error getting letter node for '{letter_name}': {e}")
        return None

def get_similar_letters(letter: str, limit: int = 3) -> List[Dict]:
    """Get letters that are visually similar to the given letter."""
    if not driver:
        logger.warning("No database connection available")
        return []

    query = """
    MATCH (a:Letter {char: $char})-[r:SIMILAR_TO]-(b:Letter)
    RETURN b {
        .char,
        .name,
        .examples,
        similarity: r.similarity
    } AS similar_letter
    ORDER BY r.similarity DESC
    LIMIT $limit
    """
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, char=letter, limit=limit)
            return [record["similar_letter"] for record in result]
    except Exception as e:
        print(f"Error getting similar letters: {e}")
        return []

def get_related_words(letter: str, limit: int = 5) -> List[Dict]:
    """Get example words containing the given letter."""
    if not driver:
        logger.warning("No database connection available")
        return []
        
    query = """
    MATCH (l:Letter {char: $char})<-[:CONTAINS]-(w:Word)
    RETURN w {
        .text,
        .translation,
        .audio
    } AS word
    ORDER BY length(w.text)
    LIMIT $limit
    """
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, char=letter, limit=limit)
            return [record["word"] for record in result]
    except Exception as e:
        print(f"Error getting related words: {e}")
        return []

def get_letter_family(letter: str) -> List[Dict]:
    """Get letters that belong to the same family (similar shapes)."""
    if not driver:
        logger.warning("No database connection available")
        return []

    query = """
    MATCH (a:Letter {char: $char})-[:FAMILY_OF]-(b:Letter)
    RETURN b {
        .char,
        .name,
        .examples
    } AS family_member
    """
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, char=letter)
            return [record["family_member"] for record in result]
    except Exception as e:
        logger.error(f"Error getting letter family: {e}")
        return []

def get_letter_suggestions(letter: str) -> Dict[str, Any]:
    """
    Get comprehensive learning suggestions for a letter.
    Returns: {
        letter: str,
        details: dict,
        similar_letters: List[dict],
        family_members: List[dict],
        example_words: List[dict],
        audio_url: Optional[str]
    }
    """
    if not driver:
        logger.warning("No database connection available")
        return {
            "letter": letter,
            "error": "Database connection not available",
            "details": None,
            "similar_letters": [],
            "family_members": [],
            "example_words": [],
            "audio_url": None
        }

    # Get all data in parallel
    with driver.session(database=NEO4J_DATABASE) as session:
        # Get letter details
        letter_data = get_letter_node(letter)
        
        # Get similar letters
        similar = get_similar_letters(letter)
        
        # Get family members
        family = get_letter_family(letter)
        
        # Get example words
        examples = get_related_words(letter)
        
        # Construct audio URL if available
        audio_url = None
        if letter_data and letter_data.get("audio"):
            audio_url = f"/static/audio/{letter_data['audio']}"
        
        return {
            "letter": letter,
            "details": letter_data,
            "similar_letters": similar,
            "family_members": family,
            "example_words": examples,
            "audio_url": audio_url
        }
