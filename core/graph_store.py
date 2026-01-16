import json
from typing import List, Dict, Any
from neo4j import GraphDatabase

class GraphStore:
    def __init__(self, config: Dict[str, Any]):
        self.driver = GraphDatabase.driver(
            config["uri"], 
            auth=(config["user"], config["password"])
        )

    def close(self):
        self.driver.close()

    def run_query(self, query: str, parameters: Dict[str, Any] = None):
        with self.driver.session() as session:
            return session.run(query, parameters).data()

    def create_clip_structure(self, video_id: str, clip_id: int):
        query = """
        MERGE (v:Video {id: $video_id})
        MERGE (c:Clip {id: $clip_id, video_id: $video_id})
        MERGE (v)-[:HAS_CLIP]->(c)
        """
        self.run_query(query, {"video_id": video_id, "clip_id": clip_id})

    def create_entity_node(self, entity_id: str, entity_type: str, video_id: str):
        query = """
        MERGE (e:Entity {id: $entity_id})
        ON CREATE SET e.type = $type, e.video_id = $video_id
        """
        self.run_query(query, {"entity_id": entity_id, "type": entity_type, "video_id": video_id})

    def create_memory_node(self, mem_id: str, content: str, mem_type: str, video_id: str, clip_id: int):
        query = """
        MATCH (c:Clip {id: $clip_id, video_id: $video_id})
        CREATE (m:Memory {id: $mem_id, content: $content, type: $mem_type})
        CREATE (c)-[:HAS_MEMORY]->(m)
        """
        self.run_query(query, {
            "mem_id": mem_id, "content": content, "mem_type": mem_type,
            "video_id": video_id, "clip_id": clip_id
        })

    def link_memory_to_entity(self, mem_id: str, entity_id: str, rel_type: str = "MENTIONS"):
        query = f"""
        MATCH (m:Memory {{id: $mem_id}}), (e:Entity {{id: $entity_id}})
        MERGE (m)-[:{rel_type}]->(e)
        """
        self.run_query(query, {"mem_id": mem_id, "entity_id": entity_id})