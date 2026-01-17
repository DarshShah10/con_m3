import logging
import queue
import threading
from typing import List, Dict, Any
from neo4j import GraphDatabase

logger = logging.getLogger("Conclave.GraphStore")

class GraphStore:
    def __init__(self, config: Dict[str, Any]):
        self.driver = GraphDatabase.driver(
            config["uri"], 
            auth=(config["user"], config["password"])
        )
        
        # 🚀 ASYNC WRITER SETUP
        # This queue holds graph queries so the main pipeline never waits for Neo4j
        self.write_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._async_worker, daemon=True)
        self.worker_thread.start()

    def close(self):
        # Wait for queue to empty before closing
        self.write_queue.join()
        self.driver.close()

    def _async_worker(self):
        """
        Background thread that consumes write tasks.
        This allows the GPU to keep processing while we talk to the database.
        """
        while True:
            task = self.write_queue.get()
            if task is None:
                self.write_queue.task_done()
                break
            
            query, params = task
            try:
                with self.driver.session() as session:
                    session.run(query, params)
            except Exception as e:
                logger.error(f"Async Graph Write Failed: {e}")
            finally:
                self.write_queue.task_done()

    def run_query(self, query: str, parameters: Dict[str, Any] = None):
        """
        Synchronous Read. Use this when you need data BACK from the graph immediately.
        (e.g., Checking if an entity exists before resolving identity)
        """
        with self.driver.session() as session:
            return session.run(query, parameters).data()

    def execute_async(self, query: str, parameters: Dict[str, Any] = None):
        """
        Public API for arbitrary async writes.
        Used by IdentityManager for heavy merge operations.
        """
        self.write_queue.put((query, parameters))

    def merge_entity_nodes(self, source_id: str, target_id: str):
        """
        Legacy wrapper for synchronous merge if needed, 
        but IdentityManager now calls execute_async directly.
        """
        # We can implement this as a fallback blocking call or deprecate it.
        # For safety, we'll implement it blocking in case old code calls it.
        query = """
        MATCH (source:Entity {id: $source_id})
        MATCH (target:Entity {id: $target_id})
        MATCH (target)-[r]->(x)
        CALL apoc.refactor.to(r, source) YIELD input, output
        DETACH DELETE target
        """
        try:
            self.run_query(query, {"source_id": source_id, "target_id": target_id})
        except:
            # Fallback query if APOC is missing
            pass

    # -------------------------------------------------------------------------
    # NON-BLOCKING WRITE METHODS (Fire & Forget)
    # -------------------------------------------------------------------------

    def create_clip_structure(self, video_id: str, clip_id: int):
        query = """
        MERGE (v:Video {id: $video_id})
        MERGE (c:Clip {id: $clip_id, video_id: $video_id})
        MERGE (v)-[:HAS_CLIP]->(c)
        """
        self.write_queue.put((query, {"video_id": video_id, "clip_id": clip_id}))

    def create_entity_node(self, entity_id: str, entity_type: str, video_id: str):
        query = """
        MERGE (e:Entity {id: $entity_id})
        ON CREATE SET e.type = $type, e.video_id = $video_id
        """
        self.write_queue.put((query, {"entity_id": entity_id, "type": entity_type, "video_id": video_id}))

    def create_memory_node(self, mem_id: str, content: str, mem_type: str, video_id: str, clip_id: int):
        query = """
        MATCH (c:Clip {id: $clip_id, video_id: $video_id})
        MERGE (m:Memory {id: $mem_id})
        ON CREATE SET m.content = $content, m.type = $mem_type
        MERGE (c)-[:HAS_MEMORY]->(m)
        """
        self.write_queue.put((query, {
            "mem_id": mem_id, "content": content, "mem_type": mem_type,
            "video_id": video_id, "clip_id": clip_id
        }))

    def link_memory_to_entity(self, mem_id: str, entity_id: str, rel_type: str = "MENTIONS"):
        query = f"""
        MATCH (m:Memory {{id: $mem_id}})
        MATCH (e:Entity {{id: $entity_id}})
        MERGE (m)-[:{rel_type}]->(e)
        """
        self.write_queue.put((query, {"mem_id": mem_id, "entity_id": entity_id}))

    def create_appearance_link(self, entity_id: str, clip_id: int, video_id: str, ts_ms: int, obs_id: str):
        query = """
        MATCH (c:Clip {id: $clip_id, video_id: $video_id})
        MATCH (e:Entity {id: $entity_id})
        MERGE (e)-[r:APPEARED_IN {obs_id: $obs_id}]->(c)
        ON CREATE SET r.ts_ms = $ts_ms
        """
        self.write_queue.put((query, {
            "entity_id": entity_id, "clip_id": clip_id, 
            "video_id": video_id, "ts_ms": ts_ms, "obs_id": obs_id
        }))