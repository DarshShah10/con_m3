import queue
import logging
import threading
from typing import Dict, Any
from neo4j import GraphDatabase

logger = logging.getLogger("Conclave.GraphStore")

class GraphStore:
    def __init__(self, config: Dict[str, Any]):
        self.uri = config["uri"]
        logger.info(f"🔌 Connecting to Neo4j at {self.uri}...")
        
        try:
            self.driver = GraphDatabase.driver(
                config["uri"], 
                auth=(config["user"], config["password"])
            )
            self.driver.verify_connectivity()
            logger.info("✅ Neo4j Connection Established.")
        except Exception as e:
            logger.error(f"❌ Neo4j Connection Failed: {e}")
            raise e
            
        self.write_queue = queue.Queue()
        self.worker = threading.Thread(target=self._async_worker, daemon=True)
        self.worker.start()

    def close(self):
        logger.info("🛑 Closing GraphStore... flushing remaining data.")
        self.flush()
        self.driver.close()

    def flush(self):
        """Blocks until all pending writes are finished."""
        self.write_queue.join()

    def _async_worker(self):
        while True:
            task = self.write_queue.get()
            if task is None:
                self.write_queue.task_done()
                break
            
            query, params = task
            try:
                with self.driver.session() as session:
                    session.execute_write(lambda tx: tx.run(query, params))
            except Exception as e:
                logger.error(f"❌ Graph Write Failed: {e}\nQuery: {query}")
            finally:
                self.write_queue.task_done()

    def execute_async(self, q, p=None):
        self.write_queue.put((q, p))

    def run_query(self, q, p=None):
        with self.driver.session() as s:
            return s.run(q, p).data()

    # --- UPDATED METHODS (Using MERGE to fix missing relationships) ---

    def create_clip_structure(self, video_id, clip_id):
        # Explicitly create structure
        q = """
        MERGE (v:Video {id: $v})
        MERGE (c:Clip {id: $c, video_id: $v})
        MERGE (v)-[:HAS_CLIP]->(c)
        """
        self.execute_async(q, {"v":video_id, "c":clip_id})

    def create_entity_node(self, eid, etype, vid):
        q = "MERGE (e:Entity {id: $id}) ON CREATE SET e.type=$t, e.video_id=$v"
        self.execute_async(q, {"id":eid,"t":etype,"v":vid})

    def create_memory_node(self, mid, con, mtype, vid, cid):
        # FIX: MERGE Clip first so Memory has something to attach to
        q = """
        MERGE (v:Video {id: $v})
        MERGE (c:Clip {id: $c, video_id: $v})
        MERGE (v)-[:HAS_CLIP]->(c)
        
        MERGE (m:Memory {id: $m}) 
        ON CREATE SET m.content = $con, m.type = $t 
        MERGE (c)-[:HAS_MEMORY]->(m)
        """
        self.execute_async(q, {"c":cid,"v":vid,"m":mid,"con":con,"t":mtype})

    def link_memory_to_entity(self, mid, eid, rel):
        # Ensure Entity exists before linking (Safety check)
        q = f"""
        MATCH (m:Memory {{id: $m}})
        MERGE (e:Entity {{id: $e}})
        MERGE (m)-[:{rel}]->(e)
        """
        self.execute_async(q, {"m":mid,"e":eid})

    def ingest_dialogue(self, vid, cid, eid, txt, ts):
        # FIX: MERGE Clip first
        q = """
        MERGE (v:Video {id: $v})
        MERGE (c:Clip {id: $c, video_id: $v})
        MERGE (v)-[:HAS_CLIP]->(c)
        
        MERGE (e:Entity {id: $e}) 
        CREATE (d:Dialogue {content: $txt, ts: $ts}) 
        MERGE (e)-[:SPOKE]->(d)
        MERGE (d)-[:OCCURRED_IN]->(c)
        """
        self.execute_async(q, {"c":cid,"v":vid,"e":eid,"txt":txt,"ts":ts})

    def ingest_hierarchical_obs(self, obs):
        # FIX: MERGE Clip first
        q_frame = """
        MERGE (v:Video {id: $video_id})
        MERGE (c:Clip {id: $clip_id, video_id: $video_id})
        MERGE (v)-[:HAS_CLIP]->(c)
        
        MERGE (f:Frame {id: $frame_id})
        SET f.ts = $ts, f.desc = $desc
        MERGE (c)-[:HAS_FRAME]->(f)
        """
        self.execute_async(q_frame, {
            "clip_id": obs.clip_id, "video_id": obs.video_id, 
            "frame_id": f"{obs.video_id}_{obs.ts_ms}", 
            "ts": obs.ts_ms, "desc": obs.scene_description or ""
        })

        for obj in obs.objects:
            self.ingest_rich_object_attributes(obs.video_id, obs.clip_id, obj.label)

    def ingest_rich_object_attributes(self, video_id: str, clip_id: int, label: str):
        clean_label = label.split("(")[0].strip()
        attributes = {}
        if "(" in label and ")" in label:
            attr_str = label.split("(")[1].strip(")")
            parts = attr_str.split(",")
            for p in parts:
                if ":" in p:
                    k, v = p.split(":", 1)
                    attributes[k.strip().lower()] = v.strip()

        # FIX: MERGE Clip first
        query = """
        MERGE (v:Video {id: $video_id})
        MERGE (c:Clip {id: $clip_id, video_id: $video_id})
        MERGE (v)-[:HAS_CLIP]->(c)
        
        MERGE (o:Object {label: $label, video_id: $video_id})
        MERGE (c)-[:CONTAINS]->(o)
        SET o += $props
        """
        self.execute_async(query, {
            "clip_id": clip_id, "video_id": video_id, 
            "label": clean_label, "props": attributes
        })
        
        if "text" in attributes:
            q_text = """
            MATCH (o:Object {label: $label, video_id: $video_id})
            MERGE (t:Text {content: $txt})
            MERGE (o)-[:HAS_TEXT]->(t)
            """
            self.execute_async(q_text, {"label": clean_label, "video_id": video_id, "txt": attributes['text']})

    def update_edge_weight(self, source_id, target_id, delta):
        q = """MATCH (a {id: $s})-[r]->(b {id: $t}) 
               SET r.weight = COALESCE(r.weight, 1.0) + $d"""
        self.execute_async(q, {"s": source_id, "t": target_id, "d": delta})