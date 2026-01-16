import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
from neo4j import GraphDatabase

def init_conclave():
    with open("configs/api_config.json") as f:
        conf = json.load(f)

    # 1. Initialize Qdrant Collections
    qc = QdrantClient(url=conf["qdrant"]["url"], api_key=conf["qdrant"]["api_key"])
    
    collections = {
        "text_memories": 3072,   # text-embedding-3-large
        "face_memories": 512,    # InsightFace buffalo_l
        "voice_memories": 192,   # WeSpeaker ERes2Net
        "visual_memories": 1152  # SigLIP-so400m
    }

    for name, size in collections.items():
        print(f"[*] Creating Qdrant collection: {name} (dim: {size})")
        qc.recreate_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE)
        )
        # Create indexing for fast filtering by video_id
        qc.create_payload_index(name, "video_id", models.PayloadSchemaType.KEYWORD)

    # 2. Initialize Neo4j Constraints
    print("[*] Setting up Neo4j Constraints...")
    driver = GraphDatabase.driver(conf["neo4j"]["uri"], auth=(conf["neo4j"]["user"], conf["neo4j"]["password"]))
    
    with driver.session() as session:
        # Unique IDs for Entities, Memories, and Videos
        session.run("CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        session.run("CREATE CONSTRAINT memory_id_unique IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE")
        session.run("CREATE CONSTRAINT video_id_unique IF NOT EXISTS FOR (v:Video) REQUIRE v.id IS UNIQUE")
        # Indexing for Clip lookups
        session.run("CREATE INDEX clip_lookup IF NOT EXISTS FOR (c:Clip) ON (c.id, c.video_id)")

    driver.close()
    print("[SUCCESS] Infrastructure Initialized.")

if __name__ == "__main__":
    init_conclave()