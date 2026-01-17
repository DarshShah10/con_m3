import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any
import openai

# Dynamic path handling
import os
import sys

# Add parent directory to path for package imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from conclave.core.engine import ConclaveEngine
from conclave.core.embedding_service import EmbeddingService

# Configure logging to be clean for CLI output
logging.basicConfig(level=logging.ERROR) 

class ConclaveQuerier:
    def __init__(self, config_path: str, video_id: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.video_id = video_id
        self.engine = ConclaveEngine(video_id=video_id, config_path=config_path)
        
        # We need the embedder to turn the question into a vector
        self.embedder = EmbeddingService(self.config["text-embedding-3-large"])
        
        # LLM Client
        api_key = self.config["api"].get("openai_api_key") or self.config["api"].get("api_key")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = self.config["api"].get("model", "gpt-4o")

    def retrieve_knowledge(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs GraphRAG: Vector Search -> Graph Context Expansion
        """
        print(f"🔍 Searching Memory Bank for: '{query_text}'...")
        
        # 1. Embed Question
        query_vec = self.embedder.get_embeddings_batched([query_text])[0]
        
        # 2. Vector Search (Qdrant) - Find relevant episodic/semantic memories
        # We search 'text_memories' which contains the synthesis of Visual+Face+Voice
        hits = self.engine.vector_store.search(
            collection="text_memories",
            vector=query_vec,
            filter_kv={"video_id": self.video_id},
            limit=top_k
        )
        
        context_results = []
        
        for hit in hits:
            mem_id = hit.id
            score = hit.score
            content = hit.payload.get("content")
            clip_id = hit.payload.get("clip_id")
            
            # 3. Graph Expansion (Neo4j)
            # Find entities mentioned in this memory AND entities physically present in the clip
            query = """
            MATCH (m:Memory {id: $mem_id})
            // Get explicitly linked entities
            OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
            WITH m, collect(e.id) as mentioned_entities
            
            // Get co-occurring entities in the same clip (Spatial Context)
            MATCH (c:Clip {id: $clip_id, video_id: $video_id})
            OPTIONAL MATCH (p:Entity)-[r:APPEARED_IN]->(c)
            
            RETURN mentioned_entities, collect({id: p.id, type: p.type, ts: r.ts_ms}) as present_entities
            """
            
            graph_data = self.engine.graph_store.run_query(query, {
                "mem_id": mem_id, 
                "clip_id": clip_id, 
                "video_id": self.video_id
            })
            
            context_results.append({
                "clip_id": clip_id,
                "score": score,
                "memory_text": content,
                "graph_context": graph_data[0] if graph_data else {}
            })
            
        return context_results

    def generate_answer(self, user_query: str, context: List[Dict[str, Any]]):
        """
        Synthesizes the answer using LLM + Context
        """
        if not context:
            print("❌ No relevant information found in the video memory.")
            return

        # Format context for the LLM
        context_str = ""
        for item in sorted(context, key=lambda x: x['clip_id']):
            clip_idx = item['clip_id']
            text = item['memory_text']
            
            # Parse graph data
            g = item['graph_context']
            mentions = g.get('mentioned_entities', [])
            present = [p['id'] for p in g.get('present_entities', []) if p['id']]
            
            # Deduplicate entities
            all_entities = list(set(mentions + present))
            
            context_str += f"""
            [Clip ID: {clip_idx}]
            Description: {text}
            Related Entities (IDs): {all_entities}
            ------------------------------------------------
            """

        system_prompt = """
        You are Conclave, an advanced video intelligence AI.
        Answer the user's question based strictly on the provided Context.
        
        FORMATTING RULES:
        1. **Direct Answer**: Start with a clear, direct answer.
        2. **Reasoning**: Explain *why* you concluded this based on the visual/audio evidence in the context.
        3. **Citations**: You MUST cite the [Clip ID] for every claim.
        4. **Timeline**: If describing a sequence, provide a chronological breakdown.
        5. **Entities**: If referring to a specific person/object ID (e.g., ent_face_...), refer to them as "Person <ID>" unless a name is available in the text.
        
        If the information is missing, state clearly that it is not in the processed video memory.
        """

        print("🧠 Reasoning...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_query}"}
                ],
                temperature=0.3 # Low temperature for factual accuracy
            )
            
            answer = response.choices[0].message.content
            
            print("\n" + "="*60)
            print(f"🤖 CONCLAVE ANSWER")
            print("="*60)
            print(answer)
            print("="*60)
            
        except Exception as e:
            print(f"Error generating answer: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Conclave Memory System")
    parser.add_argument("--query", type=str, required=True, help="Your question about the video")
    parser.add_argument("--video_id", type=str, required=True, help="The ID used during processing")
    parser.add_argument("--config", type=str, default="configs/api_config.json")
    
    args = parser.parse_args()
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        sys.exit(1)

    querier = ConclaveQuerier(args.config, args.video_id)
    
    # 1. Retrieve
    knowledge = querier.retrieve_knowledge(args.query)
    
    # 2. Answer
    querier.generate_answer(args.query, knowledge)