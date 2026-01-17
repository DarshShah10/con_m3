import json
import logging
import re
from typing import List, Dict, Any, Optional
import openai
from conclave.core.engine import ConclaveEngine
from conclave.core.embedding_service import EmbeddingService

logger = logging.getLogger("Conclave.Control")

class ControlAgent:
    """
    The Brain (M3-Control implementation).
    Instead of just searching once, it:
    1. Translates User Query -> System Query (Entity IDs)
    2. Plans: Decides if it needs to Search or Answer.
    3. Loops: Iteratively gathers clues.
    """
    def __init__(self, config_path: str, video_id: str):
        with open(config_path, "r") as f: self.config = json.load(f)
        self.video_id = video_id
        
        # Connect to existing Engine
        self.engine = ConclaveEngine(video_id=video_id, config_path=config_path)
        self.embedder = EmbeddingService(self.config["api"].get("openai_api_key"))
        
        # LLM Client
        self.client = openai.OpenAI(api_key=self.config["api"].get("openai_api_key"))
        self.model = self.config["api"].get("model", "gpt-4o")

    # --- 1. QUERY TRANSLATION (The M3 "Back Translate" Logic) ---
    def _translate_query(self, user_query: str) -> str:
        """
        M3 Logic: The user asks about "The guy in the red hat".
        Memory stores "<ent_face_05>". 
        Vector search fails here unless we translate.
        """
        # 1. Fetch all known entity descriptions from Graph
        query = """
        MATCH (e:Entity {video_id: $vid})
        RETURN e.id as id, e.alias as alias
        """
        entities = self.engine.graph_store.run_query(query, {"vid": self.video_id})
        
        # If we have too many entities, we might filter, but for now dump them.
        entity_context = "\n".join([f"ID: {e['id']} | Description/Alias: {e.get('alias', 'Unknown')}" for e in entities])

        prompt = f"""
        You are a Query Translator for a Video Memory System.
        User Query: "{user_query}"
        
        Known Entities in DB:
        {entity_context}
        
        Task: Rewrite the user query to use Entity IDs (e.g., <ent_face_...>) where appropriate.
        If no specific entity matches, keep the text description.
        Output ONLY the rewritten query.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            translated = response.choices[0].message.content.strip()
            logger.info(f"🔄 Query Translation: '{user_query}' -> '{translated}'")
            return translated
        except:
            return user_query

    # --- 2. ITERATIVE RETRIEVAL LOOP (The M3 "Action" Logic) ---
    def solve(self, user_query: str, max_steps: int = 5) -> str:
        """
        The Thinking Loop:
        Action: [Search] "query..." -> vector store
        Action: [Search] "query..." -> graph store
        Action: [Answer] "Final answer"
        """
        # 1. Translate first
        current_query = self._translate_query(user_query)
        
        context_memory = [] # Accumulates knowledge
        
        system_prompt = """
        You are the Controller of a Multimodal Memory System.
        Goal: Answer the user's question using the Memory Bank.
        
        Process:
        1. Analyze the current Context.
        2. Decide your next Action:
           - [Search] <query>: If you need more info. Be specific. Use Entity IDs.
           - [Answer] <text>: If you have enough info to answer.
        
        The memory bank contains:
        - Episodic Memory (Time-based events)
        - Dialogue (Transcripts)
        - Entity Graph (Relationships)
        """
        
        for step in range(max_steps):
            # Format context for LLM
            context_str = "\n".join([f"Info {i+1}: {c}" for i, c in enumerate(context_memory)])
            
            user_prompt = f"""
            Question: {user_query}
            (Translated: {current_query})
            
            Current Context:
            {context_str if context_memory else "No context yet."}
            
            What is your next Action?
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            output = response.choices[0].message.content.strip()
            logger.info(f"🤖 Step {step+1}: {output}")
            
            # PARSE ACTION
            if "[Answer]" in output:
                return output.split("[Answer]")[1].strip()
            
            elif "[Search]" in output:
                search_q = output.split("[Search]")[1].strip()
                
                # Perform Hybrid Search (Vector + Graph)
                # 1. Vector Search (Semantic)
                vec_results = self.engine.get_hybrid_context(
                    self.engine.embedding_service.get_embeddings_batched([search_q])[0]
                )
                
                # 2. Add results to context
                if not vec_results:
                    context_memory.append(f"Search for '{search_q}' returned no results.")
                else:
                    # Summarize findings to save context window
                    findings = [f"Clip {r['clip_id']}: {r['content']} (Entities: {r['involved_entities']})" for r in vec_results[:3]]
                    context_memory.append(f"Search '{search_q}':\n" + "\n".join(findings))
            
            else:
                # Fallback if LLM hallucinates format
                context_memory.append("Error: Invalid Action format. Please use [Search] or [Answer].")

        return "I could not find the answer after multiple reasoning steps."

if __name__ == "__main__":
    # Test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--video_id", type=str, required=True)
    args = parser.parse_args()
    
    agent = ControlAgent("configs/api_config.json", args.video_id)
    answer = agent.solve(args.query)
    print("\nFINAL ANSWER:\n", answer)
