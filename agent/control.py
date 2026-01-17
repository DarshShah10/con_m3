import json
import logging
import openai
from conclave.core.engine import ConclaveEngine

logger = logging.getLogger("Conclave.Control")

class ControlAgent:
    def __init__(self, config_path: str, video_id: str):
        with open(config_path, "r") as f: self.config = json.load(f)
        self.video_id = video_id
        self.engine = ConclaveEngine(video_id=video_id, config_path=config_path)
        self.client = openai.OpenAI(api_key=self.config["api"].get("openai_api_key"))
        self.model = self.config["api"].get("model", "gpt-4o")

    def _translate_query(self, user_query: str) -> str:
        entities = self.engine.graph_store.run_query(
            "MATCH (e:Entity {video_id: $vid}) RETURN e.id as id, e.alias as alias", 
            {"vid": self.video_id}
        )
        context = "\n".join([f"{e['id']}: {e.get('alias','Unknown')}" for e in entities[:50]])
        prompt = f"Rewrite query using IDs.\nEntities:\n{context}\nQuery: {user_query}"
        try:
            return self.client.chat.completions.create(
                model=self.model, messages=[{"role":"user","content":prompt}]
            ).choices[0].message.content
        except: return user_query

    def solve(self, user_query: str, max_steps: int = 4) -> str:
        q = self._translate_query(user_query)
        context = []
        for _ in range(max_steps):
            ctx_str = "\n".join(context)
            prompt = f"Question: {q}\nContext: {ctx_str}\nAction ([Search] query / [Answer] text):"
            resp = self.client.chat.completions.create(
                model=self.model, messages=[{"role":"user","content":prompt}]
            ).choices[0].message.content
            
            if "[Answer]" in resp: return resp.split("[Answer]")[1].strip()
            if "[Search]" in resp:
                query = resp.split("[Search]")[1].strip()
                vec = self.engine.embedding_service.get_embeddings_batched([query])[0]
                results = self.engine.get_hybrid_context(vec)
                context.append(f"Search '{query}': {json.dumps([r['content'] for r in results[:3]])}")
            else: return resp # Fallback
        return "Unsure."
