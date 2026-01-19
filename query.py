import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from conclave.agent.control import ControlAgent

logging.basicConfig(level=logging.ERROR)
console = logging.getLogger("Conclave.Query")
console.setLevel(logging.INFO)

class AdvancedGraphQuerier:
    def __init__(self, config_path: str, video_id: str):
        self.agent = ControlAgent(config_path, video_id)
        self.engine = self.agent.engine 

    def _get_entity_details(self, entity_id: str) -> str:
        """Fetch Type and Alias (Role) from Graph"""
        query = "MATCH (e:Entity {id: $eid}) RETURN e.type as type, e.alias as alias"
        res = self.engine.graph_store.run_query(query, {"eid": entity_id})
        if not res: return "Unknown Entity"
        
        data = res[0]
        role = data.get('alias')
        etype = data.get('type')
        
        # Friendly formatting
        if etype == 'pov_user':
            return "📷 CAMERA OPERATOR (Self)"
        if role:
            return f"{role} ({etype})"
        return f"Unidentified {etype}"

    def expand_graph_context(self, primary_results: List[Dict[str, Any]]) -> str:
        if not primary_results: return "No direct memories found."

        expanded_context = []
        seen_entities = set()

        for res in primary_results:
            clip_id = res['clip_id']
            content = res['content']
            
            expanded_context.append(f"\n📍 [Clip {clip_id}] {content}")
            
            involved_ids = [e['id'] for e in res['involved_entities']]
            
            for eid in involved_ids:
                if eid in seen_entities: continue
                seen_entities.add(eid)
                
                # FETCH IDENTITY DETAILS (The Fix)
                identity_desc = self._get_entity_details(eid)
                
                expanded_context.append(f"   └── 👤 Entity {eid}: {identity_desc}")
                
                # Fetch recent speech
                dial_query = """
                MATCH (e:Entity {id: $eid})-[:SPOKE]->(d:Dialogue)
                WHERE abs(d.ts - $clip_ts) < 30000 
                RETURN d.content as speech
                ORDER BY d.ts LIMIT 1
                """
                clip_ts = clip_id * 30 * 1000 
                speech = self.engine.graph_store.run_query(dial_query, {"eid": eid, "clip_ts": clip_ts})
                
                if speech:
                    expanded_context.append(f"       (Spoke): \"{speech[0]['speech']}\"")

        return "\n".join(expanded_context)

    def interactive_session(self):
        print(f"\n🤖 Conclave Memory System")
        print(f"   Context: Video {self.agent.video_id}")
        print("   (Type 'quit' to exit)\n")

        # Hook the engine
        original_search = self.agent.engine.get_hybrid_context
        def advanced_search_wrapper(vector):
            basic_results = original_search(vector)
            rich_results = []
            for res in basic_results:
                graph_ctx = self.expand_graph_context([res])
                # Inject identity info directly into content for LLM
                res['content'] = f"{res['content']}\n[GRAPH CONTEXT]:\n{graph_ctx}"
                rich_results.append(res)
            return rich_results
        
        self.agent.engine.get_hybrid_context = advanced_search_wrapper

        while True:
            q = input("\n🔎 Ask: ")
            if q.lower() in ["quit", "exit"]: break
            print(f"🧠 Thinking...")
            print(f"\n✨ Answer:\n{self.agent.solve(q)}\n" + "-"*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/api_config.json")
    args = parser.parse_args()
    AdvancedGraphQuerier(args.config, args.video_id).interactive_session()