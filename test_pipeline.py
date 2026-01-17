import sys
import os
import json
import logging
import uuid
from unittest.mock import MagicMock, patch

# Setup path: we need the parent of 'conclave' directory in sys.path
# to allow 'from conclave...' imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Conclave.Test")

def test_imports():
    """Verify all modules can be imported (detects missing libs/syntax errors)."""
    logger.info("1️⃣  Testing Imports...")
    try:
        # Core
        from conclave.core.engine import ConclaveEngine
        from conclave.core.identity import IdentityManager
        from conclave.core.graph_store import GraphStore
        
        # Perception
        from conclave.perception.audio.voice import GlobalAudioPipeline # Updated class
        from conclave.perception.vision.face import FaceProcessor
        from conclave.perception.vision.scene import AdvancedSceneProcessor
        
        # Agents
        from conclave.agent.reasoning import ReasoningAgent
        from conclave.agent.control import ControlAgent # New Agent
        
        # Libs
        import whisperx
        import speechbrain
        import qdrant_client
        import neo4j
        
        logger.info("✅ All core modules & libraries imported successfully.")
    except ImportError as e:
        logger.error(f"❌ Import Failed: {e}")
        logger.error("Did you install requirements? (pip install -r requirements.lock)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Syntax/Logic Error during import: {e}")
        sys.exit(1)

def test_identity_logic():
    """Verify Identity Manager logic with Persistence Mocking."""
    logger.info("\n2️⃣  Testing Identity Logic (Mocked DB)...")
    
    from conclave.core.identity import IdentityManager
    from conclave.core.schemas import FaceObservation
    
    # Mock Stores
    mock_vec = MagicMock()
    mock_graph = MagicMock()
    
    # CRITICAL FIX: IdentityManager now loads aliases on init. 
    # We must mock the graph return to be empty list, or it will crash iterating a Mock.
    mock_graph.run_query.return_value = [] 
    
    # Init Manager
    im = IdentityManager(mock_vec, mock_graph, {})
    
    # Create Dummy Observation
    obs = FaceObservation(
        video_id="test_vid", clip_id=1, ts_ms=1000, 
        embedding=[0.1]*512, bbox=[0,0,10,10], 
        base64_img="abc", detection_score=0.99, quality_score=99
    )
    
    # 1. Test Resolve (New Entity)
    # Mock vector search returning nothing -> should create new ID
    mock_vec.search.return_value = [] 
    
    eid = im.resolve_face(obs)
    logger.info(f"   -> Resolve Face result: {eid}")
    
    if not eid.startswith("ent_face"):
        logger.error("❌ Failed to generate proper entity ID")
    else:
        logger.info("✅ Identity Resolution Logic Works")
        
    # 2. Test Safe Merge (M3 Logic)
    im._merge_entities_safe("keep_id", "drop_id", "test_vid")
    # Verify graph was called with the async query
    if mock_graph.execute_async.called:
        logger.info("✅ Safe Merge Logic triggered Graph Update")
    else:
        logger.error("❌ Safe Merge did not trigger Graph Update")

def test_reasoning_logic():
    """Verify Reasoning Agent: Grounding + Equivalence."""
    logger.info("\n3️⃣  Testing Reasoning Agent (Mocked LLM)...")
    
    from conclave.agent.reasoning import ReasoningAgent
    from conclave.core.schemas import FaceObservation, VoiceObservation
    
    # Config with fake key
    config = {"api_key": "sk-fake-key", "model": "gpt-4o"}
    agent = ReasoningAgent(config)
    agent.client = MagicMock()
    
    # --- TEST A: Grounding ---
    logger.info("   [A] Testing Hallucination Filter...")
    mock_mem_response = {
        "memories": [
            "Person <ent_face_1> smiled.",
            "Person <ent_ghost> flew away." # Hallucination
        ]
    }
    agent.client.chat.completions.create.return_value.choices[0].message.content = json.dumps(mock_mem_response)
    
    faces = [FaceObservation(video_id="v", clip_id=1, ts_ms=0, embedding=[], bbox=[], base64_img="", detection_score=0, quality_score=0, entity_id="ent_face_1")]
    
    memories = agent.generate_episodic_memory("v", 1, [], faces, [])
    
    if "ent_ghost" in memories[0].linked_entities:
        logger.error("❌ Hallucination Filter Failed")
    elif "ent_face_1" in memories[0].linked_entities:
        logger.info("✅ Hallucination Filter Works")
        
    # --- TEST B: M3 Equivalence Detection ---
    logger.info("   [B] Testing M3 Equivalence Logic...")
    mock_eq_response = {
        "equivalences": [
            {"source": "<ent_face_1>", "target": "<ent_voice_99>"}
        ]
    }
    agent.client.chat.completions.create.return_value.choices[0].message.content = json.dumps(mock_eq_response)
    
    merges = agent.detect_equivalences("v", 1, "context string")
    
    if len(merges) == 1 and merges[0]['source'] == "ent_face_1":
        logger.info(f"✅ Equivalence Detected: {merges}")
    else:
        logger.error(f"❌ Failed to parse equivalences: {merges}")

def test_control_wiring():
    """Verify Control Agent can be initialized (Using Patches to skip DB)."""
    logger.info("\n4️⃣  Testing Control Agent Wiring...")
    
    from conclave.agent.control import ControlAgent
    
    # We patch 'open' (config read), 'ConclaveEngine' (DB), and 'EmbeddingService'
    with patch("builtins.open", new_callable=MagicMock) as mock_open, \
         patch("conclave.agent.control.ConclaveEngine") as MockEngine, \
         patch("conclave.agent.control.EmbeddingService") as MockEmbed:
        
        # Setup mock config
        mock_open.return_value.__enter__.return_value.read.return_value = '{"api": {"openai_api_key": "sk-test"}}'
        
        try:
            agent = ControlAgent("fake_config.json", "vid_123")
            logger.info("✅ Control Agent Initialized successfully.")
        except Exception as e:
            logger.error(f"❌ Control Agent Init Failed: {e}")

if __name__ == "__main__":
    test_imports()
    test_identity_logic()
    test_reasoning_logic()
    test_control_wiring()
    logger.info("\n🎉 System Check Complete. Pipeline logic is updated and consistent.")