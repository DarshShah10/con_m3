from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import uuid

@dataclass(kw_only=True)
class LinkedText:
    """Text found specifically INSIDE an object."""
    content: str
    confidence: float
    bbox_relative: List[int]

@dataclass(kw_only=True)
class DetectedObject:
    """A physical entity found in the frame."""
    label: str
    confidence: float
    bbox: List[int]
    linked_text: List[LinkedText] = field(default_factory=list)

@dataclass(kw_only=True)
class HierarchicalFrameObservation:
    """Master container for a single moment."""
    video_id: str
    clip_id: int
    ts_ms: int
    scene_description: Optional[str] = None
    objects: List[DetectedObject] = field(default_factory=list)
    clip_embedding: Optional[List[float]] = None
    obs_id: str = field(default_factory=lambda: str(uuid.uuid4()))

# --- Legacy Schemas (Required for Identity/Memory) ---
@dataclass(kw_only=True)
class FaceObservation:
    video_id: str
    clip_id: int
    ts_ms: int
    embedding: List[float]
    bbox: List[int]
    base64_img: str
    detection_score: float
    quality_score: float
    entity_id: Optional[str] = None
    obs_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass(kw_only=True)
class VoiceObservation:
    video_id: str
    clip_id: int
    ts_ms: int
    embedding: List[float]
    asr_text: str
    start_sec: float
    end_sec: float
    entity_id: Optional[str] = None
    obs_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
@dataclass(kw_only=True)
class MemoryNode:
    video_id: str
    content: str
    mem_type: str
    clip_id: int
    mem_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = None
    linked_entities: List[str] = field(default_factory=list)

@dataclass(kw_only=True)
class DialogueLine:
    """Structured dialogue linked to an entity."""
    video_id: str
    clip_id: int
    entity_id: str
    text: str
    start_ts: int
    end_ts: int
    confidence: float

@dataclass(kw_only=True)
class GlobalEntity:
    entity_id: str
    entity_type: "EntityType"
    video_id: str
    
    # Track stats to determine if this is the Body Cam user
    face_appearances: int = 0
    voice_appearances: int = 0
    
    face_embeddings: List[List[float]] = field(default_factory=list)
    voice_embeddings: List[List[float]] = field(default_factory=list)

from enum import Enum
class EntityType(Enum):
    PERSON = "person"
    OBJECT = "object"
    VEHICLE = "vehicle"
    TEXT = "text"
    LOCATION = "location"
    POV = "pov_user"

class MemoryType(Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"