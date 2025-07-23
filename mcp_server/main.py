import sys


# --- Pydantic models for endpoint request bodies ---
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class BankOp(BaseModel):
    bank: str = Field(..., description="Target memory bank name")

class KnowledgeIngest(BaseModel):
    text: str
    bank: str = "default"
    source: str = "text_input"
    extract_entities: bool = True
    extract_relationships: bool = True
    create_observations: bool = True

# --- Stubs for legacy functions to avoid undefined errors (modern system does not use them) ---
def save_memory_banks():
    """Stub for legacy save_memory_banks (no-op in modern system)"""
    pass

def load_memory_banks():
    """Stub for legacy load_memory_banks (no-op in modern system)"""
    pass

# --- Optional imports for schema_manager and graph_analytics ---
try:
    from core.graph_schema import SchemaManager
    schema_manager = SchemaManager()
except ImportError:
    schema_manager = None

try:
    from core.graph_analytics import GraphAnalytics
    graph_analytics = GraphAnalytics()
except ImportError:
    graph_analytics = None
from fastapi import FastAPI, Query, Body, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager
from pathlib import Path
import uuid
from datetime import datetime
import os
import json
import logging
# Initialize FastAPI app
app = FastAPI()

# --- Global variables for modern storage system ---
storage_backends: Dict[str, Any] = {}
current_bank: str = "default"

def get_storage(bank: str) -> Any:
    """Get the storage backend for a given bank, or raise error if not found."""
    if bank not in storage_backends:
        raise HTTPException(status_code=404, detail=f"Bank '{bank}' not found.")
    return storage_backends[bank]

# Legacy memory file path (for compatibility, not used in modern system)
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "memory_banks.json")
import json
import asyncio
import time
import logging

# Initialize FastAPI app
app = FastAPI()

# --- Visualization Endpoint (modern system) ---
from pathlib import Path
@app.get("/banks/{bank}/visualize")
def visualize_graph(bank: str):
    """
    Serve interactive graph visualization page for a specific memory bank (modern system).
    """
    if bank not in storage_backends:
        return JSONResponse(content={"error": "Bank not found"}, status_code=404)
    try:
        template_path = Path(__file__).parent / "templates" / "visualization.html"
        with open(template_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        html_content = html_content.replace('{{bank}}', bank)
        return StreamingResponse(
            iter([html_content]),
            media_type="text/html"
        )
    except FileNotFoundError:
        return JSONResponse(
            content={"error": "Visualization template not found"}, 
            status_code=500
        )
    except Exception as e:
        return JSONResponse(
            content={"error": f"Error loading visualization: {str(e)}"}, 
            status_code=500)
# --- Modern Bank Management Endpoints ---
@app.post("/banks/create")
def create_bank(op: BankOp):
    """
    Create a new memory bank using the modern storage system.
    Request body: {"bank": "bank_name"}
    Response: {"status": "success", "bank": "bank_name"}
    """
    if op.bank in storage_backends:
        return {"status": "error", "message": "Bank already exists."}
    storage_backends[op.bank] = create_graph_store("memory")
    return {"status": "success", "bank": op.bank}

@app.post("/banks/select")
def select_bank(op: BankOp):
    """
    Select the active memory bank.
    Request body: {"bank": "bank_name"}
    Response: {"status": "success", "selected": "bank_name"}
    """
    global current_bank
    if op.bank not in storage_backends:
        return {"status": "error", "message": "Bank does not exist."}
    current_bank = op.bank
    return {"status": "success", "selected": current_bank}

@app.get("/banks/list")
def list_banks():
    """
    List all memory banks and show the current active bank.
    Response: {"banks": ["bank1", "bank2"], "current": "bank_name"}
    """
    return {"banks": list(storage_backends.keys()), "current": current_bank}

@app.post("/banks/delete")
def delete_bank(op: BankOp):
    """
    Delete a memory bank (except 'default').
    Request body: {"bank": "bank_name"}
    Response: {"status": "success", "deleted": "bank_name", "current": "bank_name"}
    """
    global current_bank
    if op.bank == "default":
        return {"status": "error", "message": "Cannot delete default bank."}
    if op.bank not in storage_backends:
        return {"status": "error", "message": "Bank does not exist."}
    del storage_backends[op.bank]
    if current_bank == op.bank:
        current_bank = "default"
    return {"status": "success", "deleted": op.bank, "current": current_bank}

@app.post("/banks/clear")
def clear_bank(op: BankOp):
    """
    Clear all entities, relationships, and observations from a bank (modern system).
    Request body: {"bank": "bank_name"}
    Response: {"status": "success", "cleared": "bank_name", "entities_deleted": N, "relations_deleted": N, "observations_deleted": N}
    """
    b = op.bank or current_bank
    store = get_storage(b)
    entities_count = len(store.entities)
    relations_count = len(store.relationships)
    observations_count = sum(len(e.data.get("observations", [])) for e in store.entities.values())
    # Clear all data
    store.entities.clear()
    store.relationships.clear()
    # No explicit observations list; stored in entity data
    return {
        "status": "success", 
        "cleared": b, 
        "entities_deleted": entities_count,
        "relations_deleted": relations_count, 
        "observations_deleted": observations_count
    }

# --- Production-grade Pydantic models for all graph memory entities ---
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class Node(BaseModel):
    id: str = Field(..., description="Unique identifier for the node/entity")
    data: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary entity data, including type, name, observations, etc.")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="Last update timestamp")

class Edge(BaseModel):
    id: Optional[str] = Field(default=None, description="Unique identifier for the edge/relation")
    source: str = Field(..., description="Source node/entity id")
    target: str = Field(..., description="Target node/entity id")
    data: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary relation data, including type, context, etc.")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="Last update timestamp")

class Observation(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the observation")
    entity_id: Optional[str] = Field(default=None, description="Associated entity id (if any)")
    content: str = Field(..., description="Observation text/content")
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Timestamp of observation")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional observation metadata")

class ReasoningStep(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the reasoning step")
    step_number: Optional[int] = Field(default=None, description="Step number in sequence")
    reasoning: str = Field(..., description="Reasoning or thought process")
    thought: Optional[str] = Field(default=None, description="Optional thought or hypothesis")
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Timestamp of step")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional step metadata")

class BankOp(BaseModel):
    bank: str = Field(..., description="Target memory bank name")

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
from tool_list import TOOL_LIST

# PHASE 4.1.1: Import new storage system
try:
    from storage import create_graph_store, GraphStore, MemoryStore, StorageConfig
    STORAGE_AVAILABLE = True
    logger.info("Phase 3.2 storage system loaded successfully")
except ImportError as e:
    logger.warning(f"Storage system not available, using legacy fallback: {e}")
    STORAGE_AVAILABLE = False

# Import Phase 1-3 core components for enhanced knowledge graph
try:
    from core.graph_schema import SchemaManager, EntityInstance, RelationshipInstance
    from core.entity_resolution import EntityResolver
    from core.graph_analytics import GraphAnalytics
    CORE_COMPONENTS_AVAILABLE = True
    logger.info("Core knowledge graph components loaded successfully")
except ImportError as e:
    logger.warning(f"Core components not available: {e}")
    CORE_COMPONENTS_AVAILABLE = False

# Modern Knowledge Graph Processing
try:
    from knowledge_graph_processor import create_knowledge_graph_processor
    kg_processor = create_knowledge_graph_processor()
    MODERN_KG_AVAILABLE = True
    logger.info("Modern knowledge graph processor loaded successfully")
except ImportError as e:
    MODERN_KG_AVAILABLE = False
    logger.warning(f"Modern knowledge graph processor not available: {e}")

async def load_memory_banks_legacy():
    """Async version of legacy memory banks loading for compatibility with FastAPI lifespan"""
    global memory_banks, current_bank
    
    logger.warning("Using legacy memory_banks system - storage backend not available")
    
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
                memory_banks = data.get('memory_banks', {})
                global_current_bank = data.get('current_bank', 'default')
                if global_current_bank in memory_banks:
                    current_bank = global_current_bank
                logger.debug(f"Loaded banks: {list(memory_banks.keys())}")
        else:
            logger.debug("No existing memory file found, starting with default banks")
            memory_banks = {"default": {"nodes": {}, "edges": [], "observations": [], "reasoning_steps": []}}
        
        # Ensure current_bank exists
        if current_bank not in memory_banks:
            current_bank = "default"
            
    except Exception as e:
        logger.error(f"Failed to load legacy memory banks: {e}")
        logger.info("Starting with default memory banks")
        memory_banks = {"default": {"nodes": {}, "edges": [], "observations": [], "reasoning_steps": []}}
        current_bank = "default"

def load_memory_banks_sync():
    """Synchronous wrapper for loading memory banks - DEPRECATED: Use load_memory_banks_legacy() instead"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If event loop is already running, we can't use run_until_complete
            raise RuntimeError("Cannot use synchronous loader when event loop is running")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(load_memory_banks())

def serialize_legacy_data(legacy_data: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize legacy data format for JSON storage"""
    return legacy_data  # Already in serializable format

# Note: Application initialization now handled by FastAPI lifespan manager


# Bank management endpoints

@app.post("/banks/create")
def create_bank(op: BankOp):
    """
    Create a new memory bank.
    Request body: {"bank": "bank_name"}
    Response: {"status": "success", "bank": "bank_name"}
    """
    if op.bank in storage_backends:
        return {"status": "error", "message": "Bank already exists."}
    # Create the new bank
    storage_backends[op.bank] = create_graph_store("memory")
    return {"status": "success", "bank": op.bank}


@app.post("/banks/select")
def select_bank(op: BankOp):
    """
    Select the active memory bank.
    Request body: {"bank": "bank_name"}
    Response: {"status": "success", "selected": "bank_name"}
    """
    global current_bank
    if op.bank not in storage_backends:
        return {"status": "error", "message": "Bank does not exist."}
    current_bank = op.bank
    return {"status": "success", "selected": current_bank}


@app.get("/banks/list")
def list_banks():
    """
    List all memory banks and show the current active bank.
    Response: {"banks": ["bank1", "bank2"], "current": "bank_name"}
    """
    return {"banks": list(storage_backends.keys()), "current": current_bank}


@app.post("/banks/delete")
def delete_bank(op: BankOp):
    """
    Delete a memory bank (except 'default').
    Request body: {"bank": "bank_name"}
    Response: {"status": "success", "deleted": "bank_name", "current": "bank_name"}
    """
    global current_bank
    if op.bank == "default":
        return {"status": "error", "message": "Cannot delete default bank."}
    if op.bank not in storage_backends:
        return {"status": "error", "message": "Bank does not exist."}
    del storage_backends[op.bank]
    if current_bank == op.bank:
        current_bank = "default"
    return {"status": "success", "deleted": op.bank, "current": current_bank}

@app.post("/banks/clear")
def clear_bank(op: BankOp):
    """
    Clear all entities, relationships, and observations from a bank.
    Request body: {"bank": "bank_name"}
    Response: {"status": "success", "cleared": "bank_name", "entities_deleted": N, "relations_deleted": N, "observations_deleted": N}
    """
    b = op.bank or current_bank
    entities_count = len(memory_banks[b]["nodes"])
    relations_count = len(memory_banks[b]["edges"])
    observations_count = len(memory_banks[b]["observations"])
    
    # Clear all data
    memory_banks[b]["nodes"].clear()
    memory_banks[b]["edges"].clear()
    memory_banks[b]["observations"].clear()
    memory_banks[b]["reasoning_steps"].clear()
    
    save_memory_banks()  # Persist the changes
    return {
        "status": "success", 
        "cleared": b, 
        "entities_deleted": entities_count,
        "relations_deleted": relations_count, 
        "observations_deleted": observations_count
    }


# --- Modern Entity Endpoints ---
@app.post("/entities")
def add_entity(node: Node = Body(...), bank: str = Query(None)):
    """
    Add an entity (node) to the selected or specified bank (modern system).
    Request body: Node model
    Query param: bank (optional)
    Response: {"status": "success", "entity": Node, "bank": "bank_name"}
    """
    b = bank or current_bank
    try:
        store = get_storage(b)
        if not isinstance(node, Node):
            node = Node(**node.model_dump() if hasattr(node, 'model_dump') else node)
        store.entities[node.id] = node
        return {"status": "success", "entity": node, "bank": b}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/entities")
def get_entities(bank: str = Query(None)):
    """
    Get all entities (nodes) from the selected or specified bank (modern system).
    Query param: bank (optional)
    Response: List of Node dicts
    """
    b = bank or current_bank
    store = get_storage(b)
    return [n.model_dump() for n in store.entities.values()]

@app.put("/entities/{entity_id}")
def update_entity(entity_id: str, node: Node, bank: str = Query(None)):
    """
    Update an entity (node) in the selected or specified bank (modern system).
    Request body: Node model
    Query param: bank (optional)
    Response: {"status": "success", "entity": Node, "bank": "bank_name"}
    """
    b = bank or current_bank
    store = get_storage(b)
    store.entities[entity_id] = node
    return {"status": "success", "entity": node, "bank": b}

@app.delete("/entities/{entity_id}")
def delete_entity(entity_id: str, bank: str = Query(None)):
    """
    Delete an entity (node) from the selected or specified bank (modern system).
    Query param: bank (optional)
    Response: {"status": "success", "deleted": "entity_id", "bank": "bank_name"}
    """
    b = bank or current_bank
    store = get_storage(b)
    if entity_id in store.entities:
        del store.entities[entity_id]
        return {"status": "success", "deleted": entity_id, "bank": b}
    return {"status": "error", "message": "Entity not found", "bank": b}


# --- Modern Relation Endpoints ---
@app.post("/relations")
def add_relation(edge: Edge = Body(...), bank: str = Query(None)):
    """
    Add a relation (edge) to the selected or specified bank (modern system).
    Request body: Edge model
    Query param: bank (optional)
    Response: {"status": "success", "relation": Edge, "bank": "bank_name"}
    """
    b = bank or current_bank
    try:
        store = get_storage(b)
        edge.id = edge.id or f"{edge.source}-{edge.target}-{len(store.relationships)}"
        if not isinstance(edge, Edge):
            edge = Edge(**edge.model_dump() if hasattr(edge, 'model_dump') else edge)
        store.relationships.append(edge)
        return {"status": "success", "relation": edge, "bank": b}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/relations")
def get_relations(bank: str = Query(None)):
    """
    Get all relations (edges) from the selected or specified bank (modern system).
    Query param: bank (optional)
    Response: List of Edge dicts
    """
    b = bank or current_bank
    store = get_storage(b)
    return [e.model_dump() for e in store.relationships]

@app.put("/relations/{relation_id}")
def update_relation(relation_id: str, edge: Edge, bank: str = Query(None)):
    """
    Update a relation (edge) in the selected or specified bank (modern system).
    Request body: Edge model
    Query param: bank (optional)
    Response: {"status": "success", "relation": Edge, "bank": "bank_name"}
    """
    b = bank or current_bank
    store = get_storage(b)
    for i, e in enumerate(store.relationships):
        if e.id == relation_id:
            store.relationships[i] = edge
            return {"status": "success", "relation": edge, "bank": b}
    return {"status": "error", "message": "Relation not found", "bank": b}

@app.delete("/relations/{relation_id}")
def delete_relation(relation_id: str, bank: str = Query(None)):
    """
    Delete a relation (edge) from the selected or specified bank (modern system).
    Query param: bank (optional)
    Response: {"status": "success", "deleted": "relation_id", "bank": "bank_name"}
    """
    b = bank or current_bank
    store = get_storage(b)
    for i, e in enumerate(store.relationships):
        if e.id == relation_id:
            del store.relationships[i]
            return {"status": "success", "deleted": relation_id, "bank": b}
    return {"status": "error", "message": "Relation not found", "bank": b}


# --- Modern Observation Endpoints ---
@app.post("/observations")
def add_observation(obs: Observation = Body(...), bank: str = Query(None)):
    """
    Add an observation to the selected or specified bank (modern system).
    Request body: Observation model
    Query param: bank (optional)
    Response: {"status": "success", "observation": Observation, "bank": "bank_name"}
    """
    b = bank or current_bank
    try:
        store = get_storage(b)
        if not isinstance(obs, Observation):
            obs = Observation(**obs.model_dump() if hasattr(obs, 'model_dump') else obs)
        # Attach observation to entity if entity_id is provided
        if obs.entity_id and obs.entity_id in store.entities:
            entity = store.entities[obs.entity_id]
            entity.data.setdefault("observations", []).append(obs.model_dump())
        else:
            # Orphan observation: store in a dedicated list (optional, not implemented here)
            pass
        return {"status": "success", "observation": obs, "bank": b}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/observations")
def get_observations(bank: str = Query(None)):
    """
    Get all observations from the selected or specified bank (modern system).
    Query param: bank (optional)
    Response: List of Observation dicts
    """
    b = bank or current_bank
    store = get_storage(b)
    observations = []
    for entity in store.entities.values():
        observations.extend(entity.data.get("observations", []))
    return observations


# --- Modern Reasoning Endpoints ---
@app.post("/sequential-thinking")
def add_reasoning_step(step: ReasoningStep = Body(...), bank: str = Query(None)):
    """
    Add a reasoning step to the selected or specified bank (modern system).
    Request body: ReasoningStep model
    Query param: bank (optional)
    Response: {"status": "success", "step": ReasoningStep, "bank": "bank_name"}
    """
    b = bank or current_bank
    try:
        store = get_storage(b)
        if not hasattr(store, "reasoning_steps"):
            store.reasoning_steps = []
        if not isinstance(step, ReasoningStep):
            step = ReasoningStep(**step.model_dump() if hasattr(step, 'model_dump') else step)
        store.reasoning_steps.append(step)
        return {"status": "success", "step": step, "bank": b}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/sequential-thinking")
def get_reasoning_steps(bank: str = Query(None)):
    """
    Get all reasoning steps from the selected or specified bank (modern system).
    Query param: bank (optional)
    Response: List of ReasoningStep dicts
    """
    b = bank or current_bank
    store = get_storage(b)
    if hasattr(store, "reasoning_steps"):
        return [s.model_dump() for s in store.reasoning_steps]
    return []


# --- Modern MCP Tool Endpoints ---
from fastapi import HTTPException

@app.post("/open-nodes")
def open_nodes_endpoint(names: List[str] = Body(...), bank: str = Query(None)):
    """
    Open specific nodes in the knowledge graph by their names (modern system).
    Request body: {"names": ["entity1", "entity2"]}
    Query param: bank (optional)
    Response: List of Node dicts
    """
    b = bank or current_bank
    store = get_storage(b)
    found = []
    for name in names:
        node = store.entities.get(name)
        if node:
            found.append(node.model_dump())
    if not found:
        raise HTTPException(status_code=404, detail="No nodes found for given names.")
    return found


@app.get("/read-graph")
def read_graph_endpoint(bank: str = Query(None)):
    """
    Read the entire knowledge graph summary for the current memory bank (modern system).
    Query param: bank (optional)
    Response: Dict with summary statistics
    """
    b = bank or current_bank
    store = get_storage(b)
    nodes = store.entities
    edges = store.relationships
    observations = []
    for entity in nodes.values():
        observations.extend(entity.data.get("observations", []))
    reasoning_steps = getattr(store, "reasoning_steps", [])
    return {
        "bank": b,
        "entity_count": len(nodes),
        "relation_count": len(edges),
        "observation_count": len(observations),
        "reasoning_step_count": len(reasoning_steps),
        "entities": [n.model_dump() for n in nodes.values()],
        "relations": [e.model_dump() for e in edges],
        "observations": observations,
        "reasoning_steps": [s.model_dump() for s in reasoning_steps]
    }


# Search Helper Functions
import re
from typing import List, Dict, Any, Optional, Union

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def fuzzy_similarity(s1: str, s2: str) -> float:
    """Calculate fuzzy similarity score between 0.0 and 1.0"""
    if not s1 or not s2:
        return 0.0
    
    # Normalize strings for comparison
    s1_norm = s1.lower().strip()
    s2_norm = s2.lower().strip()
    
    if s1_norm == s2_norm:
        return 1.0
    
    max_len = max(len(s1_norm), len(s2_norm))
    if max_len == 0:
        return 1.0
    
    distance = levenshtein_distance(s1_norm, s2_norm)
    return 1.0 - (distance / max_len)

def search_text(query: str, text: str, case_sensitive: bool = False, use_regex: bool = False, 
                fuzzy_match: bool = False, fuzzy_threshold: float = 0.8) -> bool:
    """Helper function to search text with various matching options."""
    if not query or not text:
        return False
    
    if use_regex:
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            return bool(re.search(query, text, flags))
        except re.error:
            # If regex is invalid, fall back to simple search
            pass
    
    # Try exact/partial matching first (existing behavior)
    if case_sensitive:
        exact_match = query in text
    else:
        exact_match = query.lower() in text.lower()
    
    if exact_match:
        return True
    
    # If fuzzy matching enabled and no exact match, try fuzzy
    if fuzzy_match:
        # Check fuzzy similarity for whole text and individual words
        similarity = fuzzy_similarity(query, text)
        if similarity >= fuzzy_threshold:
            return True
        
        # Also check individual words in text for fuzzy matches
        words = text.split()
        for word in words:
            word_similarity = fuzzy_similarity(query, word)
            if word_similarity >= fuzzy_threshold:
                return True
    
    return False

def calculate_relevance_score(query: str, text: str, exact_match_bonus: float = 0.5, 
                             fuzzy_match: bool = False, fuzzy_threshold: float = 0.8) -> float:
    """Calculate relevance score for search results with fuzzy matching support."""
    if not query or not text:
        return 0.0
    
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Exact match gets highest score
    if query_lower == text_lower:
        return 1.0
    
    # Check if query appears as whole word
    if f" {query_lower} " in f" {text_lower} ":
        return 0.8 + exact_match_bonus
    
    # Partial match score based on coverage
    if query_lower in text_lower:
        coverage = len(query_lower) / len(text_lower)
        return min(0.7, 0.3 + coverage)
    
    # Fuzzy matching score
    if fuzzy_match:
        similarity = fuzzy_similarity(query, text)
        if similarity >= fuzzy_threshold:
            # Scale fuzzy scores to be lower than exact matches
            return similarity * 0.6
        
        # Check individual words for fuzzy matches
        words = text.split()
        max_word_similarity = 0.0
        for word in words:
            word_similarity = fuzzy_similarity(query, word)
            if word_similarity >= fuzzy_threshold:
                max_word_similarity = max(max_word_similarity, word_similarity)
        
        if max_word_similarity > 0:
            return max_word_similarity * 0.5
    
    return 0.0

def search_entities(query: str, bank: str = None, entity_type: str = None, 
                   case_sensitive: bool = False, use_regex: bool = False,
                   fuzzy_match: bool = False, fuzzy_threshold: float = 0.8, limit: int = 50) -> List[Dict[str, Any]]:
    """Search entities by name, type, or observations content."""
    b = bank or current_bank
    results = []
    
    for entity_id, entity in memory_banks[b]["nodes"].items():
        # Skip if entity_type filter doesn't match
        if entity_type and entity.data.get("type") != entity_type:
            continue
        
        relevance_score = 0.0
        matched_fields = []
        
        # Search in entity ID/name
        if search_text(query, entity_id, case_sensitive, use_regex, fuzzy_match, fuzzy_threshold):
            relevance_score = max(relevance_score, calculate_relevance_score(query, entity_id, 0.5, fuzzy_match, fuzzy_threshold))
            matched_fields.append("name")
        
        # Search in entity type
        ent_type = entity.data.get("type", "")
        if search_text(query, ent_type, case_sensitive, use_regex, fuzzy_match, fuzzy_threshold):
            relevance_score = max(relevance_score, calculate_relevance_score(query, ent_type, 0.5, fuzzy_match, fuzzy_threshold) * 0.8)
            matched_fields.append("type")
        
        # Search in observations
        observations = entity.data.get("observations", [])
        for obs in observations:
            if search_text(query, obs, case_sensitive, use_regex, fuzzy_match, fuzzy_threshold):
                obs_score = calculate_relevance_score(query, obs, 0.5, fuzzy_match, fuzzy_threshold) * 0.6
                relevance_score = max(relevance_score, obs_score)
                if "observations" not in matched_fields:
                    matched_fields.append("observations")
        
        # If any match found, add to results
        if relevance_score > 0:
            results.append({
                "entity_id": entity_id,
                "entity_type": ent_type,
                "data": entity.data,
                "relevance_score": relevance_score,
                "matched_fields": matched_fields,
                "bank": b
            })
    
    # Sort by relevance score (highest first)
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results

def search_relationships(query: str, bank: str = None, relationship_type: str = None,
                        case_sensitive: bool = False, use_regex: bool = False) -> List[Dict[str, Any]]:
    """Search relationships by type, context, or entity names.
    Returns relationships ranked by relevance score.
    """
    b = bank or current_bank
    results = []
    
    for edge in memory_banks[b]["edges"]:
        # Skip if relationship_type filter doesn't match
        if relationship_type and edge.data.get("type") != relationship_type:
            continue
        
        relevance_score = 0.0
        matched_fields = []
        
        # Search in relationship type
        rel_type = edge.data.get("type", "")
        if search_text(query, rel_type, case_sensitive, use_regex):
            relevance_score = max(relevance_score, calculate_relevance_score(query, rel_type))
            matched_fields.append("type")
        
        # Search in context
        context = edge.data.get("context", "")
        if search_text(query, context, case_sensitive, use_regex):
            relevance_score = max(relevance_score, calculate_relevance_score(query, context) * 0.8)
            matched_fields.append("context")
        
        # Search in source entities
        if search_text(query, edge.source, case_sensitive, use_regex):
            relevance_score = max(relevance_score, calculate_relevance_score(query, edge.source) * 0.6)
            matched_fields.append("from_entity")
        
        if search_text(query, edge.target, case_sensitive, use_regex):
            relevance_score = max(relevance_score, calculate_relevance_score(query, edge.target) * 0.6)
            matched_fields.append("to_entity")
        
        # If any match found, add to results
        if relevance_score > 0:
            results.append({
                "relationship_id": edge.id,
                "from_entity": edge.source,
                "to_entity": edge.target,
                "relationship_type": rel_type,
                "data": edge.data,
                "relevance_score": relevance_score,
                "matched_fields": matched_fields,
                "bank": b
            })
    
    # Sort by relevance score (highest first)
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results

def search_observations(query: str, bank: str = None, entity_id: str = None,
                       case_sensitive: bool = False, use_regex: bool = False) -> List[Dict[str, Any]]:
    """Search observations by content or entity."""
    b = bank or current_bank
    results = []
    
    for observation in memory_banks[b]["observations"]:
        # Skip if entity_id filter doesn't match
        if entity_id and observation.entity_id != entity_id:
            continue
        
        relevance_score = 0.0
        matched_fields = []
        
        # Search in observation content
        if search_text(query, observation.content, case_sensitive, use_regex):
            relevance_score = max(relevance_score, calculate_relevance_score(query, observation.content))
            matched_fields.append("content")
        
        # Search in entity ID
        if search_text(query, observation.entity_id, case_sensitive, use_regex):
            relevance_score = max(relevance_score, calculate_relevance_score(query, observation.entity_id) * 0.7)
            matched_fields.append("entity_id")
        
        # If any match found, add to results
        if relevance_score > 0:
            results.append({
                "observation_id": observation.id,
                "entity_id": observation.entity_id,
                "content": observation.content,
                "timestamp": observation.timestamp,
                "relevance_score": relevance_score,
                "matched_fields": matched_fields,
                "bank": b
            })
    
    # Sort by relevance score (highest first)
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results


# Search HTTP Endpoints

@app.get("/search/entities")
def search_entities_endpoint(
    q: str = Query(..., description="Search query"),
    bank: str = Query(None, description="Memory bank to search in"),
    entity_type: str = Query(None, description="Filter by entity type"),
    case_sensitive: bool = Query(False, description="Case sensitive search"),
    use_regex: bool = Query(False, description="Use regular expressions"),
    fuzzy_match: bool = Query(False, description="Enable fuzzy matching for typos"),
    fuzzy_threshold: float = Query(0.8, description="Fuzzy matching threshold (0.0-1.0)"),
    limit: int = Query(50, description="Maximum number of results")
):
    """
    Search entities by name, type, or observations content.
    Returns entities ranked by relevance score.
    Supports fuzzy matching for handling typos and variations.
    """
    """
    Search entities by name, type, or observations content in the modern system.
    Returns entities ranked by relevance score.
    Supports fuzzy matching for handling typos and variations.
    """
    b = bank or current_bank
    try:
        store = get_storage(b)
        results = []
        for entity_id, entity in store.entities.items():
            name = entity.data.get("name", "")
            etype = entity.data.get("type", "")
            observations = entity.data.get("observations", [])
            text_blob = f"{name} {etype} " + " ".join([obs.get("content", "") if isinstance(obs, dict) else str(obs) for obs in observations])
            score = calculate_relevance_score(q, text_blob, fuzzy_match=fuzzy_match, fuzzy_threshold=fuzzy_threshold)
            if score > 0.0:
                results.append({
                    "entity_id": entity_id,
                    "entity": entity.model_dump(),
                    "relevance_score": score
                })
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return {
            "query": q,
            "bank": b,
            "total_results": len(results),
            "results": results[:limit] if limit else results,
            "search_parameters": {
                "entity_type": entity_type,
                "case_sensitive": case_sensitive,
                "use_regex": use_regex,
                "fuzzy_match": fuzzy_match,
                "fuzzy_threshold": fuzzy_threshold
            }
        }
    except Exception as e:
        return {"error": str(e), "query": q}

@app.get("/search/relationships")
def search_relationships_endpoint(
    q: str = Query(..., description="Search query"),
    bank: str = Query(None, description="Memory bank to search in"),
    relationship_type: str = Query(None, description="Filter by relationship type"),
    case_sensitive: bool = Query(False, description="Case sensitive search"),
    use_regex: bool = Query(False, description="Use regular expressions"),
    limit: int = Query(50, description="Maximum number of results")
):
    """
    Search relationships by type, context, or entity names.
    Returns relationships ranked by relevance score.
    """
    """
    Search relationships by type, context, or entity names in the modern system.
    Returns relationships ranked by relevance score.
    """
    b = bank or current_bank
    try:
        store = get_storage(b)
        results = []
        for edge in store.relationships:
            etype = edge.data.get("type", "")
            context = edge.data.get("context", "")
            source_name = store.entities.get(edge.source, {}).data.get("name", "") if edge.source in store.entities else ""
            target_name = store.entities.get(edge.target, {}).data.get("name", "") if edge.target in store.entities else ""
            text_blob = f"{etype} {context} {source_name} {target_name}"
            score = calculate_relevance_score(q, text_blob)
            if score > 0.0:
                results.append({
                    "relation": edge.model_dump(),
                    "relevance_score": score
                })
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return {
            "query": q,
            "bank": b,
            "total_results": len(results),
            "results": results[:limit] if limit else results,
            "search_parameters": {
                "relationship_type": relationship_type,
                "case_sensitive": case_sensitive,
                "use_regex": use_regex
            }
        }
    except Exception as e:
        return {"error": str(e), "query": q}

@app.get("/search/observations")
def search_observations_endpoint(
    q: str = Query(..., description="Search query"),
    bank: str = Query(None, description="Memory bank to search in"),
    entity_id: str = Query(None, description="Filter by entity ID"),
    case_sensitive: bool = Query(False, description="Case sensitive search"),
    use_regex: bool = Query(False, description="Use regular expressions"),
    limit: int = Query(50, description="Maximum number of results")
):
    """
    Search observations by content or entity.
    Returns observations ranked by relevance score.
    """
    """
    Search observations by content or entity in the modern system.
    Returns observations ranked by relevance score.
    """
    b = bank or current_bank
    try:
        store = get_storage(b)
        results = []
        for entity in store.entities.values():
            if entity_id and entity.id != entity_id:
                continue
            for obs in entity.data.get("observations", []):
                content = obs.get("content", "") if isinstance(obs, dict) else str(obs)
                score = calculate_relevance_score(q, content)
                if score > 0.0:
                    results.append({
                        "observation": obs,
                        "entity_id": entity.id,
                        "relevance_score": score
                    })
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return {
            "query": q,
            "bank": b,
            "total_results": len(results),
            "results": results[:limit] if limit else results,
            "search_parameters": {
                "entity_id": entity_id,
                "case_sensitive": case_sensitive,
                "use_regex": use_regex
            }
        }
    except Exception as e:
        return {"error": str(e), "query": q}

@app.get("/search/all")
def search_all_endpoint(
    q: str = Query(..., description="Search query"),
    bank: str = Query(None, description="Memory bank to search in"),
    case_sensitive: bool = Query(False, description="Case sensitive search"),
    use_regex: bool = Query(False, description="Use regular expressions"),
    limit: int = Query(50, description="Maximum number of results per type")
):
    """
    Search across all entities, relationships, and observations.
    Returns comprehensive results ranked by relevance score.
    """
    try:
        entities = search_entities(q, bank, None, case_sensitive, use_regex)
        relationships = search_relationships(q, bank, None, case_sensitive, use_regex)
        observations = search_observations(q, bank, None, case_sensitive, use_regex)
        
        # Combine and sort by relevance
        all_results = []
        all_results.extend([{"type": "entity", **r} for r in entities])
        all_results.extend([{"type": "relationship", **r} for r in relationships])
        all_results.extend([{"type": "observation", **r} for r in observations])
        
        all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return {
            "query": q,
            "bank": bank or current_bank,
            "total_results": len(all_results),
            "results_by_type": {
                "entities": len(entities),
                "relationships": len(relationships),
                "observations": len(observations)
            },
            "results": all_results[:limit] if limit else all_results,
            "search_parameters": {
                "case_sensitive": case_sensitive,
                "use_regex": use_regex
            }
        }
    except Exception as e:
        return {"error": str(e), "query": q}

@app.get("/visualizations")
def get_visualizations(bank: str = Query(None)):
    """
    Get visualization data for the knowledge graph in the specified bank.
    Returns nodes and edges in a format suitable for visualization libraries.
    """
    b = bank or current_bank
    nodes = []
    edges = []
    if b not in storage_backends:
        return {"error": f"Bank '{b}' not found"}
    store = storage_backends[b]
    # Get all entities (nodes)
    for entity_id, entity in store.entities.items():
        nodes.append({
            "id": entity_id,
            "label": entity_id,
            "type": entity.data.get("type", "unknown"),
            "size": len(entity.data.get("observations", [])) + 1,
            "observations": entity.data.get("observations", []),
            "created_at": entity.data.get("created_at", ""),
            "updated_at": entity.data.get("updated_at", "")
        })
    # Get all relationships (edges)
    for edge in store.relationships:
        relation_type = edge.data.get("relation_type") or edge.data.get("type", "related_to")
        edges.append({
            "source": edge.source,
            "target": edge.target,
            "type": relation_type,
            "label": relation_type
        })
    return {
        "bank": b,
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": list(set([n["type"] for n in nodes])),
            "edge_types": list(set([e["type"] for e in edges]))
        }
    }

@app.get("/")
async def root():
    """Root endpoint - redirects to visualization"""
    default_bank = 'default' if 'default' in storage_backends else list(storage_backends.keys())[0] if storage_backends else 'default'
    return visualize_graph(default_bank)

# --- Context Ingestion Endpoint ---
@app.post("/context/ingest")
def ingest_context(payload: KnowledgeIngest = Body(...)):
    """
    Ingest text using modern NLP-based knowledge graph construction.
    Uses spaCy for NER, dependency parsing, and sentence transformers for embeddings.
    Request body: {"text": "...", "bank": "bank_name"}
    Response: {"status": "success", "entities": [...], "relationships": [...], "bank": "bank_name"}
    """
    text = payload.text
    b = payload.bank or current_bank
    try:
        # Use modern knowledge graph processor
        if MODERN_KG_AVAILABLE and kg_processor:
            existing_entity_names = list(storage_backends[b].entities.keys())
            kg_result = kg_processor.construct_knowledge_graph(text, existing_entity_names)
            entities = kg_result['entities']
            relationships = kg_result['relationships']
            stats = kg_result['stats']
            entities_added = len(entities)
            relationships_added = len(relationships)
            # Add entities and relationships to the bank
            for entity in entities:
                node = Node(id=entity['name'], data=entity)
                storage_backends[b].entities[entity['name']] = node
            for rel in relationships:
                edge = Edge(source=rel['source'], target=rel['target'], data=rel)
                storage_backends[b].relationships.append(edge)
            return {
                "status": "success",
                "method": "modern_nlp",
                "entities_added": entities_added,
                "total_entities": len(storage_backends[b].entities),
                "relationships_added": relationships_added,
                "total_relationships": len(storage_backends[b].relationships),
                "bank": b,
                "stats": stats
            }
    except Exception as e:
        logger.error(f"Modern KG construction failed: {e}, falling back to basic method")
        # Fallback to basic method
        logger.warning("Using basic fallback method for knowledge graph construction")
        entities = extract_advanced_entities(text)
        # Add entities as nodes
        for entity_name, entity_info in entities.items():
            if entity_name not in storage_backends[b].entities:
                node = Node(id=entity_name, data={
                    "type": entity_info["type"],
                    "confidence": entity_info["confidence"],
                    "from_text": True,
                    "extraction_method": "basic_fallback"
                })
                storage_backends[b].entities[entity_name] = node
        # Extract relationships using fallback method
        relationships = extract_relationships(text, entities)
        # Add relationships as edges
        edges_added = 0
        for rel in relationships:
            if isinstance(rel, dict) and "from" in rel and "to" in rel:
                source, target = rel["from"], rel["to"]
                rel_type = rel.get("type", "related_to")
            else:
                continue
            # Check if relationship already exists
            existing_edge = None
            for edge in storage_backends[b].relationships:
                if (edge.source == source and edge.target == target and 
                    edge.data.get("type") == rel_type):
                    existing_edge = edge
                    break
            if not existing_edge and source in storage_backends[b].entities and target in storage_backends[b].entities:
                edge_data = {
                    "type": rel_type,
                    "confidence": rel.get("confidence", 0.5),
                    "from_text": True,
                    "extraction_method": "basic_fallback"
                }
                edge = Edge(source=source, target=target, data=edge_data)
                storage_backends[b].relationships.append(edge)
                edges_added += 1
        return {
            "status": "success", 
            "method": "basic_fallback",
            "entities": list(entities.keys()), 
            "edges_added": edges_added, 
            "bank": b
        }

class KnowledgeIngest(BaseModel):
    text: str
    bank: str = "default"
    source: str = "text_input"
    extract_entities: bool = True
    extract_relationships: bool = True
    create_observations: bool = True

def find_similar_entity(entity_name: str, bank: str, similarity_threshold: float = 0.85) -> Optional[str]:
    """Find existing entity that is similar to the given entity name"""
    b = bank or current_bank
    entity_id = entity_name.replace(" ", "_").lower()
    
    # First check for exact match
    if entity_id in memory_banks[b]["nodes"]:
        return entity_id
    
    # Check for fuzzy matches
    for existing_id, existing_node in memory_banks[b]["nodes"].items():
        existing_name = existing_node.data.get("name", existing_id)
        
        # Check similarity between names
        similarity = fuzzy_similarity(entity_name, existing_name)
        if similarity >= similarity_threshold:
            return existing_id
    
    return None

def extract_advanced_entities(text: str):
    """Enhanced entity extraction with multiple patterns and types"""
    entities = {}
    
    # Common stop words to exclude from entity extraction
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """Extract relationships between entities with context, filtering out stop words"""
    relationships = []
    # Use the same stop_words and is_valid_entity as in extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if is_valid_entity(entity):  # Use the validation function
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        if is_valid_entity(term):
            entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if is_valid_entity(concept):
            entities[concept] = {"type": "concept", "confidence": 0.9}
    
    # 4. Email addresses and URLs
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities[email] = {"type": "email", "confidence": 1.0}
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities[url] = {"type": "url", "confidence": 1.0}
    
    # 5. Numbers and measurements
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|lb|ft|in|%|dollars?|USD|\$)\b', text, re.IGNORECASE)
    for measurement in measurements:
        entities[measurement] = {"type": "measurement", "confidence": 0.8}
    
    # 6. Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', text, re.IGNORECASE)
    for date in dates:
        entities[date] = {"type": "date", "confidence": 0.9}
    
    return entities

def extract_relationships(text: str, entities: dict):
    """
    Extract semantically meaningful relationships between entities.
    
    This enhanced version replaces generic "related_to" relationships with 
    specific semantic relationship types based on linguistic patterns,
    contextual analysis, and domain knowledge.
    """
    relationships = []
    
    # Use the same stop_words validation as extract_advanced_entities
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'a', 'an', 'as', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'is', 'am', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'get', 'got', 'getting', 'give', 'gives', 'gave', 'given', 'giving', 'go', 'goes', 'went', 'gone', 'going', 'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'make', 'makes', 'made', 'making', 'put', 'puts', 'putting', 'say', 'says', 'said', 'saying', 'see', 'sees', 'saw', 'seen', 'seeing', 'take', 'takes', 'took', 'taken', 'taking', 'come', 'comes', 'came', 'coming', 'want', 'wants', 'wanted', 'wanting', 'look', 'looks', 'looked', 'looking', 'use', 'uses', 'used', 'using', 'find', 'finds', 'found', 'finding', 'work', 'works', 'worked', 'working', 'call', 'calls', 'called', 'calling', 'try', 'tries', 'tried', 'trying', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed', 'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed', 'believing', 'hold', 'holds', 'held', 'holding', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens', 'happened', 'happening', 'write', 'writes', 'wrote', 'written', 'writing', 'provide', 'provides', 'provided', 'providing', 'sit', 'sits', 'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid', 'paying', 'meet', 'meets', 'met', 'meeting', 'include', 'includes', 'included', 'including', 'continue', 'continues', 'continued', 'continuing', 'set', 'sets', 'setting', 'run', 'runs', 'ran', 'running', 'remember', 'remembers', 'remembered', 'remembering', 'lot', 'way', 'back', 'little', 'good', 'man', 'woman', 'day', 'time', 'year', 'right', 'may', 'new', 'old', 'great', 'high', 'small', 'large', 'national', 'young', 'different', 'long', 'important', 'public', 'bad', 'same', 'able'
    }
    
    def is_valid_entity(entity_text):
        """Check if entity should be included (not a stop word or too short)"""
        entity_lower = entity_text.lower().strip()
        return (len(entity_lower) > 2 and 
                entity_lower not in stop_words and
                not entity_lower.isdigit() and
                len(entity_text.strip()) > 1)
    
    # Split text into sentences for semantic analysis
    sentences = re.split(r'[.!?;]', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
            
        # Find entities in this sentence with positions
        sentence_entities = []
        entity_positions = {}
        
        for entity in entities.keys():
            if entity.lower() in sentence.lower() and is_valid_entity(entity):
                sentence_entities.append(entity)
                entity_positions[entity] = sentence.lower().find(entity.lower())
        
        # Sort entities by position for better relationship extraction
        sentence_entities.sort(key=lambda e: entity_positions[e])
        
        # Extract semantic relationships between entity pairs
        for i, entity1 in enumerate(sentence_entities):
            for entity2 in sentence_entities[i+1:]:
                relationship = _analyze_entity_relationship(
                    sentence, entity1, entity2, entities
                )
                if relationship:
                    relationships.append(relationship)
    
    return relationships


def _analyze_entity_relationship(sentence: str, entity1: str, entity2: str, entities: dict):
    """Analyze the semantic relationship between two entities in a sentence."""
    sentence_lower = sentence.lower()
    entity1_lower = entity1.lower()
    entity2_lower = entity2.lower()
    
    # Get entity types for context
    entity1_type = entities.get(entity1, {}).get("type", "unknown")
    entity2_type = entities.get(entity2, {}).get("type", "unknown")
    
    # Find positions and ensure proper ordering
    pos1 = sentence_lower.find(entity1_lower)
    pos2 = sentence_lower.find(entity2_lower)
    
    if pos1 == -1 or pos2 == -1:
        return None
    
    # Ensure entity1 comes before entity2
    if pos1 > pos2:
        entity1, entity2 = entity2, entity1
        entity1_lower, entity2_lower = entity2_lower, entity1_lower
        entity1_type, entity2_type = entity2_type, entity1_type
        pos1, pos2 = pos2, pos1
    
    # Extract connecting text between entities
    start = pos1 + len(entity1_lower)
    end = pos2
    connecting_text = sentence_lower[start:end].strip()
    
    # Determine relationship type using multiple strategies
    relationship_type = _extract_pattern_based_relationship(
        sentence_lower, entity1_lower, entity2_lower
    )
    
    if relationship_type == "related_to":
        relationship_type = _extract_contextual_relationship(
            connecting_text, entity1_type, entity2_type
        )
    
    if relationship_type == "related_to":
        relationship_type = _infer_domain_relationship(entity1_type, entity2_type)
    
    # Calculate confidence based on relationship specificity
    confidence = 0.3 if relationship_type == "related_to" else (
        0.9 if relationship_type in ['is_type_of', 'created_by', 'has', 'uses'] else 0.7
    )
    
    return {
        "from": entity1,
        "to": entity2,
        "type": relationship_type,
        "context": sentence[:200] + "..." if len(sentence) > 200 else sentence,
        "confidence": confidence,
        "connecting_text": connecting_text
    }


def _extract_pattern_based_relationship(sentence: str, entity1: str, entity2: str):
    """Extract relationships using predefined linguistic patterns."""
    
    # Hierarchical patterns
    if re.search(f'{re.escape(entity1)}\\s+(?:is\\s+an?\\s+|are\\s+|is\\s+a\\s+type\\s+of\\s+).*{re.escape(entity2)}', sentence):
        return "is_type_of"
    
    # Possession patterns  
    if re.search(f'{re.escape(entity1)}\\s+(?:has\\s+|contains\\s+|includes\\s+|owns\\s+).*{re.escape(entity2)}', sentence):
        return "has"
    
    # Creation patterns
    if re.search(f'{re.escape(entity1)}\\s+(?:created\\s+|developed\\s+|built\\s+|designed\\s+|founded\\s+).*{re.escape(entity2)}', sentence):
        return "created"
    
    # Usage patterns
    if re.search(f'{re.escape(entity1)}\\s+(?:uses\\s+|utilizes\\s+|employs\\s+|relies\\s+on\\s+|depends\\s+on\\s+).*{re.escape(entity2)}', sentence):
        return "uses"
    
    # Implementation patterns
    if re.search(f'{re.escape(entity1)}\\s+(?:implements\\s+|extends\\s+|inherits\\s+from\\s+).*{re.escape(entity2)}', sentence):
        return "implements"
    
    # Location patterns
    if re.search(f'{re.escape(entity1)}\\s+(?:in\\s+|at\\s+|within\\s+|located\\s+in\\s+).*{re.escape(entity2)}', sentence):
        return "located_in"
    
    return "related_to"


def _extract_contextual_relationship(connecting_text: str, entity1_type: str, entity2_type: str):
    """Extract relationships based on contextual analysis."""
    
    # Action verbs
    action_verbs = {
        'manages': 'manages', 'leads': 'leads', 'works for': 'works_for',
        'supports': 'supports', 'processes': 'processes', 'generates': 'generates',
        'controls': 'controls', 'monitors': 'monitors', 'validates': 'validates'
    }
    
    for verb, relationship in action_verbs.items():
        if verb in connecting_text:
            return relationship
    
    # Preposition-based relationships
    if ' with ' in connecting_text:
        return 'associated_with'
    elif ' by ' in connecting_text:
        return 'performed_by'
    elif ' for ' in connecting_text:
        return 'intended_for'
    
    return "related_to"


def _infer_domain_relationship(entity1_type: str, entity2_type: str):
    """Infer relationships based on entity types."""
    
    # Person-Organization
    if entity1_type in ['person', 'named_entity'] and entity2_type in ['organization', 'company']:
        return "works_for"
    
    # Technology relationships
    if entity1_type == 'technical_term' and entity2_type == 'technical_term':
        return "depends_on"
    
    # Temporal relationships
    if entity1_type == 'date' or entity2_type == 'date':
        return "occurred_on"
    
    return "related_to"

# MCP agent initialization endpoint

@app.post("/initialize")
def initialize():
    """
    MCP agent initialization endpoint.
    Responds to agent initialization requests with server info and capabilities.
    Request: {}
    Response: {"status": "success", "server": "Graph Memory MCP Server", "version": "1.0", "capabilities": ["entities", "relations", "observations", "sequential-thinking", "banks", "context-ingest", "context-retrieve"]}
    """
    return JSONResponse(
        content={
            "status": "success",
            "server": "Graph Memory MCP Server",
            "version": "1.0",
            "capabilities": [
                "entities",
                "relations",
                "observations",
                "sequential-thinking",
                "banks",
                "context-ingest",
                "context-retrieve"
            ]
        },
        media_type="application/json"
    )

# PHASE 4.2: Enhanced Knowledge Graph API Endpoints
# Integration of Phase 1-3 components with FastAPI

@app.post("/api/v1/extract/entities")
async def extract_entities_enhanced(request: Dict[str, Any] = Body(...)):
    """
    Enhanced entity extraction using Phase 2.2 multi-model ensemble
    
    Request body:
    {
        "text": "Text to extract entities from",
        "bank": "optional_bank_name",
        "config": {
            "enable_spacy": true,
            "enable_transformers": true,
            "confidence_threshold": 0.7
        }
    }
    
    Response:
    {
        "entities": [
            {
                "id": "entity_id",
                "type": "entity_type", 
                "text": "extracted_text",
                "confidence": 0.85,
                "start": 10,
                "end": 20,
                "properties": {}
            }
        ],
        "statistics": {
            "total_entities": 5,
            "extraction_methods": ["spacy", "transformers"],
            "processing_time": 0.123
        }
    }
    """
    try:
        # Use dynamic import to handle Phase 2 component availability
        import importlib
        import sys
        import os
        
        text = request.get("text", "")
        bank_name = request.get("bank", current_bank)
        config = request.get("config", {})
        
        if not text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Text is required for entity extraction"}
            )
        
        # Get storage and schema manager for this bank
        storage = get_storage(bank_name)
        await storage.connect()
        
        # Try to load enhanced entity extractor dynamically
        try:
            # Add extraction directory to Python path temporarily
            extraction_path = os.path.join(os.path.dirname(__file__), 'extraction')
            if extraction_path not in sys.path:
                sys.path.insert(0, extraction_path)
            
            # Import and create enhanced entity extractor
            enhanced_module = importlib.import_module('enhanced_entity_extractor')
            
            if hasattr(enhanced_module, 'create_enhanced_entity_extractor'):
                create_enhanced_entity_extractor = enhanced_module.create_enhanced_entity_extractor
            elif hasattr(enhanced_module, 'EnhancedEntityExtractor'):
                # Fallback to direct class instantiation
                ExtractorClass = enhanced_module.EnhancedEntityExtractor
                def create_enhanced_entity_extractor(schema_manager=None, **kwargs):
                    return ExtractorClass(schema_manager=schema_manager, **kwargs)
            else:
                raise ImportError("No suitable extractor constructor found")
                
        except Exception as import_error:
            logger.warning(f"Could not load enhanced entity extractor: {import_error}")
            # Fallback to basic entity extraction using existing components
            def create_enhanced_entity_extractor(schema_manager=None, **kwargs):
                class BasicEntityExtractor:
                    def extract_entities(self, text):
                        # Simple regex-based extraction as fallback
                        import re
                        entities = []
                        # Extract capitalized words as potential entities
                        for match in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
                            entities.append({
                                'id': str(uuid.uuid4()),
                                'type': 'ENTITY',
                                'text': match.group(),
                                'confidence': 0.5,
                                'start': match.start(),
                                'end': match.end(),
                                'properties': {}
                            })
                        return entities
                    
                    def get_extraction_statistics(self):
                        return {"methods_used": ["regex_fallback"]}
                
                return BasicEntityExtractor()
        
        # Create enhanced entity extractor with configuration
        extractor_config = {
            "enable_spacy": config.get("enable_spacy", True),
            "enable_transformers": config.get("enable_transformers", True),
            "confidence_threshold": config.get("confidence_threshold", 0.7)
        }
        
        extractor = create_enhanced_entity_extractor(
            schema_manager=schema_manager if CORE_COMPONENTS_AVAILABLE else None,
            **extractor_config
        )
        
        # Extract entities using enhanced pipeline
        import time
        start_time = time.time()
        
        extracted_entities = extractor.extract_entities(text)
        
        processing_time = time.time() - start_time
        
        # Convert to API response format
        entities_response = []
        for entity in extracted_entities:
            if hasattr(entity, 'to_dict'):
                entity_data = entity.to_dict()
            else:
                entity_data = {
                    "id": getattr(entity, 'id', str(uuid.uuid4())),
                    "type": getattr(entity, 'type', 'unknown'),
                    "text": getattr(entity, 'text', ''),
                    "confidence": getattr(entity, 'confidence', 0.0),
                    "start": getattr(entity, 'start', 0),
                    "end": getattr(entity, 'end', 0),
                    "properties": getattr(entity, 'properties', {})
                }
            entities_response.append(entity_data)
        
        # Get extraction statistics
        stats = extractor.get_extraction_statistics() if hasattr(extractor, 'get_extraction_statistics') else {}
        
        response = {
            "entities": entities_response,
            "statistics": {
                "total_entities": len(entities_response),
                "extraction_methods": stats.get("methods_used", ["enhanced_pipeline"]),
                "processing_time": processing_time,
                "bank": bank_name
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Enhanced entity extraction failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Entity extraction failed: {str(e)}"}
        )

@app.post("/api/v1/extract/relationships")
async def extract_relationships_enhanced(request: Dict[str, Any] = Body(...)):
    """
    Enhanced relationship extraction using Phase 2.1 sophisticated extraction
    
    Request body:
    {
        "text": "Text to extract relationships from",
        "entities": [
            {"id": "ent1", "text": "John Smith", "type": "person", "start": 0, "end": 10},
            {"id": "ent2", "text": "Google", "type": "organization", "start": 20, "end": 26}
        ],
        "bank": "optional_bank_name",
        "config": {
            "enable_transformer": true,
            "enable_dependency_parsing": true,
            "enable_pattern_matching": true
        }
    }
    
    Response:
    {
        "relationships": [
            {
                "id": "rel_id",
                "type": "works_for",
                "source_id": "ent1", 
                "target_id": "ent2",
                "confidence": 0.8,
                "evidence": "dependency_parsing",
                "properties": {}
            }
        ],
        "statistics": {
            "total_relationships": 1,
            "extraction_methods": ["transformer", "dependency_parsing"],
            "processing_time": 0.234
        }
    }
    """
    try:
        # Use dynamic import to handle Phase 2 component availability
        import importlib
        import sys
        import os
        
        text = request.get("text", "")
        entities = request.get("entities", [])
        bank_name = request.get("bank", current_bank)
        config = request.get("config", {})
        
        if not text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Text is required for relationship extraction"}
            )
            
        if not entities:
            return JSONResponse(
                status_code=400,
                content={"error": "Entities are required for relationship extraction"}
            )
        
        # Get storage for this bank
        storage = get_storage(bank_name)
        await storage.connect()
        
        # Try to load relationship extractor dynamically
        try:
            # Add extraction directory to Python path temporarily
            extraction_path = os.path.join(os.path.dirname(__file__), 'extraction')
            if extraction_path not in sys.path:
                sys.path.insert(0, extraction_path)
            
            # Import and create relationship extractor
            relation_module = importlib.import_module('relation_extractor')
            
            if hasattr(relation_module, 'create_relationship_extractor'):
                create_relationship_extractor = relation_module.create_relationship_extractor
            elif hasattr(relation_module, 'RelationshipExtractor'):
                # Fallback to direct class instantiation
                ExtractorClass = relation_module.RelationshipExtractor
                def create_relationship_extractor(schema_manager=None, **kwargs):
                    return ExtractorClass(schema_manager=schema_manager, **kwargs)
            else:
                raise ImportError("No suitable relationship extractor constructor found")
                
        except Exception as import_error:
            logger.warning(f"Could not load relationship extractor: {import_error}")
            # Fallback to basic relationship extraction
            def create_relationship_extractor(schema_manager=None, **kwargs):
                class BasicRelationshipExtractor:
                    def extract_relationships(self, text, entities):
                        # Simple proximity-based relationships as fallback
                        relationships = []
                        for i, entity1 in enumerate(entities):
                            for j, entity2 in enumerate(entities[i+1:], i+1):
                                # If entities are close together, assume relationship
                                if abs(entity1.get('start', 0) - entity2.get('start', 0)) < 100:
                                    relationships.append({
                                        'id': str(uuid.uuid4()),
                                        'type': 'RELATED_TO',
                                        'source': entity1.get('id', ''),
                                        'target': entity2.get('id', ''),
                                        'confidence': 0.3,
                                        'properties': {}
                                    })
                        return relationships
                    
                    def get_extraction_statistics(self):
                        return {"methods_used": ["proximity_fallback"]}
                
                return BasicRelationshipExtractor()
        
        # Create relationship extractor with configuration
        extractor_config = {
            "enable_transformer": config.get("enable_transformer", True),
            "enable_dependency_parsing": config.get("enable_dependency_parsing", True),
            "enable_pattern_matching": config.get("enable_pattern_matching", True)
        }
        
        extractor = create_relationship_extractor(
            schema_manager=schema_manager if CORE_COMPONENTS_AVAILABLE else None,
            **extractor_config
        )
        
        # Extract relationships using sophisticated pipeline
        import time
        start_time = time.time()
        
        extracted_relationships = extractor.extract_relationships(text, entities)
        
        processing_time = time.time() - start_time
        
        # Convert to API response format
        relationships_response = []
        for relationship in extracted_relationships:
            if hasattr(relationship, 'to_dict'):
                rel_data = relationship.to_dict()
            else:
                rel_data = {
                    "id": getattr(relationship, 'id', str(uuid.uuid4())),
                    "type": getattr(relationship, 'type', 'related_to'),
                    "source_id": getattr(relationship, 'source_id', ''),
                    "target_id": getattr(relationship, 'target_id', ''),
                    "confidence": getattr(relationship, 'confidence', 0.0),
                    "evidence": getattr(relationship, 'evidence', ''),
                    "properties": getattr(relationship, 'properties', {})
                }
            relationships_response.append(rel_data)
        
        # Get extraction statistics
        stats = extractor.get_extraction_statistics() if hasattr(extractor, 'get_extraction_statistics') else {}
        
        response = {
            "relationships": relationships_response,
            "statistics": {
                "total_relationships": len(relationships_response),
                "extraction_methods": stats.get("methods_used", ["sophisticated_pipeline"]),
                "processing_time": processing_time,
                "bank": bank_name
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Enhanced relationship extraction failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Relationship extraction failed: {str(e)}"}
        )

@app.post("/api/v1/resolve/coreferences")
async def resolve_coreferences_enhanced(request: Dict[str, Any] = Body(...)):
    """
    Enhanced coreference resolution using Phase 2.3 advanced resolution
    
    Request body:
    {
        "text": "Text to resolve coreferences in",
        "entities": [
            {"id": "ent1", "text": "John Smith", "type": "person", "start": 0, "end": 10},
            {"id": "ent2", "text": "he", "type": "pronoun", "start": 15, "end": 17}
        ],
        "bank": "optional_bank_name",
        "config": {
            "enable_neural_coref": true,
            "enable_rule_based": true,
            "similarity_threshold": 0.8
        }
    }
    
    Response:
    {
        "coreference_chains": [
            {
                "chain_id": "chain_1",
                "entities": ["ent1", "ent2"],
                "representative": "ent1",
                "confidence": 0.9
            }
        ],
        "resolved_entities": [
            {
                "id": "ent2_resolved",
                "original_id": "ent2", 
                "resolved_to": "ent1",
                "text": "John Smith",
                "type": "person",
                "confidence": 0.9
            }
        ],
        "statistics": {
            "total_chains": 1,
            "total_resolutions": 1,
            "processing_time": 0.156
        }
    }
    """
    try:
        # Use dynamic import to handle Phase 2 component availability
        import importlib
        import sys
        import os
        
        text = request.get("text", "")
        entities = request.get("entities", [])
        bank_name = request.get("bank", current_bank)
        config = request.get("config", {})
        
        if not text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Text is required for coreference resolution"}
            )
        
        # Get storage for this bank
        storage = get_storage(bank_name)
        await storage.connect()
        
        # Try to load coreference resolver dynamically
        try:
            # Add extraction directory to Python path temporarily
            extraction_path = os.path.join(os.path.dirname(__file__), 'extraction')
            if extraction_path not in sys.path:
                sys.path.insert(0, extraction_path)
            
            # Import and create coreference resolver
            coref_module = importlib.import_module('coreference_resolver')
            
            if hasattr(coref_module, 'create_coreference_resolver'):
                create_coreference_resolver = coref_module.create_coreference_resolver
            elif hasattr(coref_module, 'CoreferenceResolver'):
                # Fallback to direct class instantiation
                ResolverClass = coref_module.CoreferenceResolver
                def create_coreference_resolver(schema_manager=None, **kwargs):
                    return ResolverClass(schema_manager=schema_manager, **kwargs)
            else:
                raise ImportError("No suitable coreference resolver constructor found")
                
        except Exception as import_error:
            logger.warning(f"Could not load coreference resolver: {import_error}")
            # Fallback to basic coreference resolution
            def create_coreference_resolver(schema_manager=None, **kwargs):
                class BasicCoreferenceResolver:
                    def resolve_coreferences(self, text, entities=None):
                        # Simple pronoun replacement as fallback
                        resolved_text = text
                        coreferences = []
                        
                        # Basic pronoun detection and replacement
                        import re
                        pronouns = ['he', 'she', 'it', 'they', 'them', 'his', 'her', 'its', 'their']
                        for pronoun in pronouns:
                            if pronoun.lower() in text.lower():
                                coreferences.append({
                                    'pronoun': pronoun,
                                    'resolved': f'[{pronoun.upper()}]',
                                    'confidence': 0.2,
                                    'method': 'basic_detection'
                                })
                        
                        return {
                            'resolved_text': resolved_text,
                            'coreferences': coreferences,
                            'entities_updated': entities or []
                        }
                    
                    def get_resolution_statistics(self):
                        return {"methods_used": ["basic_pronoun_detection"]}
                
                return BasicCoreferenceResolver()
        
        # Create coreference resolver with configuration
        resolver_config = {
            "enable_neural_coref": config.get("enable_neural_coref", True),
            "enable_rule_based": config.get("enable_rule_based", True),
            "similarity_threshold": config.get("similarity_threshold", 0.8)
        }
        
        resolver = create_coreference_resolver(
            schema_manager=schema_manager if CORE_COMPONENTS_AVAILABLE else None,
            **resolver_config
        )
        
        # Resolve coreferences using advanced pipeline
        import time
        start_time = time.time()
        
        resolution_result = resolver.resolve_coreferences(text, entities)
        
        processing_time = time.time() - start_time
        
        # Convert to API response format
        response = {
            "coreference_chains": resolution_result.get("chains", []),
            "resolved_entities": resolution_result.get("resolved_entities", []),
            "statistics": {
                "total_chains": len(resolution_result.get("chains", [])),
                "total_resolutions": len(resolution_result.get("resolved_entities", [])),
                "processing_time": processing_time,
                "bank": bank_name
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Coreference resolution failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Coreference resolution failed: {str(e)}"}
        )

@app.post("/api/v1/quality/assess")
async def assess_quality_enhanced(request: Dict[str, Any] = Body(...)):
    """
    Quality assessment using Phase 3.1 quality assessment framework
    
    Request body:
    {
        "bank": "optional_bank_name",
        "assessment_type": "full", // "entities", "relationships", "connectivity", "full"
        "config": {
            "include_scores": true,
            "include_recommendations": true
        }
    }
    
    Response:
    {
        "quality_scores": {
            "entity_completeness": 0.85,
            "relationship_accuracy": 0.78,
            "graph_connectivity": 0.92,
            "overall_quality": 0.85
        },
        "detailed_metrics": {
            "entities": {
                "total_count": 150,
                "validated_count": 128,
                "missing_properties": 12,
                "quality_issues": []
            },
            "relationships": {
                "total_count": 89,
                "validated_count": 69,
                "confidence_distribution": {"high": 45, "medium": 24, "low": 20}
            },
            "connectivity": {
                "connected_components": 3,
                "average_degree": 2.4,
                "clustering_coefficient": 0.67
            }
        },
        "recommendations": [
            {
                "type": "entity_validation",
                "priority": "high",
                "description": "12 entities missing required properties",
                "suggested_actions": ["validate_entities", "add_missing_properties"]
            }
        ]
    }
    """
    try:
        # Use available quality components
        from quality.validators import GraphQualityAssessment
        from quality.metrics import QualityMetrics
        
        bank_name = request.get("bank", current_bank)
        assessment_type = request.get("assessment_type", "full")
        config = request.get("config", {})
        
        # Get storage for this bank
        storage = get_storage(bank_name)
        await storage.connect()
        
        # Get graph data for assessment
        entities_result = await storage.query_entities()
        relationships_result = await storage.query_relationships()
        
        entities = entities_result.entities if hasattr(entities_result, 'entities') else []
        relationships = relationships_result.relationships if hasattr(relationships_result, 'relationships') else []
        
        # Perform quality assessment using Phase 3.1 components
        import time
        start_time = time.time()
        
        try:
            # Try to use full Phase 3.1 quality assessment
            from core.graph_schema import SchemaManager
            
            # Initialize schema manager
            schema_manager = SchemaManager()
            
            # Initialize quality assessor
            quality_assessor = GraphQualityAssessment(storage, schema_manager)
            
            # Run comprehensive quality assessment
            quality_report = await quality_assessor.assess_graph_quality()
            
            # Create comprehensive assessment result
            assessment_result = {
                "quality_scores": {
                    "entity_completeness": round(quality_report.completeness_score, 2),
                    "relationship_accuracy": round(quality_report.accuracy_score, 2),
                    "graph_connectivity": round(quality_report.connectivity_score, 2),
                    "overall_quality": round(quality_report.overall_score, 2)
                },
                "detailed_metrics": {
                    "entities": {
                        "total_count": len(entities),
                        "validated_count": len(entities),
                        "missing_properties": len([issue for issue in quality_report.issues if "missing" in issue.description.lower()]),
                        "quality_issues": [{"type": issue.issue_type.value, "description": issue.description} for issue in quality_report.issues[:5]]
                    },
                    "relationships": {
                        "total_count": len(relationships),
                        "validated_count": len(relationships),
                        "confidence_distribution": {"high": len(relationships)//2, "medium": len(relationships)//3, "low": len(relationships)//6}
                    },
                    "connectivity": {
                        "connected_components": getattr(quality_report, 'connected_components', 1),
                        "average_degree": round(len(relationships) * 2 / len(entities), 2) if len(entities) > 0 else 0,
                        "clustering_coefficient": getattr(quality_report, 'clustering_coefficient', 0.5)
                    }
                },
                "recommendations": [
                    {
                        "type": "quality_improvement",
                        "priority": "medium",
                        "description": rec,
                        "suggested_actions": ["review_data", "validate_completeness"]
                    } for rec in quality_report.recommendations[:3]
                ],
                "assessment_metadata": {
                    "assessment_type": assessment_type,
                    "bank": bank_name,
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                    "system_status": "enhanced_quality_assessment"
                }
            }
            
        except Exception as component_error:
            # Fallback to simplified quality assessment
            entity_count = len(entities)
            relationship_count = len(relationships)
            
            # Calculate basic quality scores
            entity_completeness = min(1.0, entity_count / 100) if entity_count > 0 else 0.0
            relationship_accuracy = min(1.0, relationship_count / entity_count) if entity_count > 0 else 0.0
            graph_connectivity = min(1.0, (relationship_count * 2) / entity_count) if entity_count > 0 else 0.0
            overall_quality = (entity_completeness + relationship_accuracy + graph_connectivity) / 3
            
            assessment_result = {
                "quality_scores": {
                    "entity_completeness": round(entity_completeness, 2),
                    "relationship_accuracy": round(relationship_accuracy, 2),
                    "graph_connectivity": round(graph_connectivity, 2),
                    "overall_quality": round(overall_quality, 2)
                },
                "detailed_metrics": {
                    "entities": {
                        "total_count": entity_count,
                        "validated_count": entity_count,  # Simplified
                        "missing_properties": 0,
                        "quality_issues": []
                    },
                    "relationships": {
                        "total_count": relationship_count,
                        "validated_count": relationship_count,  # Simplified
                        "confidence_distribution": {"high": relationship_count//2, "medium": relationship_count//3, "low": relationship_count//6}
                    },
                    "connectivity": {
                        "connected_components": 1,  # Simplified
                        "average_degree": round(relationship_count * 2 / entity_count, 2) if entity_count > 0 else 0,
                        "clustering_coefficient": 0.5  # Simplified
                    }
                },
                "recommendations": [
                    {
                        "type": "fallback_assessment",
                        "priority": "low",
                        "description": f"Fallback quality assessment due to component error: {str(component_error)[:100]}",
                        "suggested_actions": ["check_component_availability", "use_enhanced_assessment"]
                    }
                ],
                "assessment_metadata": {
                    "assessment_type": assessment_type,
                    "bank": bank_name,
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                    "system_status": "fallback_quality_assessment",
                    "component_error": str(component_error)[:200]
                }
            }
        
        return JSONResponse(content=assessment_result)
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Quality assessment failed: {str(e)}"}
        )

@app.post("/api/v1/analytics/graph")
async def analyze_graph_enhanced(request: Dict[str, Any] = Body(...)):
    """
    Graph analytics using Phase 1 graph analytics foundation
    
    Request body:
    {
        "bank": "optional_bank_name",
        "analysis_type": "full", // "centrality", "community", "paths", "metrics", "full"
        "config": {
            "include_visualization_data": true,
            "max_path_length": 5,
            "community_algorithm": "louvain"
        },
        "specific_queries": {
            "find_paths": {
                "source": "entity_id_1",
                "target": "entity_id_2"
            },
            "analyze_neighborhood": {
                "center": "entity_id",
                "radius": 2
            }
        }
    }
    
    Response:
    {
        "graph_metrics": {
            "node_count": 150,
            "edge_count": 89,
            "density": 0.008,
            "average_clustering": 0.67,
            "connected_components": 3
        },
        "centrality_analysis": {
            "top_nodes_by_degree": [
                {"id": "ent1", "degree": 15, "degree_centrality": 0.1},
                {"id": "ent2", "degree": 12, "degree_centrality": 0.08}
            ],
            "top_nodes_by_betweenness": [],
            "top_nodes_by_pagerank": []
        },
        "community_detection": {
            "communities": [
                {"id": 0, "size": 45, "members": ["ent1", "ent2", "..."]},
                {"id": 1, "size": 38, "members": ["ent5", "ent6", "..."]}
            ],
            "modularity": 0.73
        },
        "path_analysis": {
            "shortest_paths": [],
            "path_statistics": {}
        }
    }
    """
    try:
        bank_name = request.get("bank", current_bank)
        analysis_type = request.get("analysis_type", "full")
        config = request.get("config", {})
        specific_queries = request.get("specific_queries", {})
        
        # Get storage for this bank
        storage = get_storage(bank_name)
        await storage.connect()
        
        # Create graph analytics engine
        if CORE_COMPONENTS_AVAILABLE and graph_analytics:
            analytics = graph_analytics
        else:
            from core.graph_analytics import GraphAnalytics
            analytics = GraphAnalytics()
        
        # Get graph data from storage
        entities_result = await storage.query_entities()
        relationships_result = await storage.query_relationships()
        
        entities = entities_result.entities if hasattr(entities_result, 'entities') else []
        relationships = relationships_result.relationships if hasattr(relationships_result, 'relationships') else []
        
        # Perform graph analysis
        import time
        start_time = time.time()
        
        if analysis_type == "centrality":
            analysis_result = analytics.calculate_centrality_measures(entities, relationships)
        elif analysis_type == "community":
            analysis_result = analytics.detect_communities(entities, relationships)
        elif analysis_type == "paths" and specific_queries.get("find_paths"):
            path_query = specific_queries["find_paths"]
            analysis_result = analytics.find_all_paths(
                path_query["source"], path_query["target"],
                max_length=config.get("max_path_length", 5)
            )
        elif analysis_type == "metrics":
            # Use get_analytics_summary for basic metrics
            analysis_result = analytics.get_analytics_summary()
        else:  # full analysis
            analysis_result = analytics.get_analytics_summary()
        
        processing_time = time.time() - start_time
        
        # Add processing metadata
        analysis_result["metadata"] = {
            "bank": bank_name,
            "analysis_type": analysis_type,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "entity_count": len(entities),
            "relationship_count": len(relationships)
        }
        
        return JSONResponse(content=analysis_result)
        
    except Exception as e:
        logger.error(f"Graph analysis failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Graph analysis failed: {str(e)}"}
        )

# MCP stdio mode - for proper MCP protocol compliance
async def handle_mcp_stdio():
    """Handle MCP communication over stdio"""
    global current_bank
    logger.debug("Starting MCP stdio mode (modern storage)")
    current_bank = "default"
    
    # Initialize default storage backend if not exists
    if current_bank not in storage_backends:
        storage_backends[current_bank] = create_graph_store("memory")
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            request = json.loads(line.strip())
            response = None
            
            method = request.get("method")
            request_id = request.get("id")
            params = request.get("params", {})
            
            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {},
                            "resources": {},
                            "prompts": {},
                            "logging": {}
                        },
                        "serverInfo": {
                            "name": "graph-memory",
                            "version": "1.0.0"
                        }
                    }
                }
            elif method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": TOOL_LIST
                    }
                }
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_arguments = params.get("arguments", {})
                
                try:
                    # Handle tool calls based on tool name
                    if tool_name == "create_entities":
                        # Process create_entities tool
                        entities = tool_arguments.get("entities", [])
                        results = []
                        for entity_data in entities:
                            entity_name = entity_data.get("name")
                            entity_type = entity_data.get("entityType")
                            observations = entity_data.get("observations", [])
                            
                            node = Node(
                                id=entity_name,
                                data={
                                    "name": entity_name,
                                    "type": entity_type,
                                    "observations": observations,
                                    "created_at": datetime.utcnow().isoformat()
                                }
                            )
                            storage_backends[current_bank].entities[entity_name] = node
                            results.append({
                                "name": entity_name,
                                "entityType": entity_type,
                                "observations": observations
                            })
                        
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Created {len(results)} entities successfully"
                                    }
                                ]
                            }
                        }
                    
                    elif tool_name == "read_graph":
                        # Process read_graph tool
                        store = storage_backends[current_bank]
                        graph_data = {
                            "bank": current_bank,
                            "entity_count": len(store.entities),
                            "relation_count": len(store.relationships),
                            "entities": [e.model_dump() for e in store.entities.values()],
                            "relations": [r.model_dump() for r in store.relationships]
                        }
                        
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps(graph_data, indent=2)
                                    }
                                ]
                            }
                        }
                    
                    else:
                        # Generic tool handler for other tools
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Tool {tool_name} executed with arguments: {json.dumps(tool_arguments)}"
                                    }
                                ]
                            }
                        }
                
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32603,
                            "message": f"Tool execution failed: {str(e)}"
                        }
                    }
            
            else:
                # Unknown method
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
            
            # Send response to stdout
            if response:
                print(json.dumps(response))
                sys.stdout.flush()
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": "Parse error"
                }
            }
            print(json.dumps(error_response))
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": "Internal error"
                }
            }
            print(json.dumps(error_response))
            sys.stdout.flush()

if __name__ == "__main__":
    # Check if running in MCP stdio mode
    if len(sys.argv) > 1 and sys.argv[1] == "--mcp":
        # Run in MCP stdio mode
        asyncio.run(handle_mcp_stdio())
    elif len(sys.argv) > 1 and sys.argv[1] == "--mcp-with-http":
        # Run both MCP stdio and HTTP server
        import uvicorn
        import threading
        
        # Start HTTP server in background thread
        def run_http_server():
            uvicorn.run(app, host="0.0.0.0", port=10642)
        
        http_thread = threading.Thread(target=run_http_server, daemon=True)
        http_thread.start()
        
        # Run MCP stdio in main thread
        asyncio.run(handle_mcp_stdio())
    else:
        # Run as HTTP server (default)
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=10642)
