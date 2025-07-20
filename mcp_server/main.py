from fastapi import FastAPI, Query, Body, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import asyncio
import time
import logging
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI()

# Data persistence configuration
DATA_DIR = Path("/data")
MEMORY_FILE = DATA_DIR / "memory_banks.json"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

def event_stream():
    # Send a keepalive event every 5 seconds
    while True:
        yield "data: keepalive\n\n"
        time.sleep(5)

@app.get("/events")
def sse_events():
    """
    Dummy Server-Sent Events endpoint for agent compatibility.
    """
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Logging middleware for debugging

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"Incoming request: {request.method} {request.url}")
    logging.info(f"Headers: {dict(request.headers)}")
    
    # Don't consume the request body in middleware - this prevents route handlers from reading it!
    # The actual route handlers will read the body when needed
    
    response = await call_next(request)
    logging.info(f"Response status: {response.status_code}")
    return response

import re
import uuid
from datetime import datetime

# Simple in-memory graph structure
class Node(BaseModel):
    id: str
    type: str = "node"
    data: Dict[str, Any] = {}

class Edge(BaseModel):
    id: str = None
    source: str
    target: str
    type: str = "relation"
    data: Dict[str, Any] = {}

class Observation(BaseModel):
    id: str
    entity_id: str
    content: str
    timestamp: str

from pydantic import Field

class ReasoningStep(BaseModel):
    id: str
    description: str
    status: str = "pending"
    timestamp: str = None
    related_entities: List[str] = Field(default_factory=list)
    related_relations: List[str] = Field(default_factory=list)

class TextIngest(BaseModel):
    text: str
    bank: str = "default"

class BankOp(BaseModel):
    bank: str

# Memory banks: each bank has its own nodes, edges, observations, reasoning_steps
memory_banks = {"default": {"nodes": {}, "edges": [], "observations": [], "reasoning_steps": []}}
current_bank = "default"

# Persistence functions
def serialize_memory_banks():
    """Convert memory banks to JSON-serializable format"""
    serialized = {}
    for bank_name, bank_data in memory_banks.items():
        try:
            # Use dict() instead of model_dump() for compatibility
            nodes_dict = {}
            for node_id, node in bank_data["nodes"].items():
                try:
                    # Try model_dump first, fallback to dict
                    nodes_dict[node_id] = node.model_dump() if hasattr(node, 'model_dump') else node.dict()
                except Exception:
                    # If that fails, create manual dict
                    nodes_dict[node_id] = {
                        "id": node.id,
                        "data": node.data,
                        "created_at": node.created_at
                    }
            
            serialized[bank_name] = {
                "nodes": nodes_dict,
                "edges": [edge.model_dump() for edge in bank_data["edges"]],
                "observations": [obs.model_dump() for obs in bank_data["observations"]],
                "reasoning_steps": [step.model_dump() for step in bank_data["reasoning_steps"]]
            }
        except Exception as e:
            logger.error(f"Error serializing bank {bank_name}: {e}")
            # Skip problematic bank rather than failing entirely
            continue
    return serialized

def deserialize_memory_banks(data):
    """Convert JSON data back to memory banks with proper objects"""
    global memory_banks
    memory_banks = {}
    for bank_name, bank_data in data.items():
        memory_banks[bank_name] = {
            "nodes": {node_id: Node(**node_data) for node_id, node_data in bank_data["nodes"].items()},
            "edges": [Edge(**edge_data) for edge_data in bank_data["edges"]],
            "observations": [Observation(**obs_data) for obs_data in bank_data["observations"]],
            "reasoning_steps": [ReasoningStep(**step_data) for step_data in bank_data["reasoning_steps"]]
        }

def save_memory_banks():
    """Save memory banks to persistent storage"""
    try:
        serialized_data = serialize_memory_banks()
        
        # Ensure DATA_DIR exists and is writable
        try:
            DATA_DIR.mkdir(exist_ok=True)
        except PermissionError as e:
            logger.error(f"Cannot create data directory {DATA_DIR}: {e}")
            return
        
        # Check if directory is writable
        if not os.access(DATA_DIR, os.W_OK):
            logger.error(f"Data directory {DATA_DIR} is not writable")
            return
        
        # Write to temp file first, then rename for atomic operation
        temp_file = MEMORY_FILE.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(serialized_data, f, indent=2)
            
            # Atomic rename
            temp_file.rename(MEMORY_FILE)
            logger.info(f"Memory banks saved to {MEMORY_FILE}")
        except PermissionError as e:
            logger.error(f"Permission denied writing to {temp_file}: {e}")
            # Try to cleanup temp file if it exists
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
        except Exception as e:
            logger.error(f"Error writing memory banks file: {e}")
            # Try to cleanup temp file if it exists
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
    except Exception as e:
        logger.error(f"Failed to save memory banks: {e}")
        # Continue execution even if save fails - data persists in memory

def load_memory_banks():
    """Load memory banks from persistent storage"""
    global memory_banks, current_bank
    try:
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
            deserialize_memory_banks(data)
            logger.debug(f"Memory banks loaded from {MEMORY_FILE}")
            logger.debug(f"Loaded banks: {list(memory_banks.keys())}")
        else:
            logger.debug("No existing memory file found, starting with default banks")
            memory_banks = {"default": {"nodes": {}, "edges": [], "observations": [], "reasoning_steps": []}}
        
        # Ensure current_bank exists
        if current_bank not in memory_banks:
            current_bank = "default"
            
    except Exception as e:
        logger.error(f"Failed to load memory banks: {e}")
        logger.info("Starting with default memory banks")
        memory_banks = {"default": {"nodes": {}, "edges": [], "observations": [], "reasoning_steps": []}}
        current_bank = "default"

# Load existing data on startup
load_memory_banks()


# Bank management endpoints
@app.post("/banks/create")
def create_bank(op: BankOp):
    """
    Create a new memory bank.
    Request body: {"bank": "bank_name"}
    Response: {"status": "success", "bank": "bank_name"}
    """
    if op.bank in memory_banks:
        return {"status": "error", "message": "Bank already exists."}
    
    # Create the new bank
    memory_banks[op.bank] = {"nodes": {}, "edges": [], "observations": [], "reasoning_steps": []}
    
    # Save to disk
    try:
        save_memory_banks()
    except Exception as e:
        logger.warning(f"Could not save memory banks: {e}")
        # Continue anyway - bank is created in memory
    
    return {"status": "success", "bank": op.bank}

@app.post("/banks/select")
def select_bank(op: BankOp):
    """
    Select the active memory bank.
    Request body: {"bank": "bank_name"}
    Response: {"status": "success", "selected": "bank_name"}
    """
    global current_bank
    if op.bank not in memory_banks:
        return {"status": "error", "message": "Bank does not exist."}
    current_bank = op.bank
    # No need to save for selection - it's just runtime state
    return {"status": "success", "selected": current_bank}
    return {"status": "success", "selected": current_bank}

@app.get("/banks/list")
def list_banks():
    """
    List all memory banks and show the current active bank.
    Response: {"banks": ["bank1", "bank2"], "current": "bank_name"}
    """
    return {"banks": list(memory_banks.keys()), "current": current_bank}

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
    if op.bank not in memory_banks:
        return {"status": "error", "message": "Bank does not exist."}
    del memory_banks[op.bank]
    if current_bank == op.bank:
        current_bank = "default"
    save_memory_banks()  # Persist the change
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

# Bank-aware entity endpoints

@app.post("/entities")
def add_entity(node: Node = Body(...), bank: str = Query(None)):
    """
    Add an entity (node) to the selected or specified bank.
    Request body: Node model
    Query param: bank (optional)
    Response: {"status": "success", "entity": Node, "bank": "bank_name"}
    """
    b = bank or current_bank
    try:
        if not isinstance(node, Node):
            node = Node(**node.dict() if hasattr(node, 'dict') else node)
        memory_banks[b]["nodes"][node.id] = node
        save_memory_banks()  # Persist the change
        return {"status": "success", "entity": node, "bank": b}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/entities")
def get_entities(bank: str = Query(None)):
    """
    Get all entities (nodes) from the selected or specified bank.
    Query param: bank (optional)
    Response: List of Node dicts
    """
    b = bank or current_bank
    return [n.dict() for n in memory_banks[b]["nodes"].values()]


@app.put("/entities/{entity_id}")
def update_entity(entity_id: str, node: Node, bank: str = Query(None)):
    """
    Update an entity (node) in the selected or specified bank.
    Request body: Node model
    Query param: bank (optional)
    Response: {"status": "success", "entity": Node, "bank": "bank_name"}
    """
    b = bank or current_bank
    memory_banks[b]["nodes"][entity_id] = node
    return {"status": "success", "entity": node, "bank": b}


@app.delete("/entities/{entity_id}")
def delete_entity(entity_id: str, bank: str = Query(None)):
    """
    Delete an entity (node) from the selected or specified bank.
    Query param: bank (optional)
    Response: {"status": "success", "deleted": "entity_id", "bank": "bank_name"}
    """
    b = bank or current_bank
    if entity_id in memory_banks[b]["nodes"]:
        del memory_banks[b]["nodes"][entity_id]
        return {"status": "success", "deleted": entity_id, "bank": b}
    return {"status": "error", "message": "Entity not found", "bank": b}

# Bank-aware relation endpoints

@app.post("/relations")
def add_relation(edge: Edge = Body(...), bank: str = Query(None)):
    """
    Add a relation (edge) to the selected or specified bank.
    Request body: Edge model
    Query param: bank (optional)
    Response: {"status": "success", "relation": Edge, "bank": "bank_name"}
    """
    b = bank or current_bank
    try:
        edge.id = edge.id or f"{edge.source}-{edge.target}-{len(memory_banks[b]['edges'])}"
        if not isinstance(edge, Edge):
            edge = Edge(**edge.dict() if hasattr(edge, 'dict') else edge)
        memory_banks[b]["edges"].append(edge)
        save_memory_banks()  # Persist the change
        return {"status": "success", "relation": edge, "bank": b}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/relations")
def get_relations(bank: str = Query(None)):
    """
    Get all relations (edges) from the selected or specified bank.
    Query param: bank (optional)
    Response: List of Edge dicts
    """
    b = bank or current_bank
    return [e.dict() for e in memory_banks[b]["edges"]]


@app.put("/relations/{relation_id}")
def update_relation(relation_id: str, edge: Edge, bank: str = Query(None)):
    """
    Update a relation (edge) in the selected or specified bank.
    Request body: Edge model
    Query param: bank (optional)
    Response: {"status": "success", "relation": Edge, "bank": "bank_name"}
    """
    b = bank or current_bank
    for i, e in enumerate(memory_banks[b]["edges"]):
        if e.id == relation_id:
            memory_banks[b]["edges"][i] = edge
            return {"status": "success", "relation": edge, "bank": b}
    return {"status": "error", "message": "Relation not found", "bank": b}


@app.delete("/relations/{relation_id}")
def delete_relation(relation_id: str, bank: str = Query(None)):
    """
    Delete a relation (edge) from the selected or specified bank.
    Query param: bank (optional)
    Response: {"status": "success", "deleted": "relation_id", "bank": "bank_name"}
    """
    b = bank or current_bank
    for i, e in enumerate(memory_banks[b]["edges"]):
        if e.id == relation_id:
            del memory_banks[b]["edges"][i]
            return {"status": "success", "deleted": relation_id, "bank": b}
    return {"status": "error", "message": "Relation not found", "bank": b}

# Bank-aware observation endpoints

@app.post("/observations")
def add_observation(obs: Observation = Body(...), bank: str = Query(None)):
    """
    Add an observation to the selected or specified bank.
    Request body: Observation model
    Query param: bank (optional)
    Response: {"status": "success", "observation": Observation, "bank": "bank_name"}
    """
    b = bank or current_bank
    try:
        if not isinstance(obs, Observation):
            obs = Observation(**obs.dict() if hasattr(obs, 'dict') else obs)
        memory_banks[b]["observations"].append(obs)
        save_memory_banks()  # Persist the change
        return {"status": "success", "observation": obs, "bank": b}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/observations")
def get_observations(bank: str = Query(None)):
    """
    Get all observations from the selected or specified bank.
    Query param: bank (optional)
    Response: List of Observation dicts
    """
    b = bank or current_bank
    return [o.dict() for o in memory_banks[b]["observations"]]

# Bank-aware sequential thinking endpoints

@app.post("/sequential-thinking")
def add_reasoning_step(step: ReasoningStep = Body(...), bank: str = Query(None)):
    """
    Add a reasoning step to the selected or specified bank.
    Request body: ReasoningStep model
    Query param: bank (optional)
    Response: {"status": "success", "step": ReasoningStep, "bank": "bank_name"}
    """
    b = bank or current_bank
    try:
        if not isinstance(step, ReasoningStep):
            step = ReasoningStep(**step.dict() if hasattr(step, 'dict') else step)
        memory_banks[b]["reasoning_steps"].append(step)
        save_memory_banks()  # Persist the change
        return {"status": "success", "step": step, "bank": b}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/sequential-thinking")
def get_reasoning_steps(bank: str = Query(None)):
    """
    Get all reasoning steps from the selected or specified bank.
    Query param: bank (optional)
    Response: List of ReasoningStep dicts
    """
    b = bank or current_bank
    return [s.dict() for s in memory_banks[b]["reasoning_steps"]]


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
    try:
        results = search_entities(q, bank, entity_type, case_sensitive, use_regex, fuzzy_match, fuzzy_threshold, limit)
        return {
            "query": q,
            "bank": bank or current_bank,
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
    try:
        results = search_relationships(q, bank, relationship_type, case_sensitive, use_regex)
        return {
            "query": q,
            "bank": bank or current_bank,
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
    try:
        results = search_observations(q, bank, entity_id, case_sensitive, use_regex)
        return {
            "query": q,
            "bank": bank or current_bank,
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
    
    # Convert entities to visualization nodes
    for entity_id, entity in memory_banks[b]["nodes"].items():
        nodes.append({
            "id": entity_id,
            "label": entity_id,
            "type": entity.data.get("type", "unknown"),
            "size": len(entity.data.get("observations", [])) + 1,
            "observations": entity.data.get("observations", []),
            "created_at": entity.data.get("created_at", ""),
            "updated_at": entity.data.get("updated_at", "")
        })
    
    # Convert relationships to visualization edges
    for edge in memory_banks[b]["edges"]:
        edges.append({
            "source": edge.source,
            "target": edge.target,
            "type": edge.data.get("type", "related_to"),
            "label": edge.data.get("type", "related_to")
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

@app.get("/banks/{bank}/visualize")
def visualize_graph(bank: str):
    """
    Serve interactive graph visualization page for a specific memory bank.
    """
    if bank not in memory_banks:
        return JSONResponse(content={"error": "Bank not found"}, status_code=404)

    try:
        # Read the HTML template from file
        template_path = Path(__file__).parent / "templates" / "visualization.html"
        with open(template_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Replace template variables
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
            status_code=500
        )

@app.get("/")
async def root():
    """Root endpoint - redirects to visualization"""
    # Default to first available bank or 'default'
    default_bank = 'default' if 'default' in memory_banks else list(memory_banks.keys())[0] if memory_banks else 'default'
    
    # Render the enhanced visualization page
    return visualize_graph(default_bank)

# New endpoint: ingest long text and update/extend graph in a specific bank

@app.post("/context/ingest")
def ingest_context(payload: TextIngest = Body(...)):
    """
    Ingest a long piece of text and update/extend the graph in the selected or specified bank.
    Request body: {"text": "...", "bank": "bank_name"}
    Response: {"status": "success", "entities": [...], "edges_added": N, "bank": "bank_name"}
    """
    text = payload.text
    b = payload.bank or current_bank
    # Simple entity extraction: words starting with capital letters (as example)
    entities = set(re.findall(r'\b[A-Z][a-zA-Z0-9_]+\b', text))
    # Add entities as nodes
    for ent in entities:
        if ent not in memory_banks[b]["nodes"]:
            node = Node(id=ent, data={"from_text": True})
            memory_banks[b]["nodes"][ent] = node
    # Simple relationship extraction: pairs of entities in the same sentence
    sentences = re.split(r'[.!?]', text)
    for sentence in sentences:
        ents_in_sentence = [ent for ent in entities if ent in sentence]
        for i in range(len(ents_in_sentence)-1):
            edge = Edge(source=ents_in_sentence[i], target=ents_in_sentence[i+1], data={"from_text": True})
            memory_banks[b]["edges"].append(edge)
    save_memory_banks()  # Persist the changes
    return {"status": "success", "entities": list(entities), "edges_added": len(memory_banks[b]["edges"]), "bank": b}


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

# MCP stdio mode - for proper MCP protocol compliance
async def handle_mcp_stdio():
    """Handle MCP communication over stdio"""
    global current_bank
    logger.debug("Starting MCP stdio mode")
    
    while True:
        try:
            # Read JSON-RPC request from stdin
            line = sys.stdin.readline()
            if not line:
                break
                
            request = json.loads(line.strip())
            
            # Handle the request
            if request.get("method") == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "serverInfo": {
                            "name": "Graph Memory MCP Server",
                            "version": "1.0"
                        },
                        "capabilities": {
                            "tools": {"listChanged": True},
                            "resources": {"subscribe": False, "listChanged": True},
                            "roots": {"listChanged": True},
                            "prompts": {"listChanged": False},
                            "completion": {"supports": ["text"]}
                        }
                    }
                }
            elif request.get("method") == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "tools": [
                            {
                                "name": "create_entities",
                                "description": "Create multiple new entities in the knowledge graph with optional auto-extraction of additional entities and relationships from observations. IMPORTANT: Use separate memory banks for different topics/projects (e.g., 'client-acme-project', 'personal-research'). Create banks using POST /banks/create and switch using POST /banks/select before creating entities. Never mix unrelated topics in the same bank.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "entities": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "entityType": {"type": "string"},
                                                    "observations": {"type": "array", "items": {"type": "string"}}
                                                }
                                            }
                                        },
                                        "auto_extract": {
                                            "type": "boolean", 
                                            "description": "Whether to automatically extract additional entities and relationships from observation text (default: true)",
                                            "default": True
                                        }
                                    }
                                }
                            },
                            {
                                "name": "add_observations",
                                "description": "Add new observations to existing entities",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "observations": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "entityName": {"type": "string"},
                                                    "contents": {"type": "array", "items": {"type": "string"}}
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            {
                                "name": "create_relations",
                                "description": "Create relations between entities",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "relations": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "from": {"type": "string"},
                                                    "to": {"type": "string"},
                                                    "relationType": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            {
                                "name": "sequential_thinking",
                                "description": "Add reasoning steps to the knowledge graph",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "step": {
                                            "type": "object",
                                            "properties": {
                                                "thought": {"type": "string"},
                                                "step_number": {"type": "number"},
                                                "reasoning": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            },
                            {
                                "name": "ingest_knowledge",
                                "description": "Create a knowledge graph from large text with advanced entity and relationship extraction",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string", "description": "Large text content to analyze"},
                                        "bank": {"type": "string", "description": "Memory bank name"},
                                        "source": {"type": "string", "description": "Source identifier"},
                                        "extract_entities": {"type": "boolean", "description": "Extract entities"},
                                        "extract_relationships": {"type": "boolean", "description": "Extract relationships"},
                                        "create_observations": {"type": "boolean", "description": "Create observations"}
                                    },
                                    "required": ["text"]
                                }
                            },
                            {
                                "name": "search_nodes",
                                "description": "Search for nodes in the knowledge graph based on a query. IMPORTANT: Search within specific banks using bank parameter to avoid cross-topic contamination. Use GET /banks/list to see available banks first.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "The search query to match against entity names, types, and observation content"},
                                        "bank": {"type": "string", "description": "Optional: Memory bank to search in (e.g., 'client-acme-project'). If not specified, searches current bank."}
                                    },
                                    "required": ["query"]
                                }
                            },
                            {
                                "name": "delete_entities",
                                "description": "Delete multiple entities and their associated relations from the knowledge graph",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "entityNames": {"type": "array", "items": {"type": "string"}, "description": "An array of entity names to delete"}
                                    },
                                    "required": ["entityNames"]
                                }
                            },
                            {
                                "name": "delete_relations",
                                "description": "Delete multiple relations from the knowledge graph",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "relations": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "from": {"type": "string", "description": "The name of the entity where the relation starts"},
                                                    "to": {"type": "string", "description": "The name of the entity where the relation ends"},
                                                    "relationType": {"type": "string", "description": "The type of the relation"}
                                                },
                                                "required": ["from", "to", "relationType"]
                                            },
                                            "description": "An array of relations to delete"
                                        }
                                    },
                                    "required": ["relations"]
                                }
                            },
                            {
                                "name": "delete_observations",
                                "description": "Delete specific observations from entities in the knowledge graph",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "deletions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "entityName": {"type": "string", "description": "The name of the entity containing the observations"},
                                                    "observations": {"type": "array", "items": {"type": "string"}, "description": "An array of observations to delete"}
                                                },
                                                "required": ["entityName", "observations"]
                                            },
                                            "description": "An array of deletion specifications"
                                        }
                                    },
                                    "required": ["deletions"]
                                }
                            },
                            {
                                "name": "read_graph",
                                "description": "Read the entire knowledge graph summary for the current memory bank. Shows entity/relationship counts and types. Use this to understand bank contents before adding new knowledge.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "bank": {"type": "string", "description": "Optional: Memory bank to read (e.g., 'client-acme-project'). If not specified, reads current bank."}
                                    }
                                }
                            },
                            {
                                "name": "open_nodes",
                                "description": "Open specific nodes in the knowledge graph by their names",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "names": {"type": "array", "items": {"type": "string"}, "description": "An array of entity names to retrieve"}
                                    },
                                    "required": ["names"]
                                }
                            },
                            {
                                "name": "create_bank",
                                "description": "Create a new memory bank for organizing different topics/projects. CRITICAL: Always create specific banks for different projects - never use 'default' for real work.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "bank": {"type": "string", "description": "Name of the memory bank to create (e.g., 'project-acme-auth', 'research-ai-optimization')"}
                                    },
                                    "required": ["bank"]
                                }
                            },
                            {
                                "name": "select_bank",
                                "description": "Switch to a different memory bank. All subsequent operations will operate on the selected bank.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "bank": {"type": "string", "description": "Name of the memory bank to switch to"}
                                    },
                                    "required": ["bank"]
                                }
                            },
                            {
                                "name": "list_banks",
                                "description": "List all available memory banks with their statistics (entity count, relationship count, etc.)",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {}
                                }
                            },
                            {
                                "name": "delete_bank",
                                "description": "Delete a memory bank and all its contents. Cannot delete the 'default' bank.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "bank": {"type": "string", "description": "Name of the memory bank to delete"}
                                    },
                                    "required": ["bank"]
                                }
                            }
                        ]
                    }
                }
            elif request.get("method") == "notifications/initialized":
                # Handle initialized notification - no response needed
                continue  # Skip sending response for notifications
            elif request.get("method") == "prompts/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "prompts": []
                    }
                }
            elif request.get("method") == "tools/call":
                # Handle tool calls
                tool_name = request.get("params", {}).get("name")
                arguments = request.get("params", {}).get("arguments", {})
                
                if tool_name == "create_entities":
                    entities = arguments.get("entities", [])
                    auto_extract = arguments.get("auto_extract", True)  # Default to True for smart extraction
                    created_entities = []
                    observations_added = 0
                    extracted_entities = {}
                    extracted_relationships = []
                    extracted_count = 0
                    
                    try:
                        # First pass: Create the explicitly specified entities
                        for entity_data in entities:
                            node = Node(
                                id=entity_data["name"],
                                data={
                                    "type": entity_data["entityType"]
                                }
                            )
                            memory_banks[current_bank]["nodes"][node.id] = node
                            created_entities.append(node.model_dump())
                            
                            # Add observations separately to the observations collection
                            observations = entity_data.get("observations", [])
                            all_observation_text = " ".join(observations)
                            
                            for obs_content in observations:
                                obs = Observation(
                                    id=f"obs-{len(memory_banks[current_bank]['observations'])}",
                                    entity_id=entity_data["name"],
                                    content=obs_content,
                                    timestamp=str(time.time())
                                )
                                memory_banks[current_bank]["observations"].append(obs)
                                observations_added += 1
                            
                            # Extract additional entities and relationships from observations
                            if auto_extract and all_observation_text.strip():
                                try:
                                    # Extract entities from observation text
                                    obs_entities = extract_advanced_entities(all_observation_text)
                                    for entity_name, entity_info in obs_entities.items():
                                        if entity_name not in memory_banks[current_bank]["nodes"] and entity_name != entity_data["name"]:
                                            extracted_entities[entity_name] = entity_info
                                    
                                    # Extract relationships involving this entity
                                    all_entities = {entity_data["name"]: {"type": entity_data["entityType"], "confidence": 1.0}}
                                    all_entities.update(obs_entities)
                                    obs_relationships = extract_relationships(all_observation_text, all_entities)
                                    extracted_relationships.extend(obs_relationships)
                                except Exception as e:
                                    logger.error(f"Error during auto-extraction: {e}")
                        
                        # Second pass: Create extracted entities
                        if auto_extract:
                            for entity_name, entity_info in extracted_entities.items():
                                node = Node(
                                    id=entity_name,
                                    data={
                                        "type": entity_info["type"],
                                        "confidence": entity_info["confidence"],
                                        "auto_extracted": True
                                    }
                                )
                                memory_banks[current_bank]["nodes"][node.id] = node
                                extracted_count += 1
                            
                            # Create extracted relationships
                            for rel in extracted_relationships:
                                if (rel["from"] in memory_banks[current_bank]["nodes"] and 
                                    rel["to"] in memory_banks[current_bank]["nodes"]):
                                    edge = Edge(
                                        source=rel["from"],
                                        target=rel["to"],
                                        data={
                                            "type": rel["type"],
                                            "confidence": rel.get("confidence", 0.7),
                                            "auto_extracted": True
                                        }
                                    )
                                    edge.id = f"{edge.source}-{edge.target}-{len(memory_banks[current_bank]['edges'])}"
                                    memory_banks[current_bank]["edges"].append(edge)
                        
                        save_memory_banks()
                        
                        result_text = f"Created {len(created_entities)} entities: {[e['id'] for e in created_entities]}"
                        if auto_extract and extracted_count > 0:
                            result_text += f"\nAuto-extracted {extracted_count} additional entities and {len(extracted_relationships)} relationships from observations"
                        
                    except Exception as e:
                        logger.error(f"Error in create_entities: {e}")
                        result_text = f"Created {len(created_entities)} entities (with errors during auto-extraction)"
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": result_text
                                }
                            ]
                        }
                    }
                    
                elif tool_name == "add_observations":
                    observations = arguments.get("observations", [])
                    auto_extract = arguments.get("auto_extract", True)  # Default to True for smart extraction
                    added_count = 0
                    extracted_entities = {}
                    extracted_relationships = []
                    all_observation_texts = []
                    extracted_count = 0
                    
                    try:
                        # First pass: Add observations and collect text for extraction
                        for obs_data in observations:
                            entity_name = obs_data["entityName"]
                            contents = obs_data.get("contents", [])
                            if entity_name in memory_banks[current_bank]["nodes"]:
                                for content in contents:
                                    obs = Observation(
                                        id=f"obs-{len(memory_banks[current_bank]['observations'])}",
                                        entity_id=entity_name,
                                        content=content,
                                        timestamp=str(time.time())
                                    )
                                    memory_banks[current_bank]["observations"].append(obs)
                                    added_count += 1
                                    
                                    # Collect text for knowledge extraction
                                    if auto_extract:
                                        all_observation_texts.append(content)
                        
                        # Second pass: Extract knowledge from all observation text
                        if auto_extract and all_observation_texts:
                            try:
                                combined_text = " ".join(all_observation_texts)
                                
                                # Extract entities from observation text
                                obs_entities = extract_advanced_entities(combined_text)
                                for entity_name, entity_info in obs_entities.items():
                                    if entity_name not in memory_banks[current_bank]["nodes"]:
                                        extracted_entities[entity_name] = entity_info
                                
                                # Create extracted entities
                                for entity_name, entity_info in extracted_entities.items():
                                    node = Node(
                                        id=entity_name,
                                        data={
                                            "type": entity_info["type"],
                                            "confidence": entity_info["confidence"],
                                            "auto_extracted": True
                                        }
                                    )
                                    memory_banks[current_bank]["nodes"][node.id] = node
                                    extracted_count += 1
                                
                                # Extract relationships involving all entities (existing + extracted)
                                all_entities = {}
                                for entity_id in memory_banks[current_bank]["nodes"]:
                                    node = memory_banks[current_bank]["nodes"][entity_id]
                                    all_entities[entity_id] = {
                                        "type": node.data.get("type", "unknown"),
                                        "confidence": node.data.get("confidence", 1.0)
                                    }
                                
                                obs_relationships = extract_relationships(combined_text, all_entities)
                                
                                # Create extracted relationships
                                for rel in obs_relationships:
                                    if (rel["from"] in memory_banks[current_bank]["nodes"] and 
                                        rel["to"] in memory_banks[current_bank]["nodes"]):
                                        edge = Edge(
                                            source=rel["from"],
                                            target=rel["to"],
                                            data={
                                                "type": rel["type"],
                                                "confidence": rel.get("confidence", 0.7),
                                                "auto_extracted": True
                                            }
                                        )
                                        edge.id = f"{edge.source}-{edge.target}-{len(memory_banks[current_bank]['edges'])}"
                                        memory_banks[current_bank]["edges"].append(edge)
                                        extracted_relationships.append(rel)
                            except Exception as e:
                                logger.error(f"Error during auto-extraction in add_observations: {e}")
                        
                        save_memory_banks()
                        
                        result_text = f"Added {added_count} observations"
                        if auto_extract and extracted_count > 0:
                            result_text += f"\nAuto-extracted {extracted_count} additional entities and {len(extracted_relationships)} relationships from observations"
                        
                    except Exception as e:
                        logger.error(f"Error in add_observations: {e}")
                        result_text = f"Added {added_count} observations (with errors during auto-extraction)"
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": result_text
                                }
                            ]
                        }
                    }
                    
                elif tool_name == "create_relations":
                    relations = arguments.get("relations", [])
                    created_relations = []
                    for rel_data in relations:
                        edge = Edge(
                            source=rel_data["from"],
                            target=rel_data["to"],
                            data={"type": rel_data["relationType"]}
                        )
                        edge.id = f"{edge.source}-{edge.target}-{len(memory_banks[current_bank]['edges'])}"
                        memory_banks[current_bank]["edges"].append(edge)
                        created_relations.append(edge.dict())
                    save_memory_banks()
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Created {len(created_relations)} relations"
                                }
                            ]
                        }
                    }
                    
                elif tool_name == "sequential_thinking":
                    step_data = arguments.get("step", {})
                    step = ReasoningStep(
                        id=f"step-{len(memory_banks[current_bank]['reasoning_steps'])}",
                        description=step_data.get("thought", ""),
                        status="completed",
                        timestamp=str(time.time())
                    )
                    memory_banks[current_bank]["reasoning_steps"].append(step)
                    save_memory_banks()
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Added reasoning step: {step.description}"
                                }
                            ]
                        }
                    }
                    
                elif tool_name == "ingest_knowledge":
                    # Create KnowledgeIngest object from arguments
                    text = arguments.get("text", "")
                    bank = arguments.get("bank", current_bank)
                    source = arguments.get("source", "text_input")
                    extract_entities = arguments.get("extract_entities", True)
                    should_extract_relationships = arguments.get("extract_relationships", True)
                    create_observations = arguments.get("create_observations", True)
                    
                    # Process the knowledge ingestion
                    entities_created = 0
                    relationships_created = 0
                    observations_created = 0
                    
                    if extract_entities:
                        extracted_entities = extract_advanced_entities(text)
                        
                        for entity_name, entity_info in extracted_entities.items():
                            entity_id = entity_name.replace(" ", "_").lower()
                            
                            if entity_id not in memory_banks[bank]["nodes"]:
                                node = Node(
                                    id=entity_id,
                                    data={
                                        "name": entity_name,
                                        "type": entity_info["type"],
                                        "confidence": entity_info["confidence"],
                                        "source": source,
                                        "extracted_from": "text_analysis",
                                        "created_at": datetime.now().isoformat()
                                    }
                                )
                                memory_banks[bank]["nodes"][entity_id] = node
                                entities_created += 1
                            
                            if create_observations:
                                pattern = rf'(.{{0,50}}\b{re.escape(entity_name)}\b.{{0,50}})'
                                contexts = re.findall(pattern, text, re.IGNORECASE)
                                
                                for context in contexts[:2]:  # Limit to 2 contexts per entity
                                    obs = Observation(
                                        id=str(uuid.uuid4()),
                                        entity_id=entity_id,
                                        content=f"Found in context: \"{context.strip()}\"",
                                        timestamp=datetime.now().isoformat()
                                    )
                                    memory_banks[bank]["observations"].append(obs)
                                    observations_created += 1
                        
                        if should_extract_relationships and extracted_entities:
                            relationships = extract_relationships(text, extracted_entities)
                            
                            for rel in relationships:
                                from_id = rel["from"].replace(" ", "_").lower()
                                to_id = rel["to"].replace(" ", "_").lower()
                                
                                if from_id in memory_banks[bank]["nodes"] and to_id in memory_banks[bank]["nodes"]:
                                    edge = Edge(
                                        source=from_id,
                                        target=to_id,
                                        data={
                                            "type": rel["type"],
                                            "context": rel["context"],
                                            "confidence": rel["confidence"],
                                            "source": source,
                                            "extracted_from": "text_analysis",
                                            "created_at": datetime.now().isoformat()
                                        }
                                    )
                                    edge.id = f"{from_id}-{rel['type']}-{to_id}-{len(memory_banks[bank]['edges'])}"
                                    memory_banks[bank]["edges"].append(edge)
                                    relationships_created += 1
                    
                    save_memory_banks()
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Knowledge graph created: {entities_created} entities, {relationships_created} relationships, {observations_created} observations from {len(text.split())} words of text"
                                }
                            ]
                        }
                    }
                
                elif tool_name == "search_nodes":
                    query = arguments.get("query", "")
                    bank = arguments.get("bank")
                    entity_type = arguments.get("entity_type")
                    case_sensitive = arguments.get("case_sensitive", False)
                    use_regex = arguments.get("use_regex", False)
                    limit = arguments.get("limit", 50)
                    
                    # Handle special case of listing all entities
                    if query == "*" or query == "":
                        b = bank or current_bank
                        entity_ids = list(memory_banks[b]["nodes"].keys())
                        results = [{"entity_id": eid} for eid in entity_ids]
                    else:
                        results = search_entities(query, bank, entity_type, case_sensitive, use_regex)
                    
                    if limit:
                        results = results[:limit]
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Found {len(results)} entities matching '{query}': " + 
                                           ", ".join([r["entity_id"] for r in results[:10]]) +
                                           (f" (and {len(results)-10} more)" if len(results) > 10 else "")
                                }
                            ]
                        }
                    }
                
                elif tool_name == "search_relations":
                    query = arguments.get("query", "")
                    bank = arguments.get("bank")
                    relationship_type = arguments.get("relationship_type")
                    case_sensitive = arguments.get("case_sensitive", False)
                    use_regex = arguments.get("use_regex", False)
                    limit = arguments.get("limit", 50)
                    
                    results = search_relationships(query, bank, relationship_type, case_sensitive, use_regex)
                    
                    if limit:
                        results = results[:limit]
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Found {len(results)} relationships matching '{query}': " +
                                           ", ".join([f"{r['from_entity']} -> {r['to_entity']}" for r in results[:5]]) +
                                           (f" (and {len(results)-5} more)" if len(results) > 5 else "")
                                }
                            ]
                        }
                    }
                
                elif tool_name == "search_observations":
                    query = arguments.get("query", "")
                    bank = arguments.get("bank")
                    entity_id = arguments.get("entity_id")
                    case_sensitive = arguments.get("case_sensitive", False)
                    use_regex = arguments.get("use_regex", False)
                    limit = arguments.get("limit", 50)
                    
                    results = search_observations(query, bank, entity_id, case_sensitive, use_regex)
                    
                    if limit:
                        results = results[:limit]
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Found {len(results)} observations matching '{query}': " +
                                           ", ".join([f"{r['entity_id']}: {r['content'][:50]}..." for r in results[:3]]) +
                                           (f" (and {len(results)-3} more)" if len(results) > 3 else "")
                                }
                            ]
                        }
                    }
                
                elif tool_name == "search_all":
                    query = arguments.get("query", "")
                    bank = arguments.get("bank")
                    case_sensitive = arguments.get("case_sensitive", False)
                    use_regex = arguments.get("use_regex", False)
                    limit = arguments.get("limit", 50)
                    
                    entities = search_entities(query, bank, None, case_sensitive, use_regex)
                    relationships = search_relationships(query, bank, None, case_sensitive, use_regex)
                    observations = search_observations(query, bank, None, case_sensitive, use_regex)
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Search results for '{query}':\n" +
                                           f"- {len(entities)} entities\n" +
                                           f"- {len(relationships)} relationships\n" +
                                           f"- {len(observations)} observations\n\n" +
                                           f"Top entities: {', '.join([e['entity_id'] for e in entities[:5]])}\n" +
                                           "Top relationships: " + ', '.join([f"{r['from_entity']}->{r['to_entity']}" for r in relationships[:3]])
                                }
                            ]
                        }
                    }
                
                elif tool_name == "delete_entities":
                    entity_names = arguments.get("entityNames", [])
                    deleted_count = 0
                    
                    # Handle special case of deleting all entities
                    if entity_names == ["ALL"] or not entity_names:
                        deleted_count = len(memory_banks[current_bank]["nodes"])
                        memory_banks[current_bank]["nodes"].clear()
                        memory_banks[current_bank]["edges"].clear()
                        memory_banks[current_bank]["observations"].clear()
                    else:
                        for entity_name in entity_names:
                            if entity_name in memory_banks[current_bank]["nodes"]:
                                # Remove the entity
                                del memory_banks[current_bank]["nodes"][entity_name]
                                deleted_count += 1
                                
                                # Remove related edges
                                memory_banks[current_bank]["edges"] = [
                                    edge for edge in memory_banks[current_bank]["edges"]
                                    if edge.source != entity_name and edge.target != entity_name
                                ]
                                
                                # Remove related observations
                                memory_banks[current_bank]["observations"] = [
                                    obs for obs in memory_banks[current_bank]["observations"]
                                    if obs.entity_id != entity_name
                                ]
                    
                    save_memory_banks()
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Deleted {deleted_count} entities and their associated relations and observations"
                                }
                            ]
                        }
                    }
                
                elif tool_name == "delete_relations":
                    relations = arguments.get("relations", [])
                    deleted_count = 0
                    
                    for rel_data in relations:
                        from_entity = rel_data["from"]
                        to_entity = rel_data["to"]
                        relation_type = rel_data["relationType"]
                        
                        # Find and remove matching edges
                        original_count = len(memory_banks[current_bank]["edges"])
                        memory_banks[current_bank]["edges"] = [
                            edge for edge in memory_banks[current_bank]["edges"]
                            if not (edge.source == from_entity and 
                                   edge.target == to_entity and 
                                   edge.data.get("type") == relation_type)
                        ]
                        deleted_count += original_count - len(memory_banks[current_bank]["edges"])
                    
                    save_memory_banks()
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Deleted {deleted_count} relations"
                                }
                            ]
                        }
                    }
                
                elif tool_name == "delete_observations":
                    deletions = arguments.get("deletions", [])
                    total_deleted = 0
                    
                    for deletion in deletions:
                        entity_name = deletion["entityName"]
                        observations_to_delete = deletion["observations"]
                        
                        # Remove specified observations
                        original_count = len(memory_banks[current_bank]["observations"])
                        memory_banks[current_bank]["observations"] = [
                            obs for obs in memory_banks[current_bank]["observations"]
                            if not (obs.entity_id == entity_name and obs.content in observations_to_delete)
                        ]
                        total_deleted += original_count - len(memory_banks[current_bank]["observations"])
                    
                    save_memory_banks()
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Deleted {total_deleted} observations"
                                }
                            ]
                        }
                    }
                
                elif tool_name == "read_graph":
                    bank = arguments.get("bank", current_bank)
                    if bank not in memory_banks:
                        bank = current_bank
                    
                    nodes = memory_banks[bank]["nodes"]
                    edges = memory_banks[bank]["edges"]
                    observations = memory_banks[bank]["observations"]
                    
                    graph_summary = {
                        "bank": bank,
                        "entities": len(nodes),
                        "relationships": len(edges),
                        "observations": len(observations),
                        "entity_types": list(set(node.data.get("type", "unknown") for node in nodes.values())),
                        "relation_types": list(set(edge.data.get("type", "unknown") for edge in edges))
                    }
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Knowledge Graph Summary for bank '{bank}':\n" +
                                           f"- {graph_summary['entities']} entities\n" +
                                           f"- {graph_summary['relationships']} relationships\n" +
                                           f"- {graph_summary['observations']} observations\n" +
                                           f"- Entity types: {', '.join(graph_summary['entity_types'])}\n" +
                                           f"- Relation types: {', '.join(graph_summary['relation_types'])}"
                                }
                            ]
                        }
                    }
                
                elif tool_name == "open_nodes":
                    names = arguments.get("names", [])
                    found_nodes = []
                    
                    # Handle special case of opening all nodes
                    if names == ["ALL"] or not names:
                        entity_names = list(memory_banks[current_bank]["nodes"].keys())
                    else:
                        entity_names = names
                    
                    for name in entity_names:
                        if name in memory_banks[current_bank]["nodes"]:
                            node = memory_banks[current_bank]["nodes"][name]
                            # Get related observations
                            node_observations = [
                                obs.content for obs in memory_banks[current_bank]["observations"]
                                if obs.entity_id == name
                            ]
                            # Get related relationships
                            node_relations = [
                                {"from": edge.source, "to": edge.target, "type": edge.data.get("type", "unknown")}
                                for edge in memory_banks[current_bank]["edges"]
                                if edge.source == name or edge.target == name
                            ]
                            
                            found_nodes.append({
                                "name": name,
                                "type": node.data.get("type", "unknown"),
                                "observations": node_observations,
                                "relationships": node_relations
                            })
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Retrieved {len(found_nodes)} nodes:\n" +
                                           "\n".join([
                                               f"- {node['name']} ({node['type']}): {len(node['observations'])} observations, {len(node['relationships'])} relationships"
                                               for node in found_nodes
                                           ])
                                }
                            ]
                        }
                    }
                
                elif tool_name == "create_bank":
                    bank_name = arguments.get("bank", "")
                    if not bank_name:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "error": {
                                "code": -32602,
                                "message": "Bank name is required"
                            }
                        }
                    elif bank_name in memory_banks:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Bank '{bank_name}' already exists"
                                    }
                                ]
                            }
                        }
                    else:
                        memory_banks[bank_name] = {"nodes": {}, "edges": [], "observations": [], "reasoning_steps": []}
                        save_memory_banks()
                        response = {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Successfully created bank '{bank_name}'"
                                    }
                                ]
                            }
                        }
                
                elif tool_name == "select_bank":
                    bank_name = arguments.get("bank", "")
                    if not bank_name:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "error": {
                                "code": -32602,
                                "message": "Bank name is required"
                            }
                        }
                    elif bank_name not in memory_banks:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "error": {
                                "code": -32602,
                                "message": f"Bank '{bank_name}' does not exist"
                            }
                        }
                    else:
                        current_bank = bank_name
                        response = {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Successfully switched to bank '{bank_name}'"
                                    }
                                ]
                            }
                        }
                
                elif tool_name == "list_banks":
                    banks_info = []
                    for bank_name, bank_data in memory_banks.items():
                        bank_stats = {
                            "bank": bank_name,
                            "entities": len(bank_data["nodes"]),
                            "relationships": len(bank_data["edges"]),
                            "observations": len(bank_data["observations"]),
                            "is_current": bank_name == current_bank
                        }
                        banks_info.append(bank_stats)
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Available banks ({len(banks_info)} total):\n" +
                                           "\n".join([
                                               f"{'• [CURRENT] ' if bank['is_current'] else '• '}{bank['bank']}: {bank['entities']} entities, {bank['relationships']} relationships, {bank['observations']} observations"
                                               for bank in banks_info
                                           ])
                                }
                            ]
                        }
                    }
                
                elif tool_name == "delete_bank":
                    bank_name = arguments.get("bank", "")
                    if not bank_name:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "error": {
                                "code": -32602,
                                "message": "Bank name is required"
                            }
                        }
                    elif bank_name == "default":
                        response = {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "error": {
                                "code": -32602,
                                "message": "Cannot delete the default bank"
                            }
                        }
                    elif bank_name not in memory_banks:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "error": {
                                "code": -32602,
                                "message": f"Bank '{bank_name}' does not exist"
                            }
                        }
                    else:
                        del memory_banks[bank_name]
                        if current_bank == bank_name:
                            current_bank = "default"
                        save_memory_banks()
                        response = {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Successfully deleted bank '{bank_name}'"
                                    }
                                ]
                            }
                        }
                
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "error": {
                            "code": -32601,
                            "message": f"Unknown tool: {tool_name}"
                        }
                    }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {
                        "code": -32601,
                        "message": "Method not found"
                    }
                }
            
            # Send response to stdout
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
