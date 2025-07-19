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
logging.basicConfig(level=logging.INFO)
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
    try:
        body = await request.body()
        logging.info(f"Request body: {body.decode('utf-8') if body else '<empty>'}")
    except Exception as e:
        logging.info(f"Could not read request body: {e}")
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
        serialized[bank_name] = {
            "nodes": {node_id: node.model_dump() for node_id, node in bank_data["nodes"].items()},
            "edges": [edge.dict() for edge in bank_data["edges"]],
            "observations": [obs.dict() for obs in bank_data["observations"]],
            "reasoning_steps": [step.dict() for step in bank_data["reasoning_steps"]]
        }
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
        with open(MEMORY_FILE, 'w') as f:
            json.dump(serialized_data, f, indent=2)
        logger.info(f"Memory banks saved to {MEMORY_FILE}")
    except Exception as e:
        logger.error(f"Failed to save memory banks: {e}")

def load_memory_banks():
    """Load memory banks from persistent storage"""
    global memory_banks, current_bank
    try:
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
            deserialize_memory_banks(data)
            logger.info(f"Memory banks loaded from {MEMORY_FILE}")
            logger.info(f"Loaded banks: {list(memory_banks.keys())}")
        else:
            logger.info("No existing memory file found, starting with default banks")
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
    memory_banks[op.bank] = {"nodes": {}, "edges": [], "observations": [], "reasoning_steps": []}
    save_memory_banks()  # Persist the change
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

def extract_advanced_entities(text: str):
    """Enhanced entity extraction with multiple patterns and types"""
    entities = {}
    
    # 1. Named entities (capitalized sequences)
    named_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)*\b', text)
    for entity in named_entities:
        if len(entity) > 2:  # Filter out short words
            entities[entity] = {"type": "named_entity", "confidence": 0.8}
    
    # 2. Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b', text)  # camelCase
    for term in technical_terms:
        entities[term] = {"type": "technical_term", "confidence": 0.7}
    
    # 3. Quoted concepts
    quoted_concepts = re.findall(r'"([^"]*)"', text)
    for concept in quoted_concepts:
        if len(concept.strip()) > 2:
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
    """Extract relationships between entities with context"""
    relationships = []
    
    # Split text into sentences for relationship extraction
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # Skip very short sentences
            continue
            
        # Find entities in this sentence
        sentence_entities = []
        for entity in entities.keys():
            if entity.lower() in sentence.lower():
                sentence_entities.append(entity)
        
        # Extract relationships between entities in the same sentence
        for i, entity1 in enumerate(sentence_entities):
            for entity2 in sentence_entities[i+1:]:
                # Look for action words between entities
                pattern = rf'\b{re.escape(entity1.lower())}.*?\b(\w+(?:s|ed|ing)?)\b.*?{re.escape(entity2.lower())}'
                matches = re.findall(pattern, sentence.lower())
                
                if matches:
                    for action in matches:
                        if len(action) > 2:  # Filter short words
                            relationships.append({
                                "from": entity1,
                                "to": entity2,
                                "type": action,
                                "context": sentence[:100] + "..." if len(sentence) > 100 else sentence,
                                "confidence": 0.6
                            })
                else:
                    # Default relationship for co-occurrence
                    relationships.append({
                        "from": entity1,
                        "to": entity2,
                        "type": "related_to",
                        "context": sentence[:100] + "..." if len(sentence) > 100 else sentence,
                        "confidence": 0.4
                    })
    
    return relationships

@app.post("/knowledge/ingest")
def ingest_knowledge_graph(payload: KnowledgeIngest = Body(...)):
    """
    Advanced knowledge graph creation from large text with sophisticated entity and relationship extraction.
    
    Request body: {
        "text": "Large text content...",
        "bank": "bank_name",
        "source": "document_name",
        "extract_entities": true,
        "extract_relationships": true,
        "create_observations": true
    }
    
    Response: {
        "status": "success",
        "entities_created": 15,
        "relationships_created": 8,
        "observations_created": 23,
        "processing_stats": {...},
        "bank": "bank_name"
    }
    """
    text = payload.text
    b = payload.bank or current_bank
    source = payload.source
    
    processing_stats = {
        "text_length": len(text),
        "sentences": len(re.split(r'[.!?]+', text)),
        "words": len(text.split()),
        "processing_time": datetime.now().isoformat()
    }
    
    entities_created = 0
    relationships_created = 0
    observations_created = 0
    
    # Extract entities with advanced patterns
    if payload.extract_entities:
        entities = extract_advanced_entities(text)
        
        for entity_name, entity_info in entities.items():
            entity_id = entity_name.replace(" ", "_").lower()
            
            # Create or update entity
            if entity_id not in memory_banks[b]["nodes"]:
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
                memory_banks[b]["nodes"][entity_id] = node
                entities_created += 1
            
            # Create observation with source text context
            if payload.create_observations:
                # Find context around the entity in the text
                pattern = rf'(.{{0,50}}\b{re.escape(entity_name)}\b.{{0,50}})'
                contexts = re.findall(pattern, text, re.IGNORECASE)
                
                for context in contexts[:3]:  # Limit to 3 contexts per entity
                    obs = Observation(
                        id=str(uuid.uuid4()),
                        entity_id=entity_id,
                        content=f"Found in context: \"{context.strip()}\"",
                        timestamp=datetime.now().isoformat()
                    )
                    memory_banks[b]["observations"].append(obs)
                    observations_created += 1
        
        # Extract relationships
        if payload.extract_relationships and entities:
            relationships = extract_relationships(text, entities)
            
            for rel in relationships:
                from_id = rel["from"].replace(" ", "_").lower()
                to_id = rel["to"].replace(" ", "_").lower()
                
                # Only create relationship if both entities exist
                if from_id in memory_banks[b]["nodes"] and to_id in memory_banks[b]["nodes"]:
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
                    edge.id = f"{from_id}-{rel['type']}-{to_id}-{len(memory_banks[b]['edges'])}"
                    memory_banks[b]["edges"].append(edge)
                    relationships_created += 1
    
    # Save all changes
    save_memory_banks()
    
    return {
        "status": "success",
        "entities_created": entities_created,
        "relationships_created": relationships_created,
        "observations_created": observations_created,
        "processing_stats": processing_stats,
        "bank": b
    }


@app.get("/context/retrieve")
def retrieve_context(bank: str = Query(None)):
    """
    Retrieve all context (entities, relations, observations, reasoning steps) from the selected or specified bank.
    Query param: bank (optional)
    Response: {"entities": [...], "relations": [...], "observations": [...], "reasoning_steps": [...]}
    """
    b = bank or current_bank
    return {
        "entities": [n.dict() for n in memory_banks[b]["nodes"].values()],
        "relations": [e.dict() for e in memory_banks[b]["edges"]],
        "observations": [o.dict() for o in memory_banks[b]["observations"]],
        "reasoning_steps": [s.dict() for s in memory_banks[b]["reasoning_steps"]]
    }

# Graph visualization endpoints

@app.get("/banks/{bank}/graph-data")
def get_graph_data(bank: str):
    """
    Get graph data in vis.js network format for visualization.
    Returns nodes and edges formatted for interactive graph visualization.
    """
    if bank not in memory_banks:
        return {"error": "Bank not found"}
    
    nodes = []
    edges = []
    
    # Convert entities to vis.js nodes
    for node_id, node in memory_banks[bank]["nodes"].items():
        # Determine node styling based on entity type
        entity_type = node.data.get("type", "node")
        confidence = node.data.get("confidence", 0.5)
        
        # Color coding for different entity types
        color_map = {
            "named_entity": "#4A90E2",  # Blue
            "technical_term": "#7ED321", # Green  
            "concept": "#9013FE",       # Purple
            "email": "#FF6B35",         # Orange
            "url": "#FF6B35",           # Orange
            "measurement": "#F5A623",   # Yellow
            "date": "#50E3C2",          # Teal
            "node": "#B8E986"           # Light green (default)
        }
        
        # Shape coding
        shape_map = {
            "named_entity": "dot",
            "technical_term": "square", 
            "concept": "diamond",
            "email": "triangle",
            "url": "triangle",
            "measurement": "box",
            "date": "ellipse",
            "node": "dot"
        }
        
        node_data = {
            "id": node_id,
            "label": node.data.get("name", node_id),
            "title": f"Type: {entity_type}<br/>Confidence: {confidence:.2f}<br/>Source: {node.data.get('source', 'N/A')}",
            "color": {
                "background": color_map.get(entity_type, "#B8E986"),
                "border": "#2B7CE9",
                "highlight": {"background": "#FFD700", "border": "#FFA500"}
            },
            "shape": shape_map.get(entity_type, "dot"),
            "size": 10 + (confidence * 20),  # Size based on confidence
            "font": {"size": 12 + (confidence * 8)},
            "metadata": node.data
        }
        nodes.append(node_data)
    
    # Convert relationships to vis.js edges  
    for edge in memory_banks[bank]["edges"]:
        relationship_type = edge.data.get("type", "relation")
        confidence = edge.data.get("confidence", 0.5)
        
        # Color coding for relationship types
        edge_color_map = {
            "created": "#E74C3C",       # Red
            "developed": "#E67E22",     # Orange
            "leads": "#3498DB",         # Blue
            "known": "#9B59B6",        # Purple
            "like": "#1ABC9C",         # Teal
            "work": "#F39C12",         # Yellow
            "related_to": "#95A5A6",   # Gray
            "relation": "#BDC3C7"      # Light gray (default)
        }
        
        edge_data = {
            "id": edge.id,
            "from": edge.source,
            "to": edge.target,
            "label": relationship_type,
            "title": f"Type: {relationship_type}<br/>Confidence: {confidence:.2f}<br/>Context: {edge.data.get('context', 'N/A')[:100]}...",
            "color": {
                "color": edge_color_map.get(relationship_type, "#BDC3C7"),
                "highlight": "#FFD700"
            },
            "width": 1 + (confidence * 4),  # Width based on confidence
            "arrows": {"to": {"enabled": True, "scaleFactor": 1}},
            "smooth": {"type": "continuous"},
            "metadata": edge.data
        }
        edges.append(edge_data)
    
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": len(set(node.data.get("type", "node") for node in memory_banks[bank]["nodes"].values())),
            "relationship_types": len(set(edge.data.get("type", "relation") for edge in memory_banks[bank]["edges"]))
        }
    }

@app.get("/banks/{bank}/visualize")
def visualize_graph(bank: str):
    """
    Serve interactive graph visualization page for a specific memory bank.
    """
    if bank not in memory_banks:
        return JSONResponse(content={"error": "Bank not found"}, status_code=404)
    
    # HTML template with vis.js network visualization
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Knowledge Graph Visualization - Bank: {bank}</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .controls {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 10px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .control-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        #mynetworkid {{
            width: 100%;
            height: 600px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .info-panel {{
            margin-top: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 1px solid #ccc;
        }}
        button {{
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background: #4A90E2;
            color: white;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{
            background: #357ABD;
        }}
        select, input {{
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .stats {{
            background: #e8f4fd;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Knowledge Graph Visualization</h1>
        <h2>Memory Bank: <span style="color: #4A90E2;">{bank}</span></h2>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <label>Search:</label>
            <input type="text" id="searchInput" placeholder="Search entities..." onkeyup="searchNodes()">
        </div>
        <div class="control-group">
            <label>Layout:</label>
            <select id="layoutSelect" onchange="changeLayout()">
                <option value="physics">Force-Directed</option>
                <option value="hierarchical">Hierarchical</option>
                <option value="random">Random</option>
            </select>
        </div>
        <div class="control-group">
            <button onclick="fitNetwork()">Fit to Screen</button>
            <button onclick="exportNetwork()">Export PNG</button>
            <button onclick="refreshData()">Refresh</button>
        </div>
    </div>
    
    <div id="mynetworkid"></div>
    
    <div class="info-panel">
        <div id="networkStats" class="stats"></div>
        <div id="selectedInfo"></div>
        
        <h3>Legend - Entity Types</h3>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #4A90E2;"></div>
                <span>Named Entity</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #7ED321;"></div>
                <span>Technical Term</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #9013FE;"></div>
                <span>Concept</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #FF6B35;"></div>
                <span>Contact Info</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #F5A623;"></div>
                <span>Measurement</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #50E3C2;"></div>
                <span>Date</span>
            </div>
        </div>
        
        <h3>Legend - Relationship Types</h3>
        <div class="legend">
            <div class="legend-item">
                <div style="width: 20px; height: 3px; background: #E74C3C;"></div>
                <span>Created</span>
            </div>
            <div class="legend-item">
                <div style="width: 20px; height: 3px; background: #9B59B6;"></div>
                <span>Known As</span>
            </div>
            <div class="legend-item">
                <div style="width: 20px; height: 3px; background: #3498DB;"></div>
                <span>Leads To</span>
            </div>
            <div class="legend-item">
                <div style="width: 20px; height: 3px; background: #95A5A6;"></div>
                <span>Related</span>
            </div>
        </div>
    </div>

    <script type="text/javascript">
        let network;
        let nodes, edges;
        let allNodes, allEdges;
        
        // Initialize the network
        async function initNetwork() {{
            try {{
                const response = await fetch('/banks/{bank}/graph-data');
                const data = await response.json();
                
                if (data.error) {{
                    document.getElementById('selectedInfo').innerHTML = `<p style="color: red;">Error: ${{data.error}}</p>`;
                    return;
                }}
                
                allNodes = new vis.DataSet(data.nodes);
                allEdges = new vis.DataSet(data.edges);
                nodes = allNodes;
                edges = allEdges;
                
                const container = document.getElementById('mynetworkid');
                const graphData = {{ nodes: nodes, edges: edges }};
                
                const options = {{
                    physics: {{
                        enabled: true,
                        stabilization: {{ iterations: 200 }},
                        barnesHut: {{ gravitationalConstant: -80000, springConstant: 0.001, springLength: 200 }}
                    }},
                    interaction: {{
                        hover: true,
                        selectConnectedEdges: false
                    }},
                    nodes: {{
                        borderWidth: 2,
                        shadow: true,
                        font: {{ color: '#343434' }}
                    }},
                    edges: {{
                        shadow: true,
                        smooth: true,
                        font: {{ color: '#343434', size: 10 }}
                    }}
                }};
                
                network = new vis.Network(container, graphData, options);
                
                // Event listeners
                network.on("selectNode", function (params) {{
                    if (params.nodes.length > 0) {{
                        const nodeId = params.nodes[0];
                        showNodeInfo(nodeId);
                    }}
                }});
                
                network.on("deselectNode", function () {{
                    document.getElementById('selectedInfo').innerHTML = '<p>Click on a node to see details</p>';
                }});
                
                // Update stats
                updateStats(data.stats);
                
            }} catch (error) {{
                console.error('Error loading graph data:', error);
                document.getElementById('selectedInfo').innerHTML = `<p style="color: red;">Error loading graph: ${{error.message}}</p>`;
            }}
        }}
        
        function showNodeInfo(nodeId) {{
            const node = allNodes.get(nodeId);
            if (node) {{
                const metadata = node.metadata || {{}};
                const info = `
                    <h4>📍 Selected Node: ${{node.label}}</h4>
                    <p><strong>ID:</strong> ${{nodeId}}</p>
                    <p><strong>Type:</strong> ${{metadata.type || 'Unknown'}}</p>
                    <p><strong>Confidence:</strong> ${{(metadata.confidence || 0).toFixed(2)}}</p>
                    <p><strong>Source:</strong> ${{metadata.source || 'N/A'}}</p>
                    <p><strong>Created:</strong> ${{metadata.created_at || 'N/A'}}</p>
                    ${{metadata.extracted_from ? `<p><strong>Extracted From:</strong> ${{metadata.extracted_from}}</p>` : ''}}
                `;
                document.getElementById('selectedInfo').innerHTML = info;
            }}
        }}
        
        function updateStats(stats) {{
            document.getElementById('networkStats').innerHTML = `
                <strong>📊 Graph Statistics:</strong>
                ${{stats.total_nodes}} entities, ${{stats.total_edges}} relationships, 
                ${{stats.entity_types}} entity types, ${{stats.relationship_types}} relationship types
            `;
        }}
        
        function searchNodes() {{
            const searchTerm = document.getElementById('searchInput').value.toLowerCase();
            if (searchTerm === '') {{
                // Reset all nodes to original style
                const updates = allNodes.map(node => ({{
                    id: node.id,
                    color: node.color
                }}));
                nodes.update(updates);
                return;
            }}
            
            // Highlight matching nodes
            const updates = allNodes.map(node => {{
                const matches = node.label.toLowerCase().includes(searchTerm) || 
                              node.id.toLowerCase().includes(searchTerm);
                return {{
                    id: node.id,
                    color: matches ? 
                        {{ background: '#FFD700', border: '#FFA500' }} : 
                        {{ background: '#E0E0E0', border: '#CCCCCC' }}
                }};
            }});
            nodes.update(updates);
        }}
        
        function changeLayout() {{
            const layout = document.getElementById('layoutSelect').value;
            let options = {{}};
            
            if (layout === 'physics') {{
                options = {{
                    physics: {{ enabled: true }},
                    layout: {{ randomSeed: 2 }}
                }};
            }} else if (layout === 'hierarchical') {{
                options = {{
                    physics: {{ enabled: false }},
                    layout: {{
                        hierarchical: {{
                            direction: 'UD',
                            sortMethod: 'directed'
                        }}
                    }}
                }};
            }} else if (layout === 'random') {{
                options = {{
                    physics: {{ enabled: false }},
                    layout: {{ randomSeed: Math.random() }}
                }};
            }}
            
            network.setOptions(options);
        }}
        
        function fitNetwork() {{
            network.fit();
        }}
        
        function exportNetwork() {{
            const canvas = document.querySelector('#mynetworkid canvas');
            if (canvas) {{
                const link = document.createElement('a');
                link.download = 'knowledge-graph-{bank}.png';
                link.href = canvas.toDataURL();
                link.click();
            }}
        }}
        
        function refreshData() {{
            initNetwork();
        }}
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', initNetwork);
    </script>
</body>
</html>
    """
    
    return StreamingResponse(
        iter([html_content]),
        media_type="text/html"
    )

@app.get("/visualizations")
def list_visualizations():
    """
    List available graph visualizations for all memory banks.
    """
    available_banks = list(memory_banks.keys())
    visualizations = []
    
    for bank in available_banks:
        bank_stats = {
            "entities": len(memory_banks[bank]["nodes"]),
            "relationships": len(memory_banks[bank]["edges"]),
            "observations": len(memory_banks[bank]["observations"])
        }
        
        visualizations.append({
            "bank": bank,
            "visualization_url": f"/banks/{bank}/visualize",
            "data_url": f"/banks/{bank}/graph-data",
            "stats": bank_stats
        })
    
    return {
        "available_visualizations": visualizations,
        "total_banks": len(available_banks)
    }

@app.get("/")
async def root(request: Request):
    """Root endpoint for compatibility - handles both JSON and SSE"""
    logger.info("Handling root endpoint")
    
    # Check if client expects SSE
    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        # Return SSE response for agent compatibility
        async def generate_sse():
            # Send initial data
            data = {
                "status": "success",
                "server": "Graph Memory MCP Server",
                "version": "1.0",
                "capabilities": [
                    "memory_management",
                    "sequential_thinking", 
                    "graph_operations",
                    "multi_bank_support"
                ]
            }
            yield f"data: {json.dumps(data)}\n\n"
            
            # Keep connection alive
            while True:
                yield f"data: {json.dumps({'heartbeat': True, 'timestamp': time.time()})}\n\n"
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
        
        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )
    
    # Regular JSON response
    return {"message": "Graph Memory MCP Server is running."}

@app.post("/")
async def root_post(request: Request):
    """
    Handle JSON-RPC 2.0 requests for MCP protocol.
    Accepts JSON-RPC initialize and other MCP method calls.
    """
    try:
        # Get the request body
        body = await request.body()
        if not body:
            return {"message": "Graph Memory MCP Server is running."}
        
        # Parse JSON-RPC request
        try:
            rpc_request = json.loads(body)
        except json.JSONDecodeError:
            return {"message": "Graph Memory MCP Server is running."}
        
        # Handle JSON-RPC 2.0 requests
        if isinstance(rpc_request, dict) and rpc_request.get("jsonrpc") == "2.0":
            method = rpc_request.get("method")
            request_id = rpc_request.get("id")
            params = rpc_request.get("params", {})
            
            logger.info(f"Handling JSON-RPC method: {method}")
            
            if method == "initialize":
                # Return proper MCP initialize response
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "serverInfo": {
                            "name": "Graph Memory MCP Server",
                            "version": "1.0"
                        },
                        "capabilities": {
                            "tools": {
                                "listChanged": True
                            },
                            "resources": {
                                "subscribe": False,
                                "listChanged": True
                            },
                            "roots": {
                                "listChanged": True
                            },
                            "prompts": {
                                "listChanged": False
                            },
                            "completion": {
                                "supports": ["text"]
                            }
                        }
                    }
                }
            elif method == "tools/list":
                # Return available tools
                return {
                    "jsonrpc": "2.0", 
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "create_entities",
                                "description": "Create multiple new entities in the knowledge graph",
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
                            }
                        ]
                    }
                }
            else:
                # Unknown method
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": "Method not found"
                    }
                }
        
        # Fallback for non-JSON-RPC requests
        return {"message": "Graph Memory MCP Server is running."}
        
    except Exception as e:
        logger.error(f"Error handling root POST: {e}")
        return {"message": "Graph Memory MCP Server is running."}

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
    logger.info("Starting MCP stdio mode")
    
    while True:
        try:
            # Read JSON-RPC request from stdin
            line = sys.stdin.readline()
            if not line:
                break
                
            request = json.loads(line.strip())
            logger.info(f"MCP stdio request: {request}")
            
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
                                "description": "Create multiple new entities in the knowledge graph",
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
                            }
                        ]
                    }
                }
            elif request.get("method") == "tools/call":
                # Handle tool calls
                tool_name = request.get("params", {}).get("name")
                arguments = request.get("params", {}).get("arguments", {})
                
                if tool_name == "create_entities":
                    entities = arguments.get("entities", [])
                    created_entities = []
                    for entity_data in entities:
                        node = Node(
                            id=entity_data["name"],
                            data={
                                "type": entity_data["entityType"],
                                "observations": entity_data.get("observations", [])
                            }
                        )
                        memory_banks[current_bank]["nodes"][node.id] = node
                        created_entities.append(node.model_dump())
                    save_memory_banks()
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Created {len(created_entities)} entities: {[e['id'] for e in created_entities]}"
                                }
                            ]
                        }
                    }
                    
                elif tool_name == "add_observations":
                    observations = arguments.get("observations", [])
                    added_count = 0
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
                    save_memory_banks()
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Added {added_count} observations"
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
                    extract_relationships = arguments.get("extract_relationships", True)
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
                        
                        if extract_relationships and extracted_entities:
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
    else:
        # Run as HTTP server (default)
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=10642)
