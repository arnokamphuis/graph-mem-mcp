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
            "nodes": {node_id: node.dict() for node_id, node in bank_data["nodes"].items()},
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
