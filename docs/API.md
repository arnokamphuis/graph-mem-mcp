# API Reference

Complete REST API documentation for the Graph Memory MCP Server with **Enhanced Knowledge Graph Creation and Interactive Visualization**.

## Base URL

```
http://localhost:10642
```

## Authentication

No authentication required for local deployment.

## 🎨 NEW: Knowledge Graph Visualization

### Interactive Visualization Interface

```http
GET /banks/{bank}/visualize
```

**Description:** Opens interactive web-based graph visualization with vis.js Network library.

**Features:**
- Color-coded entities (blue=named_entity, green=technical_term, purple=concept)
- Interactive zoom, pan, search, and filtering
- Multiple layout algorithms
- Export to PNG capability

**Example:**
```
http://localhost:10642/banks/default/visualize
```

### Graph Data for Visualization

```http
GET /banks/{bank}/graph-data
```

**Description:** Returns graph data formatted for vis.js Network visualization.

**Response:**
```json
{
  "nodes": [
    {
      "id": "entity_id",
      "label": "Entity Name",
      "title": "Type: named_entity<br/>Confidence: 0.80<br/>Source: document",
      "color": {"background": "#4A90E2", "border": "#2B7CE9"},
      "shape": "dot",
      "size": 26.0,
      "metadata": {...}
    }
  ],
  "edges": [
    {
      "id": "relationship_id",
      "from": "entity1",
      "to": "entity2", 
      "label": "relationship_type",
      "title": "Type: created<br/>Confidence: 0.60<br/>Context: ...",
      "color": {"color": "#E74C3C"},
      "width": 3.4,
      "metadata": {...}
    }
  ],
  "stats": {
    "total_nodes": 26,
    "total_edges": 24,
    "entity_types": 3,
    "relationship_types": 12
  }
}
```

### List Available Visualizations

```http
GET /visualizations
```

**Description:** Lists all available graph visualizations across memory banks.

**Response:**
```json
{
  "available_visualizations": [
    {
      "bank": "default",
      "visualization_url": "/banks/default/visualize",
      "data_url": "/banks/default/graph-data",
      "stats": {
        "entities": 26,
        "relationships": 24,
        "observations": 35
      }
    }
  ],
  "total_banks": 1
}
```

## 🧠 NEW: Advanced Knowledge Graph Creation

### Ingest Knowledge from Large Text

```http
POST /knowledge/ingest
```

**Description:** Transform large text into sophisticated knowledge graphs with intelligent entity extraction and relationship detection.

**Request Body:**
```json
{
  "text": "Large text content for analysis...",
  "source": "document_name",
  "bank_name": "default"
}
```

**Features:**
- **Entity Types:** named_entity, technical_term, concept, email, url, measurement, date
- **Relationship Detection:** Action (created, developed), categorical (like, known as), temporal (led to)
- **Confidence Scoring:** 0.4-0.9 based on extraction quality
- **Source Attribution:** Every element tagged with source document

**Response:**
```json
{
  "status": "success",
  "extracted": {
    "entities": 26,
    "relationships": 24,
    "observations": 35
  },
  "bank": "default",
  "source": "document_name"
}
```

## Memory Banks

### List Banks

```http
GET /banks/list
```

**Response:**
```json
{
  "banks": ["default", "project-alpha"],
  "current": "default"
}
```

### Create Bank

```http
POST /banks/create
```

**Request Body:**
```json
{
  "bank": "new-bank-name"
}
```

**Response:**
```json
{
  "status": "success",
  "bank": "new-bank-name"
}
```

### Select Bank

```http
POST /banks/select
```

**Request Body:**
```json
{
  "bank": "existing-bank-name"
}
```

**Response:**
```json
{
  "status": "success",
  "selected": "existing-bank-name"
}
```

### Delete Bank

```http
POST /banks/delete
```

**Request Body:**
```json
{
  "bank": "bank-to-delete"
}
```

**Response:**
```json
{
  "status": "success",
  "deleted": "bank-to-delete",
  "current": "default"
}
```

**Note:** Cannot delete the "default" bank.

## Entities (Nodes)

### Create Entity

```http
POST /entities
POST /entities?bank=specific-bank
```

**Request Body:**
```json
{
  "id": "entity-id",
  "type": "node",
  "data": {
    "name": "Entity Name",
    "description": "Entity description",
    "custom_field": "custom value"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "entity": {
    "id": "entity-id",
    "type": "node",
    "data": {
      "name": "Entity Name",
      "description": "Entity description",
      "custom_field": "custom value"
    }
  },
  "bank": "default"
}
```

### Get All Entities

```http
GET /entities
GET /entities?bank=specific-bank
```

**Response:**
```json
[
  {
    "id": "entity-1",
    "type": "node",
    "data": {"name": "First Entity"}
  },
  {
    "id": "entity-2",
    "type": "node",
    "data": {"name": "Second Entity"}
  }
]
```

### Update Entity

```http
PUT /entities/{entity_id}
PUT /entities/{entity_id}?bank=specific-bank
```

**Request Body:**
```json
{
  "id": "entity-id",
  "type": "node",
  "data": {
    "name": "Updated Entity Name",
    "updated": true
  }
}
```

**Response:**
```json
{
  "status": "success",
  "entity": {
    "id": "entity-id",
    "type": "node",
    "data": {
      "name": "Updated Entity Name",
      "updated": true
    }
  },
  "bank": "default"
}
```

### Delete Entity

```http
DELETE /entities/{entity_id}
DELETE /entities/{entity_id}?bank=specific-bank
```

**Response:**
```json
{
  "status": "success",
  "deleted": "entity-id",
  "bank": "default"
}
```

## Relations (Edges)

### Create Relation

```http
POST /relations
POST /relations?bank=specific-bank
```

**Request Body:**
```json
{
  "source": "entity-1",
  "target": "entity-2",
  "type": "relation",
  "data": {
    "relationship_type": "connects_to",
    "weight": 0.8,
    "description": "Entity 1 connects to Entity 2"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "relation": {
    "id": "entity-1-entity-2-0",
    "source": "entity-1",
    "target": "entity-2",
    "type": "relation",
    "data": {
      "relationship_type": "connects_to",
      "weight": 0.8,
      "description": "Entity 1 connects to Entity 2"
    }
  },
  "bank": "default"
}
```

### Get All Relations

```http
GET /relations
GET /relations?bank=specific-bank
```

**Response:**
```json
[
  {
    "id": "rel-1",
    "source": "entity-1",
    "target": "entity-2",
    "type": "relation",
    "data": {"relationship_type": "connects_to"}
  }
]
```

### Update Relation

```http
PUT /relations/{relation_id}
PUT /relations/{relation_id}?bank=specific-bank
```

**Request Body:**
```json
{
  "id": "relation-id",
  "source": "entity-1",
  "target": "entity-2",
  "type": "relation",
  "data": {
    "relationship_type": "updated_connection",
    "weight": 0.9
  }
}
```

### Delete Relation

```http
DELETE /relations/{relation_id}
DELETE /relations/{relation_id}?bank=specific-bank
```

**Response:**
```json
{
  "status": "success",
  "deleted": "relation-id",
  "bank": "default"
}
```

## Observations

### Add Observation

```http
POST /observations
POST /observations?bank=specific-bank
```

**Request Body:**
```json
{
  "id": "obs-1",
  "entity_id": "entity-1",
  "content": "This entity demonstrates important behavior",
  "timestamp": "2025-01-19T12:00:00Z"
}
```

**Response:**
```json
{
  "status": "success",
  "observation": {
    "id": "obs-1",
    "entity_id": "entity-1",
    "content": "This entity demonstrates important behavior",
    "timestamp": "2025-01-19T12:00:00Z"
  },
  "bank": "default"
}
```

### Get All Observations

```http
GET /observations
GET /observations?bank=specific-bank
```

**Response:**
```json
[
  {
    "id": "obs-1",
    "entity_id": "entity-1",
    "content": "Observation content",
    "timestamp": "2025-01-19T12:00:00Z"
  }
]
```

## Sequential Thinking

### Add Reasoning Step

```http
POST /sequential-thinking
POST /sequential-thinking?bank=specific-bank
```

**Request Body:**
```json
{
  "id": "step-1",
  "description": "Analyzing the relationship between entities",
  "status": "completed",
  "timestamp": "2025-01-19T12:00:00Z",
  "related_entities": ["entity-1", "entity-2"],
  "related_relations": ["rel-1"]
}
```

**Response:**
```json
{
  "status": "success",
  "step": {
    "id": "step-1",
    "description": "Analyzing the relationship between entities",
    "status": "completed",
    "timestamp": "2025-01-19T12:00:00Z",
    "related_entities": ["entity-1", "entity-2"],
    "related_relations": ["rel-1"]
  },
  "bank": "default"
}
```

### Get All Reasoning Steps

```http
GET /sequential-thinking
GET /sequential-thinking?bank=specific-bank
```

**Response:**
```json
[
  {
    "id": "step-1",
    "description": "Reasoning step description",
    "status": "completed",
    "timestamp": "2025-01-19T12:00:00Z",
    "related_entities": ["entity-1"],
    "related_relations": []
  }
]
```

## Context Management

### Ingest Text Context

Automatically extract entities and relationships from text.

```http
POST /context/ingest
```

**Request Body:**
```json
{
  "text": "Alice works with Bob on the Project. Charlie manages the Project and coordinates with Alice.",
  "bank": "default"
}
```

**Response:**
```json
{
  "status": "success",
  "entities": ["Alice", "Bob", "Project", "Charlie"],
  "edges_added": 4,
  "bank": "default"
}
```

### Retrieve Full Context

Get all data from a memory bank.

```http
GET /context/retrieve
GET /context/retrieve?bank=specific-bank
```

**Response:**
```json
{
  "entities": [
    {
      "id": "entity-1",
      "type": "node",
      "data": {"name": "Entity 1"}
    }
  ],
  "relations": [
    {
      "id": "rel-1",
      "source": "entity-1",
      "target": "entity-2",
      "type": "relation",
      "data": {}
    }
  ],
  "observations": [
    {
      "id": "obs-1",
      "entity_id": "entity-1",
      "content": "Observation",
      "timestamp": "2025-01-19T12:00:00Z"
    }
  ],
  "reasoning_steps": [
    {
      "id": "step-1",
      "description": "Reasoning step",
      "status": "completed",
      "timestamp": "2025-01-19T12:00:00Z",
      "related_entities": [],
      "related_relations": []
    }
  ]
}
```

## MCP Protocol Endpoints

### Initialize

```http
POST /
```

**Request Body:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {"tools": {}},
    "clientInfo": {"name": "vscode", "version": "1.0.0"}
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-06-18",
    "serverInfo": {
      "name": "Graph Memory MCP Server",
      "version": "1.0"
    },
    "capabilities": {
      "tools": {"listChanged": true},
      "resources": {"subscribe": false, "listChanged": true},
      "roots": {"listChanged": true},
      "prompts": {"listChanged": false},
      "completion": {"supports": ["text"]}
    }
  }
}
```

### List Tools

```http
POST /
```

**Request Body:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
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
      }
    ]
  }
}
```

## Error Responses

### Standard Error Format

```json
{
  "status": "error",
  "message": "Error description",
  "bank": "bank-name"
}
```

### Common Error Codes

- **404**: Endpoint not found
- **405**: Method not allowed
- **422**: Validation error
- **500**: Internal server error

### MCP Error Format

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32601,
    "message": "Method not found"
  }
}
```

## Rate Limiting

No rate limiting is currently implemented for local deployment.

## Data Types

### Node Model

```typescript
interface Node {
  id: string;
  type: string; // default: "node"
  data: Record<string, any>;
}
```

### Edge Model

```typescript
interface Edge {
  id?: string; // auto-generated if not provided
  source: string;
  target: string;
  type: string; // default: "relation"
  data: Record<string, any>;
}
```

### Observation Model

```typescript
interface Observation {
  id: string;
  entity_id: string;
  content: string;
  timestamp: string;
}
```

### ReasoningStep Model

```typescript
interface ReasoningStep {
  id: string;
  description: string;
  status: string; // default: "pending"
  timestamp?: string;
  related_entities: string[];
  related_relations: string[];
}
```
