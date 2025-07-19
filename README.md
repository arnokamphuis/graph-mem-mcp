# Graph Memory MCP Server

A Model Context Protocol (MCP) compliant server that provides persistent graph-based memory management for AI agents. This server allows AI agents to store, retrieve, and manipulate entities, relationships, observations, and reasoning steps in a graph structure with multi-bank support.

## 🎯 Features

- **MCP Protocol Compliance**: Full JSON-RPC 2.0 support for VS Code Agent Chat integration
- **Persistent Storage**: File-based persistence using JSON with automatic save/load
- **Multi-Bank Architecture**: Support for multiple isolated memory banks
- **Graph Operations**: Complete CRUD operations for entities, relationships, and observations
- **Sequential Thinking**: Support for reasoning step storage and retrieval
- **Context Ingestion**: Automatic entity extraction from text
- **Container Deployment**: Docker/Podman containerization with volume mounting
- **RESTful API**: Full REST endpoints alongside MCP protocol

## 🚀 Quick Start

### 1. Build and Run with Podman

```bash
# Clone the repository
git clone <repo-url>
cd graph_mem

# Build the container
podman build -t graph-mcp-server ./mcp_server

# Run with persistent storage
podman run -d --name graph-mcp-server -p 8000:8000 -v graph-mcp-memory:/data graph-mcp-server
```

### 2. Configure VS Code Agent Chat

Create or update your VS Code `mcp.json` configuration:

```json
{
  "mcp-server-graph-memory": {
    "command": "curl",
    "args": [
      "-N",
      "-H", "Accept: text/event-stream",
      "http://localhost:8000/"
    ]
  }
}
```

### 3. Test the Server

```bash
# Check server status
curl http://localhost:8000/

# Create an entity
curl -X POST http://localhost:8000/entities \
  -H "Content-Type: application/json" \
  -d '{"id": "test-entity", "data": {"type": "example"}}'

# List all entities
curl http://localhost:8000/entities
```

## 📚 Documentation

- [API Reference](./docs/API.md) - Complete REST API documentation
- [MCP Integration](./docs/MCP_INTEGRATION.md) - AI agent integration guide
- [Deployment Guide](./docs/DEPLOYMENT.md) - Container deployment instructions
- [VS Code Setup](./docs/VS_CODE_SETUP.md) - Agent Chat configuration

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AI Agents     │    │  VS Code Agent   │    │  Direct REST    │
│                 │    │      Chat        │    │     Clients     │
└─────────┬───────┘    └────────┬─────────┘    └─────────┬───────┘
          │                     │                        │
          │ MCP Protocol        │ JSON-RPC 2.0           │ HTTP REST
          │                     │                        │
          └─────────────────────┼────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Graph Memory MCP     │
                    │       Server          │
                    │                       │
                    │  ┌─────────────────┐  │
                    │  │  Memory Banks   │  │
                    │  │                 │  │
                    │  │  ┌───────────┐  │  │
                    │  │  │ Entities  │  │  │
                    │  │  │Relations  │  │  │
                    │  │  │Observations│ │  │
                    │  │  │Reasoning  │  │  │
                    │  │  └───────────┘  │  │
                    │  └─────────────────┘  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Persistent Storage  │
                    │    (JSON Files)       │
                    └───────────────────────┘
```

## 🔧 Memory Banks

The server supports multiple isolated memory banks:

- **Default Bank**: Always available, cannot be deleted
- **Custom Banks**: Create, select, and manage multiple banks
- **Bank-Specific Operations**: All operations can target specific banks

```bash
# Create a new bank
curl -X POST http://localhost:8000/banks/create \
  -H "Content-Type: application/json" \
  -d '{"bank": "project-alpha"}'

# Switch to the bank
curl -X POST http://localhost:8000/banks/select \
  -H "Content-Type: application/json" \
  -d '{"bank": "project-alpha"}'

# Add entities to specific bank
curl -X POST http://localhost:8000/entities?bank=project-alpha \
  -H "Content-Type: application/json" \
  -d '{"id": "alpha-entity", "data": {"project": "alpha"}}'
```

## 🤖 AI Agent Integration

### VS Code Configuration

To use this MCP server with VS Code Agent Chat, create or update your MCP configuration file. The exact location depends on your VS Code setup, but common locations include:

- `%APPDATA%\Code\User\globalStorage\rooveterinaryinc.roo-cline\settings\cline_mcp_settings.json`
- Or look for MCP settings in VS Code's Agent Chat extension settings

```json
{
  "mcpServers": {
    "graph-memory": {
      "command": "podman",
      "args": [
        "run",
        "-i",
        "--rm",
        "-v",
        "graph-mcp-memory:/data",
        "graph-mcp-server",
        "python",
        "main.py",
        "--mcp"
      ],
      "type": "stdio"
    }
  }
}
```

No need to start the server manually - VS Code will manage the container lifecycle automatically.

### MCP Tools Available

1. **create_entities**: Create multiple entities with observations
2. **add_observations**: Add observations to existing entities
3. **create_relations**: Create relationships between entities
4. **sequential_thinking**: Add reasoning steps
5. **ingest_knowledge**: 🆕 **Advanced knowledge graph creation from large text with sophisticated entity and relationship extraction**

### Enhanced Knowledge Graph Creation

The new `ingest_knowledge` tool can transform any large text into a structured knowledge graph! It automatically extracts:

- **Named entities** (people, companies, technologies)
- **Technical terms** and concepts  
- **Relationships** with context (created, leads, known as)
- **Contextual observations** with source attribution
- **Confidence scores** and timestamps

**Example**: From a 500-word AI research text, it extracted 13 entities, 14 relationships, and 17 contextual observations with full source attribution!

See [KNOWLEDGE_GRAPH_FEATURES.md](KNOWLEDGE_GRAPH_FEATURES.md) for complete documentation.

### Example MCP Tool Call

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "create_entities",
    "arguments": {
      "entities": [
        {
          "name": "user-requirement",
          "entityType": "requirement",
          "observations": ["User wants graph memory", "Should persist data"]
        }
      ]
    }
  }
}
```

## 📊 Data Persistence

- **Automatic Saving**: All mutations automatically save to disk
- **JSON Format**: Human-readable storage format
- **Volume Mounting**: Container data persists in mounted volumes
- **Startup Loading**: Automatic data restoration on server restart

## 🔍 Monitoring

Check server logs for operation details:

```bash
# View container logs
podman logs graph-mcp-server

# Follow logs in real-time
podman logs -f graph-mcp-server
```

## 🛠️ Development

### Local Development

```bash
# Install dependencies
pip install fastapi uvicorn pydantic

# Run locally
cd mcp_server
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Container Management

```bash
# Stop server
podman stop graph-mcp-server

# Start server
podman start graph-mcp-server

# Remove container
podman rm graph-mcp-server

# Remove volume (WARNING: deletes all data)
podman volume rm graph-mcp-memory
```

## 📈 Use Cases

- **AI Agent Memory**: Persistent memory for conversational AI
- **Knowledge Graphs**: Building and maintaining knowledge representations
- **Context Management**: Long-term context storage across sessions
- **Reasoning Chains**: Storing complex reasoning processes
- **Multi-Project Memory**: Isolated memory spaces for different projects

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

[License information would go here]

## 🔗 Related Projects

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [VS Code Agent Chat](https://code.visualstudio.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
