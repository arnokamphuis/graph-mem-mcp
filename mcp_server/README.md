# MCP Graph Memory Server

A FastAPI-based knowledge graph server with interactive visualization that runs as both an HTTP server and MCP (Model Context Protocol) server.

## Features

- **Knowledge Graph Storage**: Store entities, relationships, and observations in memory banks
- **Interactive Visualization**: Beautiful web-based graph visualization using vis.js
- **Memory Banks**: Organize knowledge into separate, isolated banks
- **Search Capabilities**: Advanced search with fuzzy matching support
- **Dual Mode**: Run as HTTP server and/or MCP stdio server
- **Persistent Storage**: Data persistence using JSON files
- **RESTful API**: Complete REST API with automatic documentation

## Container Usage

### Prerequisites
- Podman (or Docker) installed on your system

### Quick Start

1. **Build and Run** (Windows):
   ```batch
   start_container.bat
   ```

2. **Build and Run** (Linux/macOS):
   ```bash
   chmod +x start_container.sh
   ./start_container.sh
   ```

3. **Manual Commands**:
   ```bash
   # Build the container
   podman build -t graph-mem-mcp .
   
   # Run with HTTP server + MCP support
   podman run -d \
     --name graph-mem-mcp \
     -p 10642:10642 \
     -v graph-mem-data:/data \
     graph-mem-mcp \
     python main.py --mcp-with-http
   ```

### Access Points

- **Visualization**: http://localhost:10642
- **API Documentation**: http://localhost:10642/docs
- **OpenAPI Spec**: http://localhost:10642/openapi.json

### Container Management

```bash
# View logs
podman logs graph-mem-mcp

# Stop container
podman stop graph-mem-mcp

# Remove container
podman rm graph-mem-mcp

# Remove image
podman rmi graph-mem-mcp
```

## Usage Examples

### 1. Creating Knowledge Banks

```bash
# Create a new memory bank
curl -X POST "http://localhost:10642/banks/create" \
  -H "Content-Type: application/json" \
  -d '{"bank": "my-project"}'

# List all banks
curl "http://localhost:10642/banks/list"
```

### 2. Adding Knowledge

```bash
# Add an entity
curl -X POST "http://localhost:10642/entities?bank=my-project" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "John_Doe",
    "type": "node",
    "data": {
      "type": "person",
      "observations": ["Software engineer", "Works at TechCorp"]
    }
  }'

# Add a relationship
curl -X POST "http://localhost:10642/relations?bank=my-project" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "John_Doe",
    "target": "TechCorp",
    "type": "relation",
    "data": {"type": "works_at"}
  }'
```

### 3. Knowledge Ingestion

```bash
# Ingest text and auto-extract entities/relationships
curl -X POST "http://localhost:10642/context/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "John Doe works at TechCorp as a software engineer. He collaborates with Alice Smith on various projects.",
    "bank": "my-project"
  }'
```

### 4. Searching Knowledge

```bash
# Search entities with fuzzy matching
curl "http://localhost:10642/search/entities?q=John&fuzzy_match=true&bank=my-project"

# Search all knowledge
curl "http://localhost:10642/search/all?q=software&bank=my-project"
```

## Run Modes

### HTTP Server Only
```bash
podman run -p 10642:10642 graph-mem-mcp
```

### MCP Stdio Only
```bash
podman run -i graph-mem-mcp python main.py --mcp
```

### Both HTTP + MCP (Recommended)
```bash
podman run -p 10642:10642 graph-mem-mcp python main.py --mcp-with-http
```

## Data Persistence

The container uses a named volume `graph-mem-data` to persist knowledge graphs across container restarts. Data is stored in `/data/memory_banks.json` inside the container.

## Architecture

- **Backend**: FastAPI with Pydantic models
- **Frontend**: HTML5 + CSS3 + vanilla JavaScript
- **Visualization**: vis.js network library
- **Storage**: JSON file persistence
- **Container**: Python 3.11 slim base image

## File Structure

```
mcp_server/
├── main.py              # Main server application
├── templates/
│   └── visualization.html  # Visualization template
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
├── start_container.sh  # Linux/macOS startup script
├── start_container.bat # Windows startup script
└── README.md           # This file
```

## API Endpoints

### Banks
- `POST /banks/create` - Create new memory bank
- `POST /banks/select` - Select active bank
- `GET /banks/list` - List all banks
- `POST /banks/delete` - Delete a bank
- `POST /banks/clear` - Clear bank contents

### Knowledge Management
- `POST /entities` - Add entity
- `GET /entities` - Get all entities
- `PUT /entities/{id}` - Update entity
- `DELETE /entities/{id}` - Delete entity
- `POST /relations` - Add relationship
- `GET /relations` - Get all relationships

### Search
- `GET /search/entities` - Search entities
- `GET /search/relationships` - Search relationships
- `GET /search/all` - Search everything

### Visualization
- `GET /` - Interactive visualization
- `GET /banks/{bank}/visualize` - Bank-specific visualization
- `GET /visualizations` - Get visualization data

## Troubleshooting

### Container Won't Start
```bash
# Check if port is already in use
podman ps -a
netstat -an | grep 10642

# Check logs
podman logs graph-mem-mcp
```

### Can't Access Visualization
1. Ensure container is running: `podman ps`
2. Check port mapping: `podman port graph-mem-mcp`
3. Verify firewall settings
4. Try accessing: http://localhost:10642

### Data Not Persisting
- Ensure the volume is mounted: `podman inspect graph-mem-mcp`
- Check volume exists: `podman volume ls`
- Verify permissions in container logs

## Development

For development, you can mount the source code:

```bash
podman run -it \
  -p 10642:10642 \
  -v "$(pwd)":/app \
  -v graph-mem-data:/data \
  python:3.11-slim \
  bash
```

Then install dependencies and run:
```bash
cd /app
pip install -r requirements.txt
python main.py --mcp-with-http
```
