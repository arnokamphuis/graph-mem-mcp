# Graph Memory MCP Server

A Model Context Protocol (MCP) compliant server that provides persistent graph-based memory management with **advanced knowledge graph creation and interactive visualization** for AI agents. This server allows AI agents to store, retrieve, and manipulate entities, relationships, observations, and reasoning steps in a graph structure with multi-bank support.

## 🎯 Features

- **MCP Protocol Compliance**: Full JSON-RPC 2.0 support for VS Code Agent Chat integration
- **Persistent Storage**: File-based persistence using JSON with automatic save/load
- **Multi-Bank Architecture**: Support for multiple isolated memory banks
- **🧠 Intelligent Knowledge Extraction**: **NEW!** Auto-extraction of entities and relationships from observation text
- **Graph Operations**: Complete CRUD operations for entities, relationships, and observations
- **Sequential Thinking**: Support for reasoning step storage and retrieval
- **Advanced Knowledge Graph Creation**: Sophisticated entity extraction from large text with confidence scoring
- **Fuzzy Matching & Typo Handling**: Intelligent search and entity deduplication with typo tolerance
- **Interactive Visualization**: Beautiful web-based graph visualization using vis.js Network library
- **Smart Entity Detection**: Multiple entity types (named_entity, technical_term, concept, email, url, measurement, date)
- **Intelligent Relationship Analysis**: Context-aware relationship detection with confidence scoring
- **Container Deployment**: Docker/Podman containerization with volume mounting
- **RESTful API**: Full REST endpoints alongside MCP protocol

## 🏦 **CRITICAL: Project-Based Memory Bank Organization for AI Agents**

**⚠️ AI agents MUST create and use project-specific memory banks for ALL knowledge operations.**

### 🎯 Bank Organization Rules

**MANDATORY for AI Agents:**
1. **Create a dedicated bank** for each project/task before any entity operations
2. **Never use the 'default' bank** for real project work
3. **Name banks descriptively** using project context
4. **All entities and observations** for a project must go in that project's bank

**Bank Naming Convention:**
```
project-{client/org}-{component}     ← For client work
feature-{name}-{context}             ← For feature development  
research-{topic}-{timeframe}         ← For research projects
bugfix-{issue-id}-{component}        ← For bug fixes
```

**Example Organization:**
```
📁 project-acme-ecommerce-backend    ← All Acme e-commerce backend knowledge
📁 project-acme-ecommerce-frontend   ← All Acme e-commerce frontend knowledge
📁 feature-user-authentication       ← Authentication feature development
📁 research-ai-performance-q1-2025   ← Q1 2025 AI performance research
📁 bugfix-1234-payment-gateway       ← Payment gateway bug fix
```

### 🔄 Workflow for AI Agents
```markdown
1. Identify the project/task context
2. Create project-specific bank: `create_bank {"name": "project-acme-auth"}`
3. Select the bank: `select_bank {"name": "project-acme-auth"}`
4. ALL subsequent operations use this bank automatically
5. Never mix different projects in the same bank
```

## 🚀 Quick Start

### 1. Build and Run with Podman

```bash
# Clone the repository
git clone <repo-url>
cd graph_mem

# Build the container
podman build -t graph-mcp-server ./mcp_server

# Option A: Run with persistent storage (HTTP only)
podman run -d --name graph-mcp-server -p 10642:10642 -v graph-mcp-memory:/data graph-mcp-server

# Option B: Run with dual-mode support (stdio + HTTP)
podman run -d --name graph-mcp-server -p 10462:10642 -v graph-mcp-memory:/data graph-mcp-server python main.py --mcp-with-http
```

### 2. Configure VS Code Agent Chat

**Option A: HTTP-only Mode**
Create or update your VS Code `mcp.json` configuration:

```json
{
  "graph-memory": {
    "url": "http://localhost:10642",
    "type": "http"
  }
}
```

**Option B: Dual-mode (Recommended)**
For both VS Code stdio communication AND external HTTP access:

```json
{
  "graph-memory": {
    "command": "podman",
    "args": [
      "run",
      "-i",
      "--rm",
      "-p",
      "10462:10642",
      "-v",
      "graph-mcp-memory:/data",
      "graph-mcp-server",
      "python",
      "main.py",
      "--mcp-with-http"
    ],
    "type": "stdio"
  }
}
```

**🔄 Dual-mode Benefits:**
- VS Code communicates efficiently via stdio
- External processes can access the same server instance via HTTP at `http://localhost:10462`
- Both interfaces share the same memory banks and data
- No need to run separate server instances

### 3. Test the Server

**HTTP-only Mode:**
```bash
# Check server status
curl http://localhost:10642/

# Create an entity via REST API
curl -X POST http://localhost:10642/entities \
  -H "Content-Type: application/json" \
  -d '{"id": "test-entity", "data": {"type": "example"}}'
```

**Dual-mode Testing:**
```bash
# Test HTTP interface (external access)
curl http://localhost:10462/

# Test same functionality as VS Code will use
# (VS Code uses stdio, but data is shared)

# Advanced: Ingest large text for knowledge graph creation
curl -X POST http://localhost:10462/knowledge/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Your large text here...", "source": "document_name", "bank_name": "default"}'

# View interactive visualization (works with both modes)
# HTTP-only mode: http://localhost:10642/banks/default/visualize  
# Dual-mode: http://localhost:10462/banks/default/visualize
open http://localhost:10462/banks/default/visualize
```

## 🎨 **NEW: Interactive Knowledge Graph Visualization**

The server now includes **beautiful web-based interactive visualization** powered by vis.js Network library:

### Key Visualization Features:
- **� Dynamic Bank Switching**: Switch between memory banks without page reload
- **📊 Bank Statistics**: View entity and relationship counts for each bank
- **�🌈 Color-Coded Entities**: Blue (named entities), Green (technical terms), Purple (concepts)
- **📏 Smart Sizing**: Node size reflects confidence scores
- **🔗 Relationship Styling**: Edge thickness and colors based on relationship type and confidence
- **🔍 Interactive Controls**: Zoom, pan, search, filter entities and relationships
- **📐 Multiple Layouts**: Choose from hierarchical, force-directed, or custom arrangements
- **💾 Export Capability**: Save visualizations as PNG images with bank-specific names
- **⚡ Real-time Updates**: Dynamic visualization updates as knowledge graphs evolve
- **🔄 State Preservation**: Maintains search terms and layout when switching banks

### Access Your Visualizations:
- **Main Interface with Bank Switching**: `http://localhost:10642/visualize`
- **Bank-Specific Interface**: `http://localhost:10642/banks/{bank}/visualize`
- **Graph Data API**: `http://localhost:10642/banks/{bank}/graph-data`
- **Available Visualizations**: `http://localhost:10642/visualizations`

## 🧠 **NEW: Advanced Knowledge Graph Creation**

Transform large text into sophisticated knowledge graphs with intelligent entity extraction:

### Enhanced Capabilities:
- **🏷️ Multiple Entity Types**: Named entities, technical terms, concepts, emails, URLs, measurements, dates
- **🤖 Intelligent Relationship Detection**: Context-aware analysis with action, categorical, and temporal relationships
- **📊 Confidence Scoring**: Each entity and relationship gets confidence scores (0.4-0.9)
- **📚 Source Attribution**: Every extracted element tagged with source document for traceability
- **🔍 Entity Deduplication**: **NEW!** Prevents duplicate entities from typos and variations using fuzzy matching
- **🛡️ Smart Normalization**: Automatically merges similar entities (e.g., "Goldman Sach" → "Goldman Sachs")
- **⚙️ Large Text Processing**: Efficiently handles complex documents and creates comprehensive knowledge graphs

### Example Usage:
```bash
# Ingest financial document
curl -X POST http://localhost:10642/knowledge/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Goldman Sachs is a leading global investment bank founded in 1869 by Marcus Goldman...",
    "source": "financial_overview",
    "bank_name": "default"
  }'

# Result: Automatically extracted 26+ entities, 24+ relationships with confidence scoring
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

**Recommended: Dual-mode Configuration**
```json
{
  "mcpServers": {
    "graph-memory": {
      "command": "podman",
      "args": [
        "run",
        "-i",
        "--rm",
        "-p",
        "10462:10642",
        "-v",
        "graph-mcp-memory:/data",
        "graph-mcp-server",
        "python",
        "main.py",
        "--mcp-with-http"
      ],
      "type": "stdio"
    }
  }
}
```

**Alternative: HTTP-only Mode**
```json
{
  "mcpServers": {
    "graph-memory": {
      "url": "http://localhost:10642",
      "type": "http"
    }
  }
}
```

**Legacy: stdio-only Mode**
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

**🎯 Configuration Comparison:**
- **Dual-mode**: VS Code gets efficient stdio + external HTTP access on port 10462
- **HTTP-only**: Universal HTTP access but VS Code must use HTTP protocol  
- **stdio-only**: VS Code only, no external access

### MCP Tools Available

1. **create_entities**: 🧠 **Enhanced!** Create multiple entities with automatic knowledge extraction from observations
2. **add_observations**: 🧠 **Enhanced!** Add observations to existing entities with automatic knowledge extraction
3. **create_relations**: Create relationships between entities
4. **sequential_thinking**: Add reasoning steps
5. **ingest_knowledge**: **Advanced knowledge graph creation from large text with sophisticated entity and relationship extraction**
6. **search_nodes**: **Search entities by name, type, or observations content with relevance scoring**
7. **search_relations**: **Search relationships by type, context, or entity names with filtering**
8. **search_observations**: **Search observations by content or entity with advanced matching**
9. **search_all**: **Comprehensive search across all entities, relationships, and observations**

## 🧠 **NEW: Intelligent Knowledge Extraction**

**Both `create_entities` and `add_observations` now automatically extract valuable knowledge from observation text!**

### ✨ Auto-Extraction Features
- **🔍 Smart Entity Detection**: Automatically finds people, organizations, technical terms, dates, measurements, emails, URLs
- **🔗 Relationship Discovery**: Identifies connections and relationships between entities mentioned in text
- **🎯 Confidence Scoring**: Each extracted entity and relationship includes confidence scores
- **🏷️ Type Classification**: Automatically categorizes entities by type (named_entity, technical_term, concept, etc.)
- **🔄 Intelligent Connections**: Creates meaningful relationships between existing and newly discovered entities

### 🎛️ Control Parameters
Both tools support the `auto_extract` parameter:
- **`auto_extract: true`** (default): Enable intelligent knowledge extraction
- **`auto_extract: false`**: Simple storage without extraction

### 📝 Example: Rich Knowledge Extraction

**Input:**
```json
{
  "entities": [{
    "name": "ai-research-project",
    "entityType": "project", 
    "observations": [
      "Collaboration between OpenAI and Microsoft on GPT-4 integration",
      "Team includes Sarah Johnson, Dr. Marcus Chen working on transformer architecture",
      "Budget: $2.5 million for Q1 2025 deliverables"
    ]
  }]
}
```

**Auto-Extracted:**
- **Entities**: "OpenAI", "Microsoft", "GPT-4", "Sarah Johnson", "Dr. Marcus Chen", "transformer architecture", "$2.5 million", "Q1 2025"
- **Relationships**: "OpenAI collaborates_with Microsoft", "Sarah Johnson works_on transformer architecture", etc.
- **All connected** to the main "ai-research-project" entity

## 🔍 **NEW: Comprehensive Search Capabilities**

The server now includes **powerful search functionality** to find information across your knowledge graphs:

### Search Features:
- **🎯 Relevance Scoring**: Results ranked by relevance (1.0 = perfect match, 0.3-0.7 = partial matches)
- **📝 Multiple Match Types**: Exact matches, word matches, partial text matches
- **🔤 Text Options**: Case-sensitive/insensitive search, regular expression support
- **🔍 Fuzzy Matching**: **NEW!** Intelligent typo tolerance with configurable similarity thresholds
- **🛡️ Typo Handling**: Finds entities despite 1-2 character differences, missing letters, or case variations
- **🏷️ Advanced Filtering**: Filter by entity type, relationship type, or specific memory banks
- **⚡ Cross-Bank Search**: Search across all memory banks or target specific ones
- **📊 Rich Results**: Includes matched fields, relevance scores, and comprehensive metadata

### Search HTTP Endpoints:
- **Entity Search**: `GET /search/entities?q=searchterm`
- **Relationship Search**: `GET /search/relationships?q=searchterm`
- **Observation Search**: `GET /search/observations?q=searchterm`
- **Universal Search**: `GET /search/all?q=searchterm`

### Example Search Usage:

```bash
# Search for entities related to Goldman Sachs
curl "http://localhost:10642/search/entities?q=Goldman&limit=10"

# Search with fuzzy matching for typos (NEW!)
curl "http://localhost:10642/search/entities?q=Goldmann&fuzzy_match=true&fuzzy_threshold=0.8"

# Search for acquisition relationships
curl "http://localhost:10642/search/relationships?q=acquired"

# Comprehensive search across all data types
curl "http://localhost:10642/search/all?q=Marcus&case_sensitive=false"

# Advanced regex search for dates
curl "http://localhost:10642/search/observations?q=\\d{4}&use_regex=true"

# Fuzzy search with custom threshold for more permissive matching
curl "http://localhost:10642/search/entities?q=goldman&fuzzy_match=true&fuzzy_threshold=0.7"
```

**Example Results**: 
- **Standard Search**: "Marcus" found 13 results across entities (2), relationships (4), and observations (7)
- **Fuzzy Search**: "Goldmann" with fuzzy matching finds both correct "Goldman" entities and typo variants
- **Entity Deduplication**: Ingesting "Goldman Sach" automatically merges with existing "Goldman Sachs"

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
