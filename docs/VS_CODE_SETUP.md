# VS Code Setup Guide

Complete guide for configuring the Graph Memory MCP Server with VS Code Agent Chat.

## Prerequisites

- VS Code with Agent Chat enabled
- Graph Memory MCP Server container built (see [Deployment Guide](./DEPLOYMENT.md))
- Basic understanding of MCP configuration

## Configuration Steps

### 1. Choose Your Mode

The Graph Memory MCP Server supports three modes:

- **🔄 Dual-mode (Recommended)**: VS Code uses stdio + external HTTP access
- **🌐 HTTP-only**: Universal HTTP access for all clients
- **📱 stdio-only**: VS Code only, no external access

### 2. Option A: Dual-mode Configuration (Recommended)

This mode provides VS Code with efficient stdio communication while also exposing HTTP access for external processes.

**Benefits:**
- VS Code gets optimal stdio performance
- External tools can access same server on port 10462
- Both interfaces share the same data
- Single container instance

```json
{
  "servers": {
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

### 3. Option B: HTTP-only Mode

Standard HTTP configuration if you prefer all clients to use HTTP.

**Start server manually:**
```bash
podman run -d --name graph-mcp-server -p 10642:10642 -v graph-mcp-memory:/data graph-mcp-server
```

**Configuration:**
```json
{
  "servers": {
    "graph-memory": {
      "url": "http://localhost:10642",
      "type": "http"
    }
  }
}
```

### 4. Option C: stdio-only Mode (Legacy)

VS Code only, no external access.

```json
{
  "servers": {
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

### 5. MCP Configuration File Location

Create or update your VS Code MCP configuration file. The location depends on your system:

**Windows:**
```
%APPDATA%\Code\User\mcp.json
```

**macOS:**
```
~/Library/Application Support/Code/User/mcp.json
```

**Linux:**
```
~/.config/Code/User/mcp.json
```

### 6. Restart VS Code

After updating the configuration:

1. Save the `mcp.json` file
2. Restart VS Code completely  
3. The MCP server should be automatically detected

**For dual-mode:** VS Code manages the container AND exposes HTTP on port 10462

## Verification

### 1. Check Agent Chat

1. Open VS Code Agent Chat
2. Look for the Graph Memory MCP Server in available tools
3. Try a simple query: "What memory tools are available?"

### 2. Test MCP Tools

Test each available tool:

**Create Entities:**
```
@mcp-server-graph-memory Create an entity called "test-project" of type "project" with observation "This is a test project"
```

**Add Observations:**
```
@mcp-server-graph-memory Add observation "Updated requirements" to entity "test-project"
```

**Create Relations:**
```
@mcp-server-graph-memory Create a relation from "test-project" to "user-requirements" of type "contains"
```

**Sequential Thinking:**
```
@mcp-server-graph-memory Add reasoning step: "Analyzing project structure for optimal organization"
```

### 3. Verify Persistence

1. Create some test data using Agent Chat
2. Restart the MCP server container
3. Verify data persists by querying the entities

## Troubleshooting

### Server Not Detected

**Problem:** VS Code doesn't detect the MCP server

**Solutions:**
1. Verify server is running: `curl http://localhost:8000/`
2. Check `mcp.json` syntax is valid JSON
3. Ensure file is in correct location
4. Restart VS Code completely
5. Check VS Code logs for MCP errors

### Connection Timeout

**Problem:** MCP initialization times out

**Solutions:**
1. Increase timeout in MCP configuration:
```json
{
  "mcp-server-graph-memory": {
    "command": "curl",
    "args": [
      "-N",
      "-H", "Accept: text/event-stream",
      "--max-time", "30",
      "http://localhost:8000/"
    ]
  }
}
```

2. Check server logs: `podman logs graph-mcp-server`
3. Verify no firewall blocking port 8000

### Tools Not Available

**Problem:** MCP server connects but tools aren't available

**Solutions:**
1. Test tools endpoint directly:
```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
```

2. Check server logs for initialization errors
3. Verify MCP protocol compliance in server responses

### Permission Issues

**Problem:** VS Code can't access MCP configuration

**Solutions:**
1. Check file permissions on `mcp.json`
2. Run VS Code as administrator (Windows) or with appropriate permissions
3. Verify directory exists and is writable

## Advanced Configuration

### Multiple Servers

You can configure multiple MCP servers:

```json
{
  "mcp-server-graph-memory": {
    "command": "curl",
    "args": [
      "-N",
      "-H", "Accept: text/event-stream",
      "http://localhost:8000/"
    ]
  },
  "mcp-server-other": {
    "command": "other-mcp-server",
    "args": ["--config", "other-config.json"]
  }
}
```

### Environment Variables

Pass environment variables to the MCP server:

```json
{
  "mcp-server-graph-memory": {
    "command": "curl",
    "args": [
      "-N",
      "-H", "Accept: text/event-stream",
      "http://localhost:8000/"
    ],
    "env": {
      "MCP_LOG_LEVEL": "DEBUG",
      "MCP_TIMEOUT": "30"
    }
  }
}
```

### Custom Timeouts

Configure custom timeouts for slow connections:

```json
{
  "mcp-server-graph-memory": {
    "command": "timeout",
    "args": [
      "60s",
      "curl",
      "-N",
      "-H", "Accept: text/event-stream",
      "http://localhost:8000/"
    ]
  }
}
```

## Usage Examples

### Basic Conversation Memory

Use the MCP server to maintain conversation context:

```
User: "I'm working on a web application with user authentication"

Agent: I'll help you with that. Let me store this context in memory.

@mcp-server-graph-memory Create entities for "web-application" (type: project) and "user-authentication" (type: feature) with relevant observations

@mcp-server-graph-memory Create relation from "web-application" to "user-authentication" of type "includes"
```

### Project Knowledge Base

Build a cumulative knowledge base:

```
User: "The authentication should support OAuth2 and JWT tokens"

Agent: I'll add this requirement to our project knowledge.

@mcp-server-graph-memory Add observations to "user-authentication": "Requires OAuth2 support" and "Must handle JWT tokens"

@mcp-server-graph-memory Create entity "oauth2" (type: technology) with observation "Required for user authentication"

@mcp-server-graph-memory Create relation from "user-authentication" to "oauth2" of type "uses"
```

### Reasoning Chain Tracking

Track complex problem-solving:

```
User: "How should I structure the authentication system?"

Agent: Let me think through this systematically.

@mcp-server-graph-memory Add reasoning step: "Analyzing authentication system architecture requirements"

@mcp-server-graph-memory Add reasoning step: "Considering OAuth2 integration patterns and JWT token management"
```

## Best Practices

### 1. Memory Bank Organization

- Create separate banks for different projects
- Use descriptive bank names
- Switch banks contextually

### 2. Entity Management

- Use consistent naming conventions
- Include meaningful observations
- Update entities rather than creating duplicates

### 3. Relationship Modeling

- Use standard relationship types
- Create bidirectional relationships when appropriate
- Document relationship semantics

### 4. Error Handling

- Monitor for MCP connection issues
- Have fallback strategies for offline scenarios
- Log important interactions for debugging

## Maintenance

### Regular Tasks

1. **Monitor Logs:** Check MCP server logs regularly
```bash
podman logs graph-mcp-server | tail -50
```

2. **Update Configuration:** Keep MCP configuration current
3. **Backup Data:** Backup persistent volumes
```bash
podman volume export graph-mcp-memory > backup.tar
```

4. **Test Connectivity:** Regularly verify MCP connection
```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "initialize"}'
```

### Updates

When updating the server:

1. Stop the container
2. Pull/rebuild latest image
3. Start with same volume mount
4. Verify MCP tools still work
5. Test data persistence

## Related Documentation

- [MCP Integration Guide](./MCP_INTEGRATION.md) - AI agent integration
- [API Reference](./API.md) - Complete REST API
- [Deployment Guide](./DEPLOYMENT.md) - Container deployment
