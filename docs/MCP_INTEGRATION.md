# MCP Integration Guide

This guide explains how AI agents can integrate with the Graph Memory MCP Server using the Model Context Protocol.

## Overview

The Graph Memory MCP Server provides AI agents with persistent, graph-based memory capabilities through the Model Context Protocol (MCP). Agents can store entities, relationships, observations, and reasoning steps that persist across conversations and sessions.

## MCP Protocol Support

### Supported Methods

1. **initialize**: Initialize MCP connection
2. **tools/list**: Get available tools
3. **tools/call**: Execute tools (via MCP client)

### Protocol Version

- **Server Protocol**: 2025-06-18
- **Supported Client Versions**: 2024-11-05 and later

## Available Tools

### 1. create_entities

Create multiple entities in the knowledge graph.

**Input Schema:**
```json
{
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
```

**Example Usage:**
```json
{
  "name": "create_entities",
  "arguments": {
    "entities": [
      {
        "name": "user-requirement",
        "entityType": "requirement",
        "observations": [
          "User wants to implement authentication",
          "Should support OAuth2",
          "Must be secure"
        ]
      },
      {
        "name": "authentication-system",
        "entityType": "system",
        "observations": [
          "Handles user login",
          "Validates credentials"
        ]
      }
    ]
  }
}
```

### 2. add_observations

Add observations to existing entities.

**Input Schema:**
```json
{
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
```

**Example Usage:**
```json
{
  "name": "add_observations",
  "arguments": {
    "observations": [
      {
        "entityName": "user-requirement",
        "contents": [
          "Updated to include MFA support",
          "Performance requirement: <200ms login"
        ]
      }
    ]
  }
}
```

### 3. create_relations

Create relationships between entities.

**Input Schema:**
```json
{
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
```

**Example Usage:**
```json
{
  "name": "create_relations",
  "arguments": {
    "relations": [
      {
        "from": "user-requirement",
        "to": "authentication-system",
        "relationType": "implements"
      },
      {
        "from": "authentication-system",
        "to": "oauth2-provider",
        "relationType": "uses"
      }
    ]
  }
}
```

### 4. sequential_thinking

Add reasoning steps to track thought processes.

**Input Schema:**
```json
{
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
```

**Example Usage:**
```json
{
  "name": "sequential_thinking",
  "arguments": {
    "step": {
      "thought": "Need to analyze security requirements",
      "step_number": 1,
      "reasoning": "Before implementing authentication, must understand security context and threat model"
    }
  }
}
```

## Integration Patterns

### 1. Context Preservation

Store conversation context for future reference:

```python
# Pseudo-code for AI agent
def store_conversation_context(user_message, ai_response):
    # Extract key entities from conversation
    entities = extract_entities(user_message + ai_response)
    
    # Store entities with observations
    mcp_client.call_tool("create_entities", {
        "entities": [
            {
                "name": entity.name,
                "entityType": "conversation_topic",
                "observations": [
                    f"Discussed in conversation on {timestamp}",
                    f"User interest level: {entity.importance}",
                    f"Context: {entity.context}"
                ]
            }
            for entity in entities
        ]
    })
```

### 2. Knowledge Building

Build cumulative knowledge across conversations:

```python
def update_knowledge(new_information):
    # Add new observations to existing entities
    mcp_client.call_tool("add_observations", {
        "observations": [
            {
                "entityName": "project-requirements",
                "contents": [new_information.requirement]
            }
        ]
    })
    
    # Create relationships between concepts
    if new_information.relationships:
        mcp_client.call_tool("create_relations", {
            "relations": new_information.relationships
        })
```

### 3. Reasoning Chain Storage

Track complex reasoning processes:

```python
def store_reasoning_step(thought, step_num, reasoning):
    mcp_client.call_tool("sequential_thinking", {
        "step": {
            "thought": thought,
            "step_number": step_num,
            "reasoning": reasoning
        }
    })
```

### 4. Multi-Bank Organization

Organize memory by project or context:

```python
# Direct REST API calls for bank management
def setup_project_memory(project_name):
    # Create dedicated bank
    requests.post(f"{server_url}/banks/create", 
                  json={"bank": project_name})
    
    # Switch to project bank
    requests.post(f"{server_url}/banks/select", 
                  json={"bank": project_name})
    
    # Now all MCP tool calls will use this bank
    mcp_client.call_tool("create_entities", {
        "entities": [
            {
                "name": f"{project_name}-overview",
                "entityType": "project",
                "observations": [f"Project {project_name} started"]
            }
        ]
    })
```

## Error Handling

### MCP Errors

Handle standard JSON-RPC errors:

```python
def handle_mcp_error(response):
    if "error" in response:
        error_code = response["error"]["code"]
        error_message = response["error"]["message"]
        
        if error_code == -32601:  # Method not found
            print(f"Unsupported method: {error_message}")
        elif error_code == -32602:  # Invalid params
            print(f"Invalid parameters: {error_message}")
        else:
            print(f"MCP error {error_code}: {error_message}")
```

### Tool Call Errors

Handle tool-specific errors:

```python
def safe_tool_call(tool_name, arguments):
    try:
        result = mcp_client.call_tool(tool_name, arguments)
        if result.get("status") == "error":
            print(f"Tool error: {result.get('message')}")
            return None
        return result
    except Exception as e:
        print(f"Tool call failed: {e}")
        return None
```

## Best Practices

### 1. Entity Naming

- Use descriptive, consistent naming conventions
- Include context in entity names when needed
- Example: `user-auth-requirement` instead of `requirement1`

### 2. Observation Quality

- Store specific, actionable observations
- Include timestamps when relevant
- Reference sources of information
- Example: `"User reported slow login (3+ seconds) on mobile devices"`

### 3. Relationship Types

- Use clear, standardized relationship types
- Common types: `implements`, `depends_on`, `contains`, `relates_to`
- Be consistent across your application

### 4. Memory Bank Organization

- Create banks for different contexts/projects
- Use descriptive bank names
- Switch banks appropriately for context

### 5. Error Recovery

- Always check for errors in MCP responses
- Implement retry logic for transient failures
- Have fallback strategies for missing data

## Example: Complete Integration

Here's a complete example of an AI agent using the Graph Memory MCP Server:

```python
class MemoryAwareAgent:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.current_conversation = []
    
    def process_user_input(self, user_input):
        # Store user input as entity
        self.store_user_input(user_input)
        
        # Retrieve relevant context
        context = self.get_relevant_context(user_input)
        
        # Generate response using context
        response = self.generate_response(user_input, context)
        
        # Store reasoning steps
        self.store_reasoning(user_input, response)
        
        return response
    
    def store_user_input(self, user_input):
        # Extract key concepts
        concepts = self.extract_concepts(user_input)
        
        # Store as entities
        self.mcp_client.call_tool("create_entities", {
            "entities": [
                {
                    "name": f"user-input-{timestamp}",
                    "entityType": "user_input",
                    "observations": [user_input, f"Received at {timestamp}"]
                }
            ] + [
                {
                    "name": concept,
                    "entityType": "concept",
                    "observations": [f"Mentioned by user at {timestamp}"]
                }
                for concept in concepts
            ]
        })
    
    def store_reasoning(self, input_text, response):
        self.mcp_client.call_tool("sequential_thinking", {
            "step": {
                "thought": f"Processed user input: {input_text[:100]}...",
                "step_number": len(self.current_conversation) + 1,
                "reasoning": f"Generated response based on context analysis"
            }
        })
```

## VS Code Agent Chat Integration

When using with VS Code Agent Chat, the MCP server automatically handles protocol negotiation. Your agent just needs to:

1. Connect to the server via the MCP configuration
2. Call tools as needed
3. Handle responses appropriately

The server will manage persistence and memory across VS Code sessions.

## Troubleshooting

### Connection Issues

- Verify server is running: `curl http://localhost:10642/`
- Check MCP configuration in VS Code
- Review server logs: `podman logs graph-mcp-server`

### Tool Call Failures

- Validate input schemas carefully
- Check entity names exist before creating relations
- Handle missing entities gracefully

### Performance Considerations

- Use memory banks to partition large datasets
- Batch multiple operations when possible
- Monitor memory usage for large graphs

## Related Documentation

- [API Reference](./API.md) - Complete REST API documentation
- [Deployment Guide](./DEPLOYMENT.md) - Container deployment
- [VS Code Setup](./VS_CODE_SETUP.md) - Configuration guide
