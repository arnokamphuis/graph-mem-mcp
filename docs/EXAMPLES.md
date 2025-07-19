# Examples and Use Cases

Complete examples demonstrating how AI agents can use the Graph Memory MCP Server.

## Example 1: Building a Project Knowledge Base

This example shows how an AI agent can build and maintain a comprehensive knowledge base for a software project.

### Scenario: Web Application Development

**User Input:** "I need to build a web application with user authentication, file upload, and real-time chat features."

**Agent Response and Actions:**

#### Step 1: Create Core Project Entities

```json
{
  "tool": "create_entities",
  "arguments": {
    "entities": [
      {
        "name": "web-application-project",
        "entityType": "project",
        "observations": [
          "Main project for web application development",
          "Started on 2025-01-19",
          "Requirements include authentication, file upload, and chat"
        ]
      },
      {
        "name": "user-authentication",
        "entityType": "feature",
        "observations": [
          "Required feature for user management",
          "Should include login, registration, password reset",
          "Security is critical"
        ]
      },
      {
        "name": "file-upload-system",
        "entityType": "feature",
        "observations": [
          "Allow users to upload files",
          "Need to handle multiple file types",
          "Security scanning required"
        ]
      },
      {
        "name": "real-time-chat",
        "entityType": "feature",
        "observations": [
          "Real-time messaging between users",
          "WebSocket implementation needed",
          "Message persistence required"
        ]
      }
    ]
  }
}
```

#### Step 2: Create Relationships

```json
{
  "tool": "create_relations",
  "arguments": {
    "relations": [
      {
        "from": "web-application-project",
        "to": "user-authentication",
        "relationType": "includes"
      },
      {
        "from": "web-application-project",
        "to": "file-upload-system",
        "relationType": "includes"
      },
      {
        "from": "web-application-project",
        "to": "real-time-chat",
        "relationType": "includes"
      },
      {
        "from": "file-upload-system",
        "to": "user-authentication",
        "relationType": "depends_on"
      },
      {
        "from": "real-time-chat",
        "to": "user-authentication",
        "relationType": "depends_on"
      }
    ]
  }
}
```

#### Step 3: Add Technical Analysis

**User Input:** "I think we should use React for the frontend and Node.js for the backend."

```json
{
  "tool": "create_entities",
  "arguments": {
    "entities": [
      {
        "name": "react-frontend",
        "entityType": "technology",
        "observations": [
          "Modern JavaScript library for UI",
          "Component-based architecture",
          "Good ecosystem for web development",
          "Chosen for frontend implementation"
        ]
      },
      {
        "name": "nodejs-backend",
        "entityType": "technology",
        "observations": [
          "JavaScript runtime for backend",
          "Express.js framework recommended",
          "Good for real-time applications",
          "Chosen for backend implementation"
        ]
      }
    ]
  }
}
```

```json
{
  "tool": "create_relations",
  "arguments": {
    "relations": [
      {
        "from": "web-application-project",
        "to": "react-frontend",
        "relationType": "uses"
      },
      {
        "from": "web-application-project",
        "to": "nodejs-backend",
        "relationType": "uses"
      },
      {
        "from": "real-time-chat",
        "to": "nodejs-backend",
        "relationType": "implemented_with"
      }
    ]
  }
}
```

#### Step 4: Document Reasoning Process

```json
{
  "tool": "sequential_thinking",
  "arguments": {
    "step": {
      "thought": "Analyzing technology stack for web application",
      "step_number": 1,
      "reasoning": "React provides component reusability and state management. Node.js enables full-stack JavaScript and has excellent WebSocket support for real-time chat feature."
    }
  }
}
```

#### Step 5: Add Updated Requirements

**User Input:** "Actually, let's add OAuth2 support and make the file uploads go to cloud storage."

```json
{
  "tool": "add_observations",
  "arguments": {
    "observations": [
      {
        "entityName": "user-authentication",
        "contents": [
          "Updated to include OAuth2 support",
          "Should support Google and GitHub OAuth",
          "JWT tokens for session management"
        ]
      },
      {
        "entityName": "file-upload-system",
        "contents": [
          "Files stored in cloud storage (AWS S3 or similar)",
          "Presigned URLs for secure uploads",
          "Automatic virus scanning"
        ]
      }
    ]
  }
}
```

```json
{
  "tool": "create_entities",
  "arguments": {
    "entities": [
      {
        "name": "oauth2-integration",
        "entityType": "technology",
        "observations": [
          "Third-party authentication system",
          "Supports Google and GitHub providers",
          "Reduces password management burden"
        ]
      },
      {
        "name": "cloud-storage",
        "entityType": "service",
        "observations": [
          "AWS S3 or equivalent cloud storage",
          "Scalable file storage solution",
          "Integrates with CDN for fast delivery"
        ]
      }
    ]
  }
}
```

## Example 2: Debugging and Problem Solving

This example shows how to track debugging processes and solutions.

### Scenario: Performance Issue Investigation

**User Input:** "The chat feature is really slow. Messages take 3-4 seconds to appear."

#### Step 1: Document the Problem

```json
{
  "tool": "create_entities",
  "arguments": {
    "entities": [
      {
        "name": "chat-performance-issue",
        "entityType": "bug",
        "observations": [
          "Messages taking 3-4 seconds to appear",
          "Reported by user on 2025-01-19",
          "Affects real-time chat functionality",
          "Critical performance issue"
        ]
      }
    ]
  }
}
```

```json
{
  "tool": "create_relations",
  "arguments": {
    "relations": [
      {
        "from": "chat-performance-issue",
        "to": "real-time-chat",
        "relationType": "affects"
      }
    ]
  }
}
```

#### Step 2: Track Investigation Steps

```json
{
  "tool": "sequential_thinking",
  "arguments": {
    "step": {
      "thought": "Starting performance investigation for chat delays",
      "step_number": 1,
      "reasoning": "Need to check WebSocket connection, database queries, and message processing pipeline. Will start with network latency analysis."
    }
  }
}
```

**After investigation:**

```json
{
  "tool": "sequential_thinking",
  "arguments": {
    "step": {
      "thought": "Found database query performance issue",
      "step_number": 2,
      "reasoning": "Chat message retrieval is doing full table scan without proper indexing. Each message load queries entire chat history."
    }
  }
}
```

#### Step 3: Document the Solution

```json
{
  "tool": "create_entities",
  "arguments": {
    "entities": [
      {
        "name": "database-indexing-solution",
        "entityType": "solution",
        "observations": [
          "Add composite index on (chat_room_id, timestamp)",
          "Limit message history queries to last 50 messages",
          "Implement pagination for older messages",
          "Expected improvement: <500ms response time"
        ]
      }
    ]
  }
}
```

```json
{
  "tool": "create_relations",
  "arguments": {
    "relations": [
      {
        "from": "database-indexing-solution",
        "to": "chat-performance-issue",
        "relationType": "solves"
      },
      {
        "from": "database-indexing-solution",
        "to": "real-time-chat",
        "relationType": "improves"
      }
    ]
  }
}
```

#### Step 4: Track Implementation

```json
{
  "tool": "add_observations",
  "arguments": {
    "observations": [
      {
        "entityName": "database-indexing-solution",
        "contents": [
          "Solution implemented on 2025-01-19",
          "Performance improved to 200ms average",
          "User feedback: chat now feels responsive",
          "Monitoring shows 95% of messages under 300ms"
        ]
      }
    ]
  }
}
```

## Example 3: Learning and Knowledge Accumulation

This example shows how to build domain knowledge over time.

### Scenario: Learning Web Security Best Practices

#### Session 1: Initial Learning

**User Input:** "What are the most important security considerations for web applications?"

```json
{
  "tool": "create_entities",
  "arguments": {
    "entities": [
      {
        "name": "web-security-fundamentals",
        "entityType": "knowledge_domain",
        "observations": [
          "Critical area for web application development",
          "Includes authentication, authorization, data protection",
          "Constantly evolving threat landscape"
        ]
      },
      {
        "name": "owasp-top-10",
        "entityType": "security_framework",
        "observations": [
          "Standard reference for web security vulnerabilities",
          "Updated regularly by security community",
          "Essential knowledge for developers"
        ]
      },
      {
        "name": "sql-injection",
        "entityType": "vulnerability",
        "observations": [
          "Most common web application attack",
          "Occurs when user input not properly sanitized",
          "Can lead to data breach"
        ]
      }
    ]
  }
}
```

#### Session 2: Deeper Learning

**User Input:** "Tell me more about protecting against CSRF attacks."

```json
{
  "tool": "create_entities",
  "arguments": {
    "entities": [
      {
        "name": "csrf-protection",
        "entityType": "security_measure",
        "observations": [
          "Cross-Site Request Forgery protection",
          "Uses anti-CSRF tokens in forms",
          "Validates requests with token verification",
          "Essential for state-changing operations"
        ]
      }
    ]
  }
}
```

```json
{
  "tool": "create_relations",
  "arguments": {
    "relations": [
      {
        "from": "csrf-protection",
        "to": "web-security-fundamentals",
        "relationType": "part_of"
      },
      {
        "from": "user-authentication",
        "to": "csrf-protection",
        "relationType": "requires"
      }
    ]
  }
}
```

```json
{
  "tool": "add_observations",
  "arguments": {
    "observations": [
      {
        "entityName": "web-security-fundamentals",
        "contents": [
          "Learned about CSRF protection mechanisms",
          "CSRF tokens prevent unauthorized state changes",
          "Must be implemented in all forms and AJAX requests"
        ]
      }
    ]
  }
}
```

## Example 4: Multi-Bank Project Organization

This example shows how to use memory banks for different projects.

### Setting Up Project-Specific Memory

```bash
# Switch to project-specific bank
curl -X POST http://localhost:8000/banks/create \
  -H "Content-Type: application/json" \
  -d '{"bank": "ecommerce-project"}'

curl -X POST http://localhost:8000/banks/select \
  -H "Content-Type: application/json" \
  -d '{"bank": "ecommerce-project"}'
```

### Ecommerce Project Knowledge

```json
{
  "tool": "create_entities",
  "arguments": {
    "entities": [
      {
        "name": "ecommerce-platform",
        "entityType": "project",
        "observations": [
          "Online store with product catalog",
          "Shopping cart and checkout process",
          "Payment processing integration",
          "Order management system"
        ]
      },
      {
        "name": "product-catalog",
        "entityType": "feature",
        "observations": [
          "Display products with images and descriptions",
          "Search and filtering capabilities",
          "Category organization",
          "Price management"
        ]
      },
      {
        "name": "payment-gateway",
        "entityType": "integration",
        "observations": [
          "Stripe integration for card processing",
          "PayPal for alternative payments",
          "PCI compliance requirements",
          "Fraud detection needed"
        ]
      }
    ]
  }
}
```

### Switching Between Projects

```bash
# Work on different project
curl -X POST http://localhost:8000/banks/select \
  -H "Content-Type: application/json" \
  -d '{"bank": "default"}'
```

Now MCP tools will operate on the original web application project, keeping knowledge separate.

## Example 5: Complex Reasoning Chain

This example shows tracking multi-step problem solving.

### Scenario: System Architecture Design

**User Input:** "How should I design the microservices architecture for scalability?"

#### Step-by-step reasoning:

```json
{
  "tool": "sequential_thinking",
  "arguments": {
    "step": {
      "thought": "Starting microservices architecture analysis",
      "step_number": 1,
      "reasoning": "Need to identify service boundaries based on business domains. Will use Domain-Driven Design principles to separate concerns."
    }
  }
}
```

```json
{
  "tool": "sequential_thinking",
  "arguments": {
    "step": {
      "thought": "Identified core service domains",
      "step_number": 2,
      "reasoning": "User service for authentication, Product service for catalog, Order service for transactions, Notification service for messaging. Each has distinct data and business logic."
    }
  }
}
```

```json
{
  "tool": "sequential_thinking",
  "arguments": {
    "step": {
      "thought": "Analyzing inter-service communication patterns",
      "step_number": 3,
      "reasoning": "Synchronous REST for real-time queries, async messaging for events. Event-driven architecture for loose coupling between services."
    }
  }
}
```

```json
{
  "tool": "create_entities",
  "arguments": {
    "entities": [
      {
        "name": "microservices-architecture",
        "entityType": "architecture",
        "observations": [
          "Service-oriented architecture for scalability",
          "Domain-driven service boundaries",
          "Event-driven communication pattern",
          "Independent deployment and scaling"
        ]
      },
      {
        "name": "user-service",
        "entityType": "microservice",
        "observations": [
          "Handles authentication and user management",
          "JWT token generation and validation",
          "User profile and preferences",
          "Separate database for user data"
        ]
      }
    ]
  }
}
```

## Common Patterns and Best Practices

### 1. Hierarchical Knowledge Organization

Create parent-child relationships for organized knowledge:

```json
{
  "relations": [
    {
      "from": "web-development",
      "to": "frontend-development",
      "relationType": "contains"
    },
    {
      "from": "frontend-development",
      "to": "react-development",
      "relationType": "contains"
    }
  ]
}
```

### 2. Temporal Tracking

Include timestamps and version information:

```json
{
  "observations": [
    "Feature implemented on 2025-01-19",
    "Version 1.0 - basic functionality",
    "Version 1.1 - added error handling on 2025-01-20"
  ]
}
```

### 3. Cross-Reference Patterns

Link related concepts across domains:

```json
{
  "relations": [
    {
      "from": "authentication-bug",
      "to": "oauth2-integration",
      "relationType": "related_to"
    },
    {
      "from": "performance-issue",
      "to": "database-optimization",
      "relationType": "solved_by"
    }
  ]
}
```

### 4. Evidence-Based Observations

Include sources and evidence:

```json
{
  "observations": [
    "Performance test shows 200ms average response time",
    "User feedback: 95% satisfaction rate",
    "Code review passed - no security issues found",
    "Production metrics: 99.9% uptime"
  ]
}
```

## Error Handling Examples

### Handling Missing Entities

```python
# Before creating relations, check entities exist
response = requests.get(f"{base_url}/entities")
entities = [entity["id"] for entity in response.json()]

if "source-entity" not in entities:
    # Create the missing entity first
    mcp_client.call_tool("create_entities", {
        "entities": [{
            "name": "source-entity",
            "entityType": "concept",
            "observations": ["Created to establish relationship"]
        }]
    })
```

### Graceful Degradation

```python
def safe_knowledge_update(entity_name, observation):
    try:
        mcp_client.call_tool("add_observations", {
            "observations": [{
                "entityName": entity_name,
                "contents": [observation]
            }]
        })
    except Exception as e:
        # Fallback: create new entity
        mcp_client.call_tool("create_entities", {
            "entities": [{
                "name": entity_name,
                "entityType": "auto_created",
                "observations": [observation, "Auto-created due to missing entity"]
            }]
        })
```

## Integration with Other Tools

### Export to External Systems

```python
# Export knowledge graph to external analysis tools
def export_to_graphml():
    response = requests.get(f"{base_url}/context/retrieve")
    data = response.json()
    
    # Convert to GraphML format
    # ... export logic
```

### Backup and Sync

```python
# Regular backup of important entities
def backup_critical_knowledge():
    critical_entities = ["project-requirements", "architecture-decisions"]
    
    for entity in critical_entities:
        # Retrieve and backup
        pass
```

These examples demonstrate the full capabilities of the Graph Memory MCP Server for AI agents to build, maintain, and utilize persistent knowledge across conversations and sessions.
