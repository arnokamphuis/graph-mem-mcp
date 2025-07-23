TOOL_LIST = [
    {
        "name": "create_entities",
        "description": "Create multiple new entities in the knowledge graph with optional auto-extraction of additional entities and relationships from observations.",
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
                    "description": "Whether to automatically extract additional entities and relationships from observation text (default: false - observations should be descriptive)",
                    "default": False
                }
            }
        }
    },
    {
        "name": "add_observations",
        "description": "Add new observations to existing entities.",
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
        "name": "ingest_knowledge_enhanced",
        "description": "Create a high-quality knowledge graph using enhanced processing with LLM concepts, semantic clustering, and quality filtering",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Large text content to analyze"},
                "bank": {"type": "string", "description": "Memory bank name"},
                "source": {"type": "string", "description": "Source identifier"},
                "create_observations": {"type": "boolean", "description": "Create observations"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "search_nodes",
        "description": "Search for nodes in the knowledge graph based on a query.",
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
        "description": "Read the entire knowledge graph summary for the current memory bank.",
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
        "description": "Create a new memory bank for organizing different topics/projects.",
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
        "description": "Switch to a different memory bank.",
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
        "description": "List all available memory banks with their statistics.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "delete_bank",
        "description": "Delete a memory bank and all its contents.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "bank": {"type": "string", "description": "Name of the memory bank to delete"}
            },
            "required": ["bank"]
        }
    }
]
