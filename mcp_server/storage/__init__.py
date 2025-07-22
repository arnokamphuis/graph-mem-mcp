"""
Storage abstraction layer for knowledge graph persistence and optimization.

This module provides a unified interface for different storage backends including
in-memory storage, Neo4j graph database, and other persistence mechanisms.

Phase 3.2 Performance Optimization Components:
- GraphStore: Abstract storage interface
- MemoryStore: In-memory storage implementation  
- Neo4jAdapter: Neo4j database integration
- Indexing and query optimization utilities
"""

from typing import TYPE_CHECKING

# Import storage interfaces
from .graph_store import GraphStore, StorageConfig
from .memory_store import MemoryStore

# Conditional imports for optional dependencies
try:
    from .neo4j_adapter import Neo4jAdapter
    NEO4J_AVAILABLE = True
except ImportError:
    Neo4j_adapter = None
    NEO4J_AVAILABLE = False

# Factory functions following established patterns
def create_graph_store(storage_type: str = "memory", **config) -> GraphStore:
    """
    Factory function to create appropriate graph store instance.
    
    Args:
        storage_type: Type of storage backend ("memory", "neo4j")
        **config: Configuration parameters for the storage backend
        
    Returns:
        GraphStore instance configured for the specified backend
    """
    if storage_type == "memory":
        return MemoryStore(**config)
    elif storage_type == "neo4j" and NEO4J_AVAILABLE:
        return Neo4jAdapter(**config)
    elif storage_type == "neo4j" and not NEO4J_AVAILABLE:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Neo4j not available, falling back to memory storage")
        return MemoryStore(**config)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")

def create_memory_store(**config) -> MemoryStore:
    """Factory function to create in-memory storage."""
    return MemoryStore(**config)

# Export all public interfaces
__all__ = [
    "GraphStore",
    "StorageConfig", 
    "MemoryStore",
    "create_graph_store",
    "create_memory_store",
    "NEO4J_AVAILABLE"
]

# Add Neo4j exports if available
if NEO4J_AVAILABLE:
    __all__.extend(["Neo4jAdapter"])
