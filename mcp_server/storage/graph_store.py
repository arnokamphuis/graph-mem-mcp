"""
Graph storage abstraction layer providing unified interface for different storage backends.

This module defines the abstract storage interface and configuration classes
following Phase 3.2 Performance Optimization requirements.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple, Union, TYPE_CHECKING
from enum import Enum

# Import Phase 1 core components with proper type checking
if TYPE_CHECKING:
    from ..core.graph_schema import EntityInstance, RelationshipInstance, SchemaManager

try:
    from ..core.graph_schema import EntityInstance, RelationshipInstance, SchemaManager
    CORE_SCHEMA_AVAILABLE = True
except ImportError:
    EntityInstance = Any
    RelationshipInstance = Any
    SchemaManager = Any
    CORE_SCHEMA_AVAILABLE = False


class StorageBackend(Enum):
    """Supported storage backend types"""
    MEMORY = "memory"
    NEO4J = "neo4j"
    ARANGO = "arango"  # Future extension


@dataclass
class StorageConfig:
    """Configuration for graph storage backends"""
    backend: StorageBackend = StorageBackend.MEMORY
    connection_uri: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database_name: Optional[str] = "knowledge_graph"
    
    # Performance settings
    batch_size: int = 1000
    enable_indexing: bool = True
    enable_caching: bool = True
    cache_size: int = 10000
    
    # Query optimization
    query_timeout: int = 30  # seconds
    max_query_complexity: int = 1000
    enable_query_optimization: bool = True
    
    # Persistence settings
    auto_commit: bool = True
    commit_interval: int = 100
    backup_enabled: bool = False
    backup_interval: int = 3600  # seconds


@dataclass
class QueryResult:
    """Result container for graph queries"""
    entities: List[Any] = field(default_factory=list)
    relationships: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    query_complexity: int = 0
    cache_hit: bool = False


@dataclass
class IndexInfo:
    """Information about storage indexes"""
    name: str
    type: str  # "entity", "relationship", "property"
    properties: List[str]
    unique: bool = False
    created_at: Optional[str] = None
    size_bytes: int = 0


class GraphStore(ABC):
    """
    Abstract base class for graph storage backends.
    
    Provides unified interface for storing, querying, and managing knowledge graphs
    with performance optimization features including indexing, caching, and query optimization.
    """
    
    def __init__(self, config: StorageConfig):
        """Initialize storage with configuration"""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._is_connected = False
        self._schema_manager: Optional[Any] = None
        
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to storage backend.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to storage backend"""
        pass
    
    @abstractmethod
    async def create_entity(self, entity: Any) -> bool:
        """
        Create a new entity in storage.
        
        Args:
            entity: Entity instance to create
            
        Returns:
            True if creation successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def create_relationship(self, relationship: Any) -> bool:
        """
        Create a new relationship in storage.
        
        Args:
            relationship: Relationship instance to create
            
        Returns:
            True if creation successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[Any]:
        """
        Retrieve entity by ID.
        
        Args:
            entity_id: Unique identifier for the entity
            
        Returns:
            EntityInstance if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_relationship(self, relationship_id: str) -> Optional[Any]:
        """
        Retrieve relationship by ID.
        
        Args:
            relationship_id: Unique identifier for the relationship
            
        Returns:
            RelationshipInstance if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update entity properties.
        
        Args:
            entity_id: Unique identifier for the entity
            updates: Dictionary of property updates
            
        Returns:
            True if update successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def update_relationship(self, relationship_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update relationship properties.
        
        Args:
            relationship_id: Unique identifier for the relationship
            updates: Dictionary of property updates
            
        Returns:
            True if update successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete_entity(self, entity_id: str, cascade: bool = False) -> bool:
        """
        Delete entity from storage.
        
        Args:
            entity_id: Unique identifier for the entity
            cascade: Whether to delete connected relationships
            
        Returns:
            True if deletion successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete_relationship(self, relationship_id: str) -> bool:
        """
        Delete relationship from storage.
        
        Args:
            relationship_id: Unique identifier for the relationship
            
        Returns:
            True if deletion successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def query_entities(self, 
                           entity_type: Optional[str] = None,
                           properties: Optional[Dict[str, Any]] = None,
                           limit: Optional[int] = None) -> List[Any]:
        """
        Query entities with filtering criteria.
        
        Args:
            entity_type: Filter by entity type
            properties: Filter by property values
            limit: Maximum number of results
            
        Returns:
            List of matching entities
        """
        pass
    
    @abstractmethod
    async def query_relationships(self,
                                relationship_type: Optional[str] = None,
                                source_entity_id: Optional[str] = None,
                                target_entity_id: Optional[str] = None,
                                properties: Optional[Dict[str, Any]] = None,
                                limit: Optional[int] = None) -> List[Any]:
        """
        Query relationships with filtering criteria.
        
        Args:
            relationship_type: Filter by relationship type
            source_entity_id: Filter by source entity
            target_entity_id: Filter by target entity
            properties: Filter by property values
            limit: Maximum number of results
            
        Returns:
            List of matching relationships
        """
        pass
    
    @abstractmethod
    async def find_path(self, 
                       source_id: str, 
                       target_id: str, 
                       max_depth: int = 5,
                       relationship_types: Optional[List[str]] = None) -> List[List[str]]:
        """
        Find paths between two entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_depth: Maximum path length
            relationship_types: Filter by relationship types
            
        Returns:
            List of paths (each path is list of entity IDs)
        """
        pass
    
    @abstractmethod
    async def get_neighbors(self, 
                          entity_id: str, 
                          relationship_types: Optional[List[str]] = None,
                          direction: str = "both") -> List[Any]:
        """
        Get neighboring entities.
        
        Args:
            entity_id: Central entity ID
            relationship_types: Filter by relationship types
            direction: "incoming", "outgoing", or "both"
            
        Returns:
            List of neighboring entities
        """
        pass
    
    # Performance optimization methods
    @abstractmethod
    async def create_index(self, 
                          index_name: str,
                          target_type: str,  # "entity" or "relationship"
                          properties: List[str],
                          unique: bool = False) -> bool:
        """
        Create index for performance optimization.
        
        Args:
            index_name: Name of the index
            target_type: Type of objects to index
            properties: Properties to index
            unique: Whether index should enforce uniqueness
            
        Returns:
            True if index creation successful
        """
        pass
    
    @abstractmethod
    async def drop_index(self, index_name: str) -> bool:
        """
        Drop an existing index.
        
        Args:
            index_name: Name of the index to drop
            
        Returns:
            True if index removal successful
        """
        pass
    
    @abstractmethod
    async def list_indexes(self) -> List[IndexInfo]:
        """
        List all available indexes.
        
        Returns:
            List of index information objects
        """
        pass
    
    # Bulk operations for performance
    @abstractmethod
    async def bulk_create_entities(self, entities: List[Any]) -> int:
        """
        Create multiple entities in batch.
        
        Args:
            entities: List of entities to create
            
        Returns:
            Number of entities successfully created
        """
        pass
    
    @abstractmethod
    async def bulk_create_relationships(self, relationships: List[Any]) -> int:
        """
        Create multiple relationships in batch.
        
        Args:
            relationships: List of relationships to create
            
        Returns:
            Number of relationships successfully created
        """
        pass
    
    # Statistics and monitoring
    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics and metrics.
        
        Returns:
            Dictionary containing storage statistics
        """
        pass
    
    @abstractmethod
    async def clear_cache(self) -> None:
        """Clear internal caches"""
        pass
    
    # Transaction support
    @abstractmethod
    async def begin_transaction(self) -> str:
        """
        Begin a transaction.
        
        Returns:
            Transaction ID
        """
        pass
    
    @abstractmethod
    async def commit_transaction(self, transaction_id: str) -> bool:
        """
        Commit a transaction.
        
        Args:
            transaction_id: Transaction identifier
            
        Returns:
            True if commit successful
        """
        pass
    
    @abstractmethod
    async def rollback_transaction(self, transaction_id: str) -> bool:
        """
        Rollback a transaction.
        
        Args:
            transaction_id: Transaction identifier
            
        Returns:
            True if rollback successful
        """
        pass
    
    # Schema management integration
    def set_schema_manager(self, schema_manager: Any) -> None:
        """Set schema manager for validation"""
        self._schema_manager = schema_manager
    
    def get_schema_manager(self) -> Optional[Any]:
        """Get current schema manager"""
        return self._schema_manager
    
    # Connection status
    @property
    def is_connected(self) -> bool:
        """Check if storage is connected"""
        return self._is_connected
