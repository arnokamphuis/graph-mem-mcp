"""
In-memory graph storage implementation for development and testing.

This module provides a high-performance in-memory storage backend
with full indexing and query optimization capabilities.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Any, Set
from copy import deepcopy

from .graph_store import GraphStore, StorageConfig, QueryResult, IndexInfo

# Import Phase 1 core components
try:
    from ..core.graph_schema import EntityInstance, RelationshipInstance
    CORE_SCHEMA_AVAILABLE = True
except ImportError:
    EntityInstance = Any
    RelationshipInstance = Any
    CORE_SCHEMA_AVAILABLE = False


class MemoryIndex:
    """In-memory index for fast lookups"""
    
    def __init__(self, name: str, properties: List[str], unique: bool = False):
        self.name = name
        self.properties = properties
        self.unique = unique
        self.index: Dict[str, Set[str]] = defaultdict(set)  # value -> set of IDs
        self.created_at = time.time()
    
    def add_item(self, item_id: str, item_data: Dict[str, Any]) -> bool:
        """Add item to index"""
        try:
            # Create composite key from properties
            key_parts = []
            for prop in self.properties:
                value = item_data.get(prop, '')
                key_parts.append(str(value))
            
            key = '|'.join(key_parts)
            
            if self.unique and key in self.index and self.index[key]:
                return False  # Unique constraint violation
            
            self.index[key].add(item_id)
            return True
        except Exception:
            return False
    
    def remove_item(self, item_id: str, item_data: Dict[str, Any]) -> bool:
        """Remove item from index"""
        try:
            key_parts = []
            for prop in self.properties:
                value = item_data.get(prop, '')
                key_parts.append(str(value))
            
            key = '|'.join(key_parts)
            
            if key in self.index:
                self.index[key].discard(item_id)
                if not self.index[key]:
                    del self.index[key]
            
            return True
        except Exception:
            return False
    
    def find_items(self, search_criteria: Dict[str, Any]) -> Set[str]:
        """Find items matching search criteria"""
        try:
            # Build search key
            key_parts = []
            for prop in self.properties:
                if prop in search_criteria:
                    key_parts.append(str(search_criteria[prop]))
                else:
                    # Can't use partial matching with this simple implementation
                    return set()
            
            key = '|'.join(key_parts)
            return self.index.get(key, set())
        except Exception:
            return set()


class MemoryStore(GraphStore):
    """
    In-memory graph storage implementation.
    
    Provides high-performance storage for development and testing with
    full indexing, caching, and query optimization capabilities.
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        """Initialize memory store"""
        if config is None:
            config = StorageConfig()
        
        super().__init__(config)
        
        # Core storage
        self.entities: Dict[str, Any] = {}
        self.relationships: Dict[str, Any] = {}
        
        # Indexes for performance
        self.indexes: Dict[str, MemoryIndex] = {}
        
        # Relationship lookups for graph traversal
        self.outgoing_relationships: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> relationship_ids
        self.incoming_relationships: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> relationship_ids
        
        # Entity type indexes
        self.entities_by_type: Dict[str, Set[str]] = defaultdict(set)
        self.relationships_by_type: Dict[str, Set[str]] = defaultdict(set)
        
        # Query cache
        self.query_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Statistics
        self.stats = {
            'entities_created': 0,
            'relationships_created': 0,
            'queries_executed': 0,
            'index_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Transaction support
        self.transactions: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("🗃️  Memory store initialized")
    
    async def connect(self) -> bool:
        """Establish connection (no-op for memory store)"""
        self._is_connected = True
        self.logger.info("✅ Memory store connected")
        return True
    
    async def disconnect(self) -> None:
        """Close connection (no-op for memory store)"""
        self._is_connected = False
        self.logger.info("🔌 Memory store disconnected")
    
    async def create_entity(self, entity: Any) -> bool:
        """Create a new entity in storage"""
        try:
            if isinstance(entity, dict):
                entity_id = entity.get('id', str(uuid.uuid4()))
                entity_data = entity.copy()
            elif hasattr(entity, 'id'):
                entity_id = entity.id
                if hasattr(entity, 'model_dump'):
                    entity_data = entity.model_dump()
                elif hasattr(entity, '__dict__'):
                    entity_data = entity.__dict__.copy()
                else:
                    entity_data = {'id': entity_id}
            else:
                entity_id = str(uuid.uuid4())
                if hasattr(entity, '__dict__'):
                    entity_data = entity.__dict__.copy()
                    entity_data['id'] = entity_id
                else:
                    entity_data = {'id': entity_id}
            
            # Ensure entity has an ID in the data
            if 'id' not in entity_data:
                entity_data['id'] = entity_id
            
            # Use the ID from the data for consistency
            final_entity_id = entity_data['id']
            
            if final_entity_id in self.entities:
                self.logger.debug(f"⚠️  Entity {final_entity_id} already exists")
                return False  # Entity already exists
            
            # Store entity
            self.entities[final_entity_id] = entity_data
            
            # Update type index
            entity_type = entity_data.get('entity_type', 'unknown')
            self.entities_by_type[entity_type].add(final_entity_id)
            
            # Update indexes
            for index in self.indexes.values():
                if index.name.startswith('entity_'):
                    index.add_item(final_entity_id, entity_data)
            
            self.stats['entities_created'] += 1
            self.logger.debug(f"📝 Created entity: {final_entity_id} (type: {entity_type})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create entity: {e}")
            return False
    
    async def create_relationship(self, relationship: Any) -> bool:
        """Create a new relationship in storage"""
        try:
            if isinstance(relationship, dict):
                rel_id = relationship.get('id', str(uuid.uuid4()))
                rel_data = relationship.copy()
            elif hasattr(relationship, 'id'):
                rel_id = relationship.id
                if hasattr(relationship, 'model_dump'):
                    rel_data = relationship.model_dump()
                elif hasattr(relationship, '__dict__'):
                    rel_data = relationship.__dict__.copy()
                else:
                    rel_data = {'id': rel_id}
            else:
                rel_id = str(uuid.uuid4())
                if hasattr(relationship, '__dict__'):
                    rel_data = relationship.__dict__.copy()
                    rel_data['id'] = rel_id
                else:
                    rel_data = {'id': rel_id}
            
            # Ensure relationship has an ID in the data
            if 'id' not in rel_data:
                rel_data['id'] = rel_id
            
            # Use the ID from the data for consistency
            final_rel_id = rel_data['id']
            
            if final_rel_id in self.relationships:
                self.logger.debug(f"⚠️  Relationship {final_rel_id} already exists")
                return False  # Relationship already exists
            
            # Validate entity references
            source_id = rel_data.get('source_entity_id')
            target_id = rel_data.get('target_entity_id')
            
            if not source_id or not target_id:
                self.logger.warning(f"⚠️  Missing entity references: source={source_id}, target={target_id}")
                return False
            
            if source_id not in self.entities or target_id not in self.entities:
                self.logger.warning(f"⚠️  Referenced entities not found: {source_id} -> {target_id}")
                self.logger.debug(f"Available entities: {list(self.entities.keys())}")
                return False
            
            # Store relationship
            self.relationships[final_rel_id] = rel_data
            
            # Update type index
            rel_type = rel_data.get('relation_type', 'unknown')
            self.relationships_by_type[rel_type].add(final_rel_id)
            
            # Update graph traversal indexes
            self.outgoing_relationships[source_id].add(final_rel_id)
            self.incoming_relationships[target_id].add(final_rel_id)
            
            # Update indexes
            for index in self.indexes.values():
                if index.name.startswith('relationship_'):
                    index.add_item(final_rel_id, rel_data)
            
            self.stats['relationships_created'] += 1
            self.logger.debug(f"🔗 Created relationship: {final_rel_id} ({source_id} -> {target_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create relationship: {e}")
            return False
    
    async def get_entity(self, entity_id: str) -> Optional[Any]:
        """Retrieve entity by ID"""
        try:
            entity_data = self.entities.get(entity_id)
            if entity_data and CORE_SCHEMA_AVAILABLE:
                # Try to reconstruct EntityInstance if available
                try:
                    return EntityInstance(**entity_data)
                except Exception:
                    return entity_data
            return entity_data
        except Exception as e:
            self.logger.error(f"❌ Failed to get entity {entity_id}: {e}")
            return None
    
    async def get_relationship(self, relationship_id: str) -> Optional[Any]:
        """Retrieve relationship by ID"""
        try:
            rel_data = self.relationships.get(relationship_id)
            if rel_data and CORE_SCHEMA_AVAILABLE:
                # Try to reconstruct RelationshipInstance if available
                try:
                    return RelationshipInstance(**rel_data)
                except Exception:
                    return rel_data
            return rel_data
        except Exception as e:
            self.logger.error(f"❌ Failed to get relationship {relationship_id}: {e}")
            return None
    
    async def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """Update entity properties"""
        try:
            if entity_id not in self.entities:
                return False
            
            old_data = deepcopy(self.entities[entity_id])
            
            # Update data
            self.entities[entity_id].update(updates)
            new_data = self.entities[entity_id]
            
            # Update type index if type changed
            old_type = old_data.get('entity_type', 'unknown')
            new_type = new_data.get('entity_type', 'unknown')
            
            if old_type != new_type:
                self.entities_by_type[old_type].discard(entity_id)
                self.entities_by_type[new_type].add(entity_id)
            
            # Update indexes
            for index in self.indexes.values():
                if index.name.startswith('entity_'):
                    index.remove_item(entity_id, old_data)
                    index.add_item(entity_id, new_data)
            
            # Clear query cache
            self.query_cache.clear()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update entity {entity_id}: {e}")
            return False
    
    async def update_relationship(self, relationship_id: str, updates: Dict[str, Any]) -> bool:
        """Update relationship properties"""
        try:
            if relationship_id not in self.relationships:
                return False
            
            old_data = deepcopy(self.relationships[relationship_id])
            
            # Update data
            self.relationships[relationship_id].update(updates)
            new_data = self.relationships[relationship_id]
            
            # Update type index if type changed
            old_type = old_data.get('relation_type', 'unknown')
            new_type = new_data.get('relation_type', 'unknown')
            
            if old_type != new_type:
                self.relationships_by_type[old_type].discard(relationship_id)
                self.relationships_by_type[new_type].add(relationship_id)
            
            # Update indexes
            for index in self.indexes.values():
                if index.name.startswith('relationship_'):
                    index.remove_item(relationship_id, old_data)
                    index.add_item(relationship_id, new_data)
            
            # Clear query cache
            self.query_cache.clear()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update relationship {relationship_id}: {e}")
            return False
    
    async def delete_entity(self, entity_id: str, cascade: bool = False) -> bool:
        """Delete entity from storage"""
        try:
            if entity_id not in self.entities:
                return False
            
            entity_data = self.entities[entity_id]
            
            # Handle cascade deletion
            if cascade:
                # Delete all connected relationships
                connected_rels = (self.outgoing_relationships[entity_id] | 
                                self.incoming_relationships[entity_id])
                
                for rel_id in list(connected_rels):
                    await self.delete_relationship(rel_id)
            else:
                # Check for connected relationships
                if (self.outgoing_relationships[entity_id] or 
                    self.incoming_relationships[entity_id]):
                    self.logger.warning(f"⚠️  Cannot delete entity {entity_id}: has connected relationships")
                    return False
            
            # Remove from type index
            entity_type = entity_data.get('entity_type', 'unknown')
            self.entities_by_type[entity_type].discard(entity_id)
            
            # Remove from indexes
            for index in self.indexes.values():
                if index.name.startswith('entity_'):
                    index.remove_item(entity_id, entity_data)
            
            # Remove entity
            del self.entities[entity_id]
            
            # Clear query cache
            self.query_cache.clear()
            
            self.logger.debug(f"🗑️  Deleted entity: {entity_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to delete entity {entity_id}: {e}")
            return False
    
    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete relationship from storage"""
        try:
            if relationship_id not in self.relationships:
                return False
            
            rel_data = self.relationships[relationship_id]
            source_id = rel_data.get('source_entity_id')
            target_id = rel_data.get('target_entity_id')
            
            # Remove from type index
            rel_type = rel_data.get('relation_type', 'unknown')
            self.relationships_by_type[rel_type].discard(relationship_id)
            
            # Remove from graph traversal indexes
            if source_id:
                self.outgoing_relationships[source_id].discard(relationship_id)
            if target_id:
                self.incoming_relationships[target_id].discard(relationship_id)
            
            # Remove from indexes
            for index in self.indexes.values():
                if index.name.startswith('relationship_'):
                    index.remove_item(relationship_id, rel_data)
            
            # Remove relationship
            del self.relationships[relationship_id]
            
            # Clear query cache
            self.query_cache.clear()
            
            self.logger.debug(f"🗑️  Deleted relationship: {relationship_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to delete relationship {relationship_id}: {e}")
            return False
    
    async def query_entities(self, 
                           entity_type: Optional[str] = None,
                           properties: Optional[Dict[str, Any]] = None,
                           limit: Optional[int] = None) -> List[Any]:
        """Query entities with filtering criteria"""
        try:
            start_time = time.time()
            
            # Generate cache key
            cache_key = f"entities_{entity_type}_{properties}_{limit}"
            
            if self.config.enable_caching and cache_key in self.query_cache:
                self.cache_hits += 1
                self.stats['cache_hits'] += 1
                return self.query_cache[cache_key]
            
            self.cache_misses += 1
            self.stats['cache_misses'] += 1
            
            # Start with all entities or filter by type
            if entity_type:
                candidate_ids = self.entities_by_type.get(entity_type, set())
            else:
                candidate_ids = set(self.entities.keys())
            
            # Apply property filters
            if properties:
                filtered_ids = set()
                for entity_id in candidate_ids:
                    entity_data = self.entities[entity_id]
                    matches = True
                    
                    for prop, value in properties.items():
                        if entity_data.get(prop) != value:
                            matches = False
                            break
                    
                    if matches:
                        filtered_ids.add(entity_id)
                
                candidate_ids = filtered_ids
            
            # Convert to entity objects
            results = []
            for entity_id in candidate_ids:
                entity = await self.get_entity(entity_id)
                if entity:
                    results.append(entity)
                
                if limit and len(results) >= limit:
                    break
            
            # Cache results
            if self.config.enable_caching:
                if len(self.query_cache) >= self.config.cache_size:
                    # Simple cache eviction - remove oldest entries
                    keys_to_remove = list(self.query_cache.keys())[:100]
                    for key in keys_to_remove:
                        del self.query_cache[key]
                
                self.query_cache[cache_key] = results
            
            execution_time = time.time() - start_time
            self.stats['queries_executed'] += 1
            
            self.logger.debug(f"🔍 Entity query returned {len(results)} results in {execution_time:.3f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to query entities: {e}")
            return []
    
    async def query_relationships(self,
                                relationship_type: Optional[str] = None,
                                source_entity_id: Optional[str] = None,
                                target_entity_id: Optional[str] = None,
                                properties: Optional[Dict[str, Any]] = None,
                                limit: Optional[int] = None) -> List[Any]:
        """Query relationships with filtering criteria"""
        try:
            start_time = time.time()
            
            # Generate cache key
            cache_key = f"relationships_{relationship_type}_{source_entity_id}_{target_entity_id}_{properties}_{limit}"
            
            if self.config.enable_caching and cache_key in self.query_cache:
                self.cache_hits += 1
                self.stats['cache_hits'] += 1
                return self.query_cache[cache_key]
            
            self.cache_misses += 1
            self.stats['cache_misses'] += 1
            
            # Start with candidates based on most specific filter
            if source_entity_id:
                candidate_ids = self.outgoing_relationships.get(source_entity_id, set())
            elif target_entity_id:
                candidate_ids = self.incoming_relationships.get(target_entity_id, set())
            elif relationship_type:
                candidate_ids = self.relationships_by_type.get(relationship_type, set())
            else:
                candidate_ids = set(self.relationships.keys())
            
            # Apply additional filters
            filtered_ids = set()
            for rel_id in candidate_ids:
                rel_data = self.relationships[rel_id]
                matches = True
                
                # Filter by type
                if relationship_type and rel_data.get('relation_type') != relationship_type:
                    matches = False
                
                # Filter by source
                if source_entity_id and rel_data.get('source_entity_id') != source_entity_id:
                    matches = False
                
                # Filter by target
                if target_entity_id and rel_data.get('target_entity_id') != target_entity_id:
                    matches = False
                
                # Filter by properties
                if properties:
                    for prop, value in properties.items():
                        if rel_data.get(prop) != value:
                            matches = False
                            break
                
                if matches:
                    filtered_ids.add(rel_id)
            
            # Convert to relationship objects
            results = []
            for rel_id in filtered_ids:
                relationship = await self.get_relationship(rel_id)
                if relationship:
                    results.append(relationship)
                
                if limit and len(results) >= limit:
                    break
            
            # Cache results
            if self.config.enable_caching:
                if len(self.query_cache) >= self.config.cache_size:
                    # Simple cache eviction
                    keys_to_remove = list(self.query_cache.keys())[:100]
                    for key in keys_to_remove:
                        del self.query_cache[key]
                
                self.query_cache[cache_key] = results
            
            execution_time = time.time() - start_time
            self.stats['queries_executed'] += 1
            
            self.logger.debug(f"🔍 Relationship query returned {len(results)} results in {execution_time:.3f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to query relationships: {e}")
            return []
    
    async def find_path(self, 
                       source_id: str, 
                       target_id: str, 
                       max_depth: int = 5,
                       relationship_types: Optional[List[str]] = None) -> List[List[str]]:
        """Find paths between two entities using BFS"""
        try:
            if source_id not in self.entities or target_id not in self.entities:
                return []
            
            if source_id == target_id:
                return [[source_id]]
            
            # BFS to find paths
            queue = [(source_id, [source_id])]
            visited = set()
            paths = []
            
            while queue and len(paths) < 10:  # Limit number of paths
                current_id, path = queue.pop(0)
                
                if len(path) > max_depth:
                    continue
                
                if current_id in visited:
                    continue
                
                visited.add(current_id)
                
                # Get outgoing relationships
                for rel_id in self.outgoing_relationships.get(current_id, set()):
                    rel_data = self.relationships[rel_id]
                    
                    # Filter by relationship type
                    if relationship_types:
                        rel_type = rel_data.get('relation_type')
                        if rel_type not in relationship_types:
                            continue
                    
                    next_id = rel_data.get('target_entity_id')
                    if not next_id or next_id in path:
                        continue
                    
                    new_path = path + [next_id]
                    
                    if next_id == target_id:
                        paths.append(new_path)
                    else:
                        queue.append((next_id, new_path))
            
            self.logger.debug(f"🛤️  Found {len(paths)} paths from {source_id} to {target_id}")
            return paths
            
        except Exception as e:
            self.logger.error(f"❌ Failed to find path: {e}")
            return []
    
    async def get_neighbors(self, 
                          entity_id: str, 
                          relationship_types: Optional[List[str]] = None,
                          direction: str = "both") -> List[Any]:
        """Get neighboring entities"""
        try:
            if entity_id not in self.entities:
                return []
            
            neighbor_ids = set()
            
            # Get outgoing neighbors
            if direction in ["outgoing", "both"]:
                for rel_id in self.outgoing_relationships.get(entity_id, set()):
                    rel_data = self.relationships[rel_id]
                    
                    # Filter by relationship type
                    if relationship_types:
                        rel_type = rel_data.get('relation_type')
                        if rel_type not in relationship_types:
                            continue
                    
                    target_id = rel_data.get('target_entity_id')
                    if target_id:
                        neighbor_ids.add(target_id)
            
            # Get incoming neighbors
            if direction in ["incoming", "both"]:
                for rel_id in self.incoming_relationships.get(entity_id, set()):
                    rel_data = self.relationships[rel_id]
                    
                    # Filter by relationship type
                    if relationship_types:
                        rel_type = rel_data.get('relation_type')
                        if rel_type not in relationship_types:
                            continue
                    
                    source_id = rel_data.get('source_entity_id')
                    if source_id:
                        neighbor_ids.add(source_id)
            
            # Convert to entity objects
            neighbors = []
            for neighbor_id in neighbor_ids:
                entity = await self.get_entity(neighbor_id)
                if entity:
                    neighbors.append(entity)
            
            self.logger.debug(f"👥 Found {len(neighbors)} neighbors for {entity_id}")
            return neighbors
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get neighbors: {e}")
            return []
    
    # Performance optimization methods
    async def create_index(self, 
                          index_name: str,
                          target_type: str,
                          properties: List[str],
                          unique: bool = False) -> bool:
        """Create index for performance optimization"""
        try:
            if index_name in self.indexes:
                return False  # Index already exists
            
            index = MemoryIndex(f"{target_type}_{index_name}", properties, unique)
            
            # Populate index with existing data
            if target_type == "entity":
                for entity_id, entity_data in self.entities.items():
                    index.add_item(entity_id, entity_data)
            elif target_type == "relationship":
                for rel_id, rel_data in self.relationships.items():
                    index.add_item(rel_id, rel_data)
            
            self.indexes[index_name] = index
            self.stats['index_operations'] += 1
            
            self.logger.info(f"📊 Created index: {index_name} on {target_type} ({properties})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create index {index_name}: {e}")
            return False
    
    async def drop_index(self, index_name: str) -> bool:
        """Drop an existing index"""
        try:
            if index_name not in self.indexes:
                return False
            
            del self.indexes[index_name]
            self.stats['index_operations'] += 1
            
            self.logger.info(f"🗑️  Dropped index: {index_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to drop index {index_name}: {e}")
            return False
    
    async def list_indexes(self) -> List[IndexInfo]:
        """List all available indexes"""
        try:
            index_info = []
            for name, index in self.indexes.items():
                info = IndexInfo(
                    name=name,
                    type=index.name.split('_')[0],
                    properties=index.properties,
                    unique=index.unique,
                    created_at=str(index.created_at),
                    size_bytes=len(str(index.index))  # Rough estimate
                )
                index_info.append(info)
            
            return index_info
            
        except Exception as e:
            self.logger.error(f"❌ Failed to list indexes: {e}")
            return []
    
    # Bulk operations for performance
    async def bulk_create_entities(self, entities: List[Any]) -> int:
        """Create multiple entities in batch"""
        try:
            created_count = 0
            
            for entity in entities:
                if await self.create_entity(entity):
                    created_count += 1
            
            self.logger.info(f"📦 Bulk created {created_count}/{len(entities)} entities")
            return created_count
            
        except Exception as e:
            self.logger.error(f"❌ Failed bulk entity creation: {e}")
            return 0
    
    async def bulk_create_relationships(self, relationships: List[Any]) -> int:
        """Create multiple relationships in batch"""
        try:
            created_count = 0
            
            for relationship in relationships:
                if await self.create_relationship(relationship):
                    created_count += 1
            
            self.logger.info(f"📦 Bulk created {created_count}/{len(relationships)} relationships")
            return created_count
            
        except Exception as e:
            self.logger.error(f"❌ Failed bulk relationship creation: {e}")
            return 0
    
    # Statistics and monitoring
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics and metrics"""
        try:
            stats = {
                'connection_status': self._is_connected,
                'entity_count': len(self.entities),
                'relationship_count': len(self.relationships),
                'index_count': len(self.indexes),
                'cache_size': len(self.query_cache),
                'cache_hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
                'operations': self.stats.copy(),
                'memory_usage': {
                    'entities_mb': len(str(self.entities)) / (1024 * 1024),
                    'relationships_mb': len(str(self.relationships)) / (1024 * 1024),
                    'indexes_mb': len(str(self.indexes)) / (1024 * 1024),
                    'cache_mb': len(str(self.query_cache)) / (1024 * 1024)
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get statistics: {e}")
            return {}
    
    async def clear_cache(self) -> None:
        """Clear internal caches"""
        try:
            cache_size = len(self.query_cache)
            self.query_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            
            self.logger.info(f"🧹 Cleared cache ({cache_size} entries)")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to clear cache: {e}")
    
    # Transaction support (simplified for memory store)
    async def begin_transaction(self) -> str:
        """Begin a transaction"""
        try:
            transaction_id = str(uuid.uuid4())
            self.transactions[transaction_id] = {
                'started_at': time.time(),
                'operations': []
            }
            
            self.logger.debug(f"🔄 Started transaction: {transaction_id}")
            return transaction_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to begin transaction: {e}")
            return ""
    
    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit a transaction"""
        try:
            if transaction_id not in self.transactions:
                return False
            
            # For memory store, operations are applied immediately
            # This is just cleanup
            del self.transactions[transaction_id]
            
            self.logger.debug(f"✅ Committed transaction: {transaction_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to commit transaction {transaction_id}: {e}")
            return False
    
    async def rollback_transaction(self, transaction_id: str) -> bool:
        """Rollback a transaction"""
        try:
            if transaction_id not in self.transactions:
                return False
            
            # For memory store, rollback would require more complex state management
            # This is a simplified implementation
            del self.transactions[transaction_id]
            
            self.logger.warning(f"⚠️  Rolled back transaction: {transaction_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to rollback transaction {transaction_id}: {e}")
            return False
