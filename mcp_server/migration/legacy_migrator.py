"""
Legacy Data Migration Utilities

This module provides utilities for migrating data from the legacy memory_banks 
dictionary format to the new Phase 3.2 storage abstraction layer.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import uuid
from datetime import datetime

# Import new storage system
try:
    from storage import MemoryStore, create_memory_store, StorageConfig
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    
# Import core components
try:
    from core.graph_schema import EntityInstance, RelationshipInstance, SchemaManager
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

logger = logging.getLogger(__name__)

class LegacyDataMigrator:
    """Handles migration from legacy memory_banks format to new storage system"""
    
    def __init__(self, schema_manager: Optional[Any] = None):
        self.schema_manager = schema_manager
        
    def migrate_memory_banks(self, legacy_banks: Dict[str, Any], 
                           storage_backends: Dict[str, MemoryStore]) -> bool:
        """
        Migrate legacy memory banks to new storage system
        
        Args:
            legacy_banks: Dictionary with legacy bank data
            storage_backends: Dictionary of MemoryStore instances by bank name
            
        Returns:
            bool: True if migration successful, False otherwise
        """
        try:
            for bank_name, bank_data in legacy_banks.items():
                if bank_name not in storage_backends:
                    # Create new storage backend for this bank
                    storage_backends[bank_name] = create_memory_store()
                    
                storage = storage_backends[bank_name]
                
                # Migrate entities (nodes)
                if "nodes" in bank_data:
                    self._migrate_nodes(bank_data["nodes"], storage)
                    
                # Migrate relationships (edges)
                if "edges" in bank_data:
                    self._migrate_edges(bank_data["edges"], storage)
                    
                # Migrate observations
                if "observations" in bank_data:
                    self._migrate_observations(bank_data["observations"], storage)
                    
                # Migrate reasoning steps
                if "reasoning_steps" in bank_data:
                    self._migrate_reasoning_steps(bank_data["reasoning_steps"], storage)
                    
                logger.info(f"Successfully migrated bank: {bank_name}")
                
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
            
    def _migrate_nodes(self, legacy_nodes: Dict[str, Any], storage: MemoryStore):
        """Migrate legacy nodes to entities"""
        for node_id, node_data in legacy_nodes.items():
            try:
                # Convert legacy node to EntityInstance format
                if CORE_AVAILABLE and self.schema_manager:
                    entity = EntityInstance(
                        id=node_data.get("id", node_id),
                        type=node_data.get("type", "node"),
                        properties=node_data.get("data", {}),
                        schema_manager=self.schema_manager
                    )
                    storage.create_entity(entity)
                else:
                    # Fallback format
                    entity_data = {
                        "id": node_data.get("id", node_id),
                        "type": node_data.get("type", "node"),
                        "properties": node_data.get("data", {})
                    }
                    storage.create_entity(entity_data)
                    
            except Exception as e:
                logger.warning(f"Failed to migrate node {node_id}: {e}")
                
    def _migrate_edges(self, legacy_edges: List[Dict[str, Any]], storage: MemoryStore):
        """Migrate legacy edges to relationships"""
        for edge_data in legacy_edges:
            try:
                # Convert legacy edge to RelationshipInstance format
                if CORE_AVAILABLE and self.schema_manager:
                    relationship = RelationshipInstance(
                        id=edge_data.get("id", str(uuid.uuid4())),
                        type=edge_data.get("type", "relation"),
                        source_id=edge_data.get("source"),
                        target_id=edge_data.get("target"),
                        properties=edge_data.get("data", {}),
                        schema_manager=self.schema_manager
                    )
                    storage.create_relationship(relationship)
                else:
                    # Fallback format
                    relationship_data = {
                        "id": edge_data.get("id", str(uuid.uuid4())),
                        "type": edge_data.get("type", "relation"),
                        "source_id": edge_data.get("source"),
                        "target_id": edge_data.get("target"),
                        "properties": edge_data.get("data", {})
                    }
                    storage.create_relationship(relationship_data)
                    
            except Exception as e:
                logger.warning(f"Failed to migrate edge: {e}")
                
    def _migrate_observations(self, legacy_observations: List[Dict[str, Any]], 
                            storage: MemoryStore):
        """Migrate legacy observations"""
        for obs_data in legacy_observations:
            try:
                # Store observations as special entities
                obs_entity = {
                    "id": obs_data.get("id", str(uuid.uuid4())),
                    "type": "observation",
                    "properties": {
                        "entity_id": obs_data.get("entity_id"),
                        "content": obs_data.get("content"),
                        "timestamp": obs_data.get("timestamp", datetime.now().isoformat())
                    }
                }
                storage.create_entity(obs_entity)
                
            except Exception as e:
                logger.warning(f"Failed to migrate observation: {e}")
                
    def _migrate_reasoning_steps(self, legacy_steps: List[Dict[str, Any]], 
                               storage: MemoryStore):
        """Migrate legacy reasoning steps"""
        for step_data in legacy_steps:
            try:
                # Store reasoning steps as special entities
                step_entity = {
                    "id": step_data.get("id", str(uuid.uuid4())),
                    "type": "reasoning_step",
                    "properties": {
                        "description": step_data.get("description"),
                        "status": step_data.get("status", "pending"),
                        "timestamp": step_data.get("timestamp", datetime.now().isoformat()),
                        "related_entities": step_data.get("related_entities", []),
                        "related_relations": step_data.get("related_relations", [])
                    }
                }
                storage.create_entity(step_entity)
                
            except Exception as e:
                logger.warning(f"Failed to migrate reasoning step: {e}")

def migrate_legacy_data(legacy_banks: Dict[str, Any], 
                       storage_backends: Dict[str, MemoryStore],
                       schema_manager: Optional[Any] = None) -> bool:
    """
    Convenience function to migrate legacy data
    
    Args:
        legacy_banks: Legacy memory banks data
        storage_backends: New storage backends
        schema_manager: Optional schema manager for validation
        
    Returns:
        bool: True if migration successful
    """
    migrator = LegacyDataMigrator(schema_manager)
    return migrator.migrate_memory_banks(legacy_banks, storage_backends)
