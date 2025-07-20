"""
Core Knowledge Graph Components

This package contains the fundamental components for building and managing
knowledge graphs including schema management, entity resolution, and analytics.
"""

try:
    from .graph_schema import (
        SchemaManager,
        EntityTypeSchema,
        RelationshipTypeSchema,
        GraphSchema,
        EntityInstance,
        RelationshipInstance,
        PropertySchema,
        PropertyType,
        Cardinality,
        create_default_schema_manager
    )
    SCHEMA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Schema management not available: {e}")
    SCHEMA_AVAILABLE = False

__all__ = [
    'SchemaManager',
    'EntityTypeSchema', 
    'RelationshipTypeSchema',
    'GraphSchema',
    'EntityInstance',
    'RelationshipInstance',
    'PropertySchema',
    'PropertyType',
    'Cardinality',
    'create_default_schema_manager',
    'SCHEMA_AVAILABLE'
]
