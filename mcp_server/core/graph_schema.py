"""
Graph Schema Management System

This module provides comprehensive schema definition and validation for knowledge graphs,
including entity types, relationship types, property schemas, and namespace management.
"""

from typing import Dict, List, Optional, Any, Set, Union, Type, Tuple
try:
    from pydantic import BaseModel, Field, validator
except ImportError:
    # Fallback if pydantic not available
    print("Warning: pydantic not available, using basic dataclasses")
    from dataclasses import dataclass, field
    BaseModel = object
    Field = field
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
from enum import Enum
import json
import uuid
from datetime import datetime
from pathlib import Path


class PropertyType(str, Enum):
    """Supported property data types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    LIST = "list"
    DICT = "dict"
    EMBEDDING = "embedding"


class Cardinality(str, Enum):
    """Relationship cardinality constraints"""
    ONE_TO_ONE = "1:1"
    ONE_TO_MANY = "1:N"
    MANY_TO_ONE = "N:1"
    MANY_TO_MANY = "N:N"


class PropertySchema(BaseModel):
    """Schema definition for entity and relationship properties"""
    name: str
    property_type: PropertyType
    required: bool = False
    default_value: Optional[Any] = None
    description: Optional[str] = None
    validation_pattern: Optional[str] = None  # Regex pattern for string validation
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    
    @validator('validation_pattern')
    def validate_pattern(cls, v, values):
        """Ensure regex pattern is only used with string types"""
        if v is not None and values.get('property_type') != PropertyType.STRING:
            raise ValueError("validation_pattern can only be used with string properties")
        return v


class EntityTypeSchema(BaseModel):
    """Schema definition for entity types"""
    name: str
    description: Optional[str] = None
    parent_type: Optional[str] = None  # For inheritance
    properties: List[PropertySchema] = Field(default_factory=list)
    required_properties: Set[str] = Field(default_factory=set)
    namespace: str = "default"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    
    @validator('required_properties', pre=True)
    def validate_required_properties(cls, v, values):
        """Ensure required properties exist in property definitions"""
        if isinstance(v, list):
            v = set(v)
        properties = values.get('properties', [])
        property_names = {prop.name for prop in properties}
        if not v.issubset(property_names):
            missing = v - property_names
            raise ValueError(f"Required properties {missing} not found in property definitions")
        return v


class RelationshipTypeSchema(BaseModel):
    """Schema definition for relationship types"""
    name: str
    description: Optional[str] = None
    source_entity_types: List[str] = Field(default_factory=list)  # Allowed source types
    target_entity_types: List[str] = Field(default_factory=list)  # Allowed target types
    cardinality: Cardinality = Cardinality.MANY_TO_MANY
    properties: List[PropertySchema] = Field(default_factory=list)
    bidirectional: bool = False
    namespace: str = "default"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"


class GraphSchema(BaseModel):
    """Complete graph schema with entity and relationship types"""
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    entity_types: Dict[str, EntityTypeSchema] = Field(default_factory=dict)
    relationship_types: Dict[str, RelationshipTypeSchema] = Field(default_factory=dict)
    namespaces: Set[str] = Field(default_factory=lambda: {"default"})
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EntityInstance(BaseModel):
    """Instance of an entity following a schema"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    entity_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    namespace: str = "default"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class RelationshipInstance(BaseModel):
    """Instance of a relationship following a schema"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    context: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SchemaManager:
    """Manages graph schemas with validation and evolution capabilities"""
    
    def __init__(self, schema_file: Optional[Path] = None):
        self.schema = GraphSchema(name="default_schema")
        self.schema_file = schema_file
        if schema_file and schema_file.exists():
            self.load_schema(schema_file)
        else:
            self._initialize_default_schema()
    
    def _initialize_default_schema(self):
        """Initialize with common entity and relationship types"""
        # Common entity types
        self.add_entity_type(EntityTypeSchema(
            name="person",
            description="Individual person",
            properties=[
                PropertySchema(name="full_name", property_type=PropertyType.STRING),
                PropertySchema(name="age", property_type=PropertyType.INTEGER, min_value=0),
                PropertySchema(name="occupation", property_type=PropertyType.STRING),
                PropertySchema(name="location", property_type=PropertyType.STRING),
            ],
            required_properties={"full_name"}
        ))
        
        self.add_entity_type(EntityTypeSchema(
            name="organization",
            description="Company, institution, or group",
            properties=[
                PropertySchema(name="industry", property_type=PropertyType.STRING),
                PropertySchema(name="size", property_type=PropertyType.STRING),
                PropertySchema(name="founded_year", property_type=PropertyType.INTEGER),
            ]
        ))
        
        self.add_entity_type(EntityTypeSchema(
            name="location",
            description="Geographic location",
            properties=[
                PropertySchema(name="country", property_type=PropertyType.STRING),
                PropertySchema(name="city", property_type=PropertyType.STRING),
                PropertySchema(name="coordinates", property_type=PropertyType.LIST),
            ]
        ))
        
        self.add_entity_type(EntityTypeSchema(
            name="concept",
            description="Abstract concept or idea",
            properties=[
                PropertySchema(name="domain", property_type=PropertyType.STRING),
                PropertySchema(name="definition", property_type=PropertyType.STRING),
            ]
        ))
        
        # Common relationship types
        self.add_relationship_type(RelationshipTypeSchema(
            name="works_for",
            description="Employment relationship",
            source_entity_types=["person"],
            target_entity_types=["organization"],
            cardinality=Cardinality.MANY_TO_ONE,
            properties=[
                PropertySchema(name="position", property_type=PropertyType.STRING),
                PropertySchema(name="start_date", property_type=PropertyType.DATE),
                PropertySchema(name="end_date", property_type=PropertyType.DATE),
            ]
        ))
        
        self.add_relationship_type(RelationshipTypeSchema(
            name="located_in",
            description="Geographic containment",
            target_entity_types=["location"],
            cardinality=Cardinality.MANY_TO_ONE
        ))
        
        self.add_relationship_type(RelationshipTypeSchema(
            name="related_to",
            description="General relationship",
            cardinality=Cardinality.MANY_TO_MANY,
            properties=[
                PropertySchema(name="relationship_strength", property_type=PropertyType.FLOAT, min_value=0.0, max_value=1.0),
            ]
        ))
    
    def add_entity_type(self, entity_type: EntityTypeSchema) -> bool:
        """Add a new entity type to the schema"""
        if entity_type.name in self.schema.entity_types:
            return False
        
        # Validate parent type exists if specified
        if entity_type.parent_type and entity_type.parent_type not in self.schema.entity_types:
            raise ValueError(f"Parent type '{entity_type.parent_type}' not found")
        
        self.schema.entity_types[entity_type.name] = entity_type
        self.schema.namespaces.add(entity_type.namespace)
        self.schema.updated_at = datetime.utcnow()
        return True
    
    def add_relationship_type(self, relationship_type: RelationshipTypeSchema) -> bool:
        """Add a new relationship type to the schema"""
        if relationship_type.name in self.schema.relationship_types:
            return False
        
        # Validate referenced entity types exist
        for entity_type in relationship_type.source_entity_types + relationship_type.target_entity_types:
            if entity_type and entity_type not in self.schema.entity_types:
                raise ValueError(f"Entity type '{entity_type}' not found in schema")
        
        self.schema.relationship_types[relationship_type.name] = relationship_type
        self.schema.namespaces.add(relationship_type.namespace)
        self.schema.updated_at = datetime.utcnow()
        return True
    
    def validate_entity(self, entity: EntityInstance) -> Tuple[bool, List[str]]:
        """Validate an entity instance against the schema"""
        errors = []
        
        # Check if entity type exists
        if entity.entity_type not in self.schema.entity_types:
            errors.append(f"Entity type '{entity.entity_type}' not found in schema")
            return False, errors
        
        entity_schema = self.schema.entity_types[entity.entity_type]
        
        # Check required properties
        required_props = entity_schema.required_properties
        for prop in required_props:
            if prop not in entity.properties:
                errors.append(f"Required property '{prop}' missing")
        
        # Validate property types and values
        property_schemas = {prop.name: prop for prop in entity_schema.properties}
        for prop_name, prop_value in entity.properties.items():
            if prop_name in property_schemas:
                prop_schema = property_schemas[prop_name]
                if not self._validate_property_value(prop_value, prop_schema):
                    errors.append(f"Property '{prop_name}' validation failed")
        
        return len(errors) == 0, errors
    
    def validate_relationship(self, relationship: RelationshipInstance, 
                            source_entity: EntityInstance, 
                            target_entity: EntityInstance) -> Tuple[bool, List[str]]:
        """Validate a relationship instance against the schema"""
        errors = []
        
        # Check if relationship type exists
        if relationship.relationship_type not in self.schema.relationship_types:
            errors.append(f"Relationship type '{relationship.relationship_type}' not found")
            return False, errors
        
        rel_schema = self.schema.relationship_types[relationship.relationship_type]
        
        # Check entity type constraints
        if rel_schema.source_entity_types and source_entity.entity_type not in rel_schema.source_entity_types:
            errors.append(f"Source entity type '{source_entity.entity_type}' not allowed")
        
        if rel_schema.target_entity_types and target_entity.entity_type not in rel_schema.target_entity_types:
            errors.append(f"Target entity type '{target_entity.entity_type}' not allowed")
        
        # Validate properties
        property_schemas = {prop.name: prop for prop in rel_schema.properties}
        for prop_name, prop_value in relationship.properties.items():
            if prop_name in property_schemas:
                prop_schema = property_schemas[prop_name]
                if not self._validate_property_value(prop_value, prop_schema):
                    errors.append(f"Relationship property '{prop_name}' validation failed")
        
        return len(errors) == 0, errors
    
    def _validate_property_value(self, value: Any, schema: PropertySchema) -> bool:
        """Validate a property value against its schema"""
        if value is None:
            return not schema.required
        
        # Type validation
        if schema.property_type == PropertyType.STRING and not isinstance(value, str):
            return False
        elif schema.property_type == PropertyType.INTEGER and not isinstance(value, int):
            return False
        elif schema.property_type == PropertyType.FLOAT and not isinstance(value, (int, float)):
            return False
        elif schema.property_type == PropertyType.BOOLEAN and not isinstance(value, bool):
            return False
        elif schema.property_type == PropertyType.LIST and not isinstance(value, list):
            return False
        elif schema.property_type == PropertyType.DICT and not isinstance(value, dict):
            return False
        
        # Range validation
        if schema.min_value is not None and value < schema.min_value:
            return False
        if schema.max_value is not None and value > schema.max_value:
            return False
        
        # Allowed values validation
        if schema.allowed_values and value not in schema.allowed_values:
            return False
        
        # Pattern validation for strings
        if schema.validation_pattern and schema.property_type == PropertyType.STRING:
            import re
            if not re.match(schema.validation_pattern, value):
                return False
        
        return True
    
    def get_entity_hierarchy(self, entity_type: str) -> List[str]:
        """Get the inheritance hierarchy for an entity type"""
        hierarchy = [entity_type]
        current_type = entity_type
        
        while current_type in self.schema.entity_types:
            parent = self.schema.entity_types[current_type].parent_type
            if parent and parent != current_type:
                hierarchy.append(parent)
                current_type = parent
            else:
                break
        
        return hierarchy
    
    def save_schema(self, file_path: Optional[Path] = None) -> bool:
        """Save schema to JSON file"""
        target_file = file_path or self.schema_file
        if not target_file:
            return False
        
        try:
            with open(target_file, 'w') as f:
                json.dump(self.schema.dict(), f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving schema: {e}")
            return False
    
    def load_schema(self, file_path: Path) -> bool:
        """Load schema from JSON file"""
        try:
            with open(file_path, 'r') as f:
                schema_data = json.load(f)
            self.schema = GraphSchema(**schema_data)
            return True
        except Exception as e:
            print(f"Error loading schema: {e}")
            return False


def create_default_schema_manager() -> SchemaManager:
    """Factory function to create a schema manager with default configuration"""
    return SchemaManager()


if __name__ == "__main__":
    # Example usage
    schema_manager = create_default_schema_manager()
    
    # Create a person entity
    person = EntityInstance(
        name="John Doe",
        entity_type="person",
        properties={
            "full_name": "John Doe",
            "age": 30,
            "occupation": "Software Engineer"
        }
    )
    
    # Validate the entity
    is_valid, errors = schema_manager.validate_entity(person)
    print(f"Entity valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    # Save schema
    schema_file = Path("default_schema.json")
    schema_manager.save_schema(schema_file)
    print(f"Schema saved to {schema_file}")
