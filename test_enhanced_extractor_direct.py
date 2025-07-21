"""
Direct Test of Enhanced Entity Extractor

This script directly tests the enhanced entity extractor by creating
a simple schema setup that bypasses the complex GraphSchema constructor.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.core.graph_schema import (
    EntityTypeSchema, PropertySchema, PropertyType, EntityInstance
)
from mcp_server.extraction.enhanced_entity_extractor import (
    EnhancedEntityExtractor, ExtractionContext, ExtractionStrategy
)

# Create a minimal schema manager for testing
class TestSchemaManager:
    def __init__(self):
        self.entity_types = {}
    
    def add_entity_type(self, entity_schema):
        self.entity_types[entity_schema.name] = entity_schema
    
    def validate_entity(self, entity):
        # Simple validation
        return True, []

def test_enhanced_entity_extraction_direct():
    """Test enhanced entity extraction with direct setup"""
    print("🧪 Direct Test - Enhanced Entity Extraction - Phase 2.2")
    print("=" * 60)
    
    # Create minimal schema manager
    schema_manager = TestSchemaManager()
    
    # Define entity types with correct constructor calls
    # Note: Using basic field assignment due to dataclass fallback
    person_schema = EntityTypeSchema()
    person_schema.name = "person"
    person_schema.description = "Individual person"
    person_schema.properties = []
    person_schema.required_properties = set()
    
    org_schema = EntityTypeSchema()
    org_schema.name = "organization"
    org_schema.description = "Organization or company"
    org_schema.properties = []
    org_schema.required_properties = set()
    
    location_schema = EntityTypeSchema()
    location_schema.name = "location"
    location_schema.description = "Geographic location"
    location_schema.properties = []
    location_schema.required_properties = set()
    
    # Add to schema manager
    schema_manager.add_entity_type(person_schema)
    schema_manager.add_entity_type(org_schema)
    schema_manager.add_entity_type(location_schema)
    
    print(f"✅ Test schema manager created with {len(schema_manager.entity_types)} entity types")
    for name in schema_manager.entity_types.keys():
        print(f"  📋 {name}")
    print()
    
    # Test text
    test_text = """
    Dr. Sarah Chen works for Microsoft Corporation in Seattle.
    She founded the AI Research Lab using advanced machine learning.
    Microsoft was established by Bill Gates in 1975.
    Prof. John Smith works at Stanford University in California.
    """
    
    print(f"📝 Test text ({len(test_text)} chars):")
    print(test_text.strip())
    print()
    
    # Create extractor
    print("🔧 Creating enhanced entity extractor...")
    extractor = EnhancedEntityExtractor(schema_manager)
    
    # Create extraction context
    context = ExtractionContext(
        text=test_text,
        schema_manager=schema_manager,
        confidence_threshold=0.6,
        enable_resolution=False  # Disable resolution for this test
    )
    
    print(f"🔍 Extracting entities with {len(context.strategies)} strategies...")
    print(f"  📊 Strategies: {[s.value for s in context.strategies]}")
    print()
    
    # Extract entities
    entities = extractor.extract_entities(context)
    
    print(f"✅ Extracted {len(entities)} entities:")
    print()
    
    # Display results by entity type
    entities_by_type = {}
    for entity in entities:
        entity_type = entity.entity_type
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []
        entities_by_type[entity_type].append(entity)
    
    for entity_type, type_entities in entities_by_type.items():
        print(f"📊 {entity_type.upper()} ({len(type_entities)} entities):")
        for entity in type_entities:
            name = entity.name
            confidence = entity.confidence
            strategy = entity.properties.get('extraction_strategy', 'N/A')
            evidence = entity.properties.get('evidence', 'N/A')
            print(f"  📝 '{name}'")
            print(f"     💯 Confidence: {confidence:.3f}")
            print(f"     🎯 Strategy: {strategy}")
            print(f"     🔍 Evidence: {evidence[:50]}...")
            print()
    
    # Get extraction statistics
    stats = extractor.get_extraction_statistics()
    
    print(f"📊 Extraction Statistics:")
    print(f"  🔢 Total extractions: {stats['total_extractions']}")
    print(f"  ⭐ High confidence extractions: {stats['high_confidence_extractions']}")
    print(f"  🎯 Strategy counts:")
    for strategy, count in stats['strategy_counts'].items():
        print(f"     {strategy.value}: {count}")
    print(f"  📋 Entity type counts:")
    for entity_type, count in stats['entity_type_counts'].items():
        print(f"     {entity_type}: {count}")
    print(f"  🔧 Models available:")
    for model, available in stats['models_available'].items():
        status = "✅" if available else "❌"
        print(f"     {status} {model}: {available}")
    print(f"  ✅ Strategies enabled: {sum(1 for v in stats['strategies_enabled'].values() if v)}/{len(stats['strategies_enabled'])}")
    
    print()
    print("🎉 Enhanced Entity Extraction direct test completed successfully!")
    print()
    
    # Test individual extraction strategies
    print("🔬 Testing individual extraction strategies...")
    
    for strategy in [ExtractionStrategy.SCHEMA_GUIDED, ExtractionStrategy.PATTERN_BASED, ExtractionStrategy.CONTEXTUAL]:
        if stats['strategies_enabled'].get(strategy, False):
            test_context = ExtractionContext(
                text=test_text,
                schema_manager=schema_manager,
                confidence_threshold=0.6,
                strategies=[strategy]  # Test single strategy
            )
            
            strategy_entities = extractor.extract_entities(test_context)
            print(f"  🎯 {strategy.value}: {len(strategy_entities)} entities")
    
    return entities

if __name__ == "__main__":
    test_enhanced_entity_extraction_direct()
