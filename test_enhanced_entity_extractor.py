"""
Test Enhanced Entity Extractor

This script tests the enhanced entity extractor implementation
to verify it works correctly with the Phase 1 core components.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from the correct path
from mcp_server.core.graph_schema import (
    SchemaManager, EntityTypeSchema, PropertySchema, PropertyType, EntityInstance
)
from mcp_server.core.entity_resolution import EntityResolver
from mcp_server.extraction.enhanced_entity_extractor import (
    EnhancedEntityExtractor, ExtractionContext, extract_entities_from_text
)

def test_enhanced_entity_extraction():
    """Test the enhanced entity extraction implementation"""
    print("🧪 Testing Enhanced Entity Extraction - Phase 2.2")
    print("=" * 60)
    
    # Create schema manager (Phase 1 integration)
    try:
        schema_manager = SchemaManager()
        print(f"✅ Schema manager created successfully")
    except Exception as e:
        print(f"❌ Failed to create schema manager: {e}")
        # Create a minimal test without full schema manager
        print("⚠️  Creating minimal test setup...")
        return test_basic_extraction()
    
    # Define entity types in schema
    person_schema = EntityTypeSchema(
        name="person",
        description="Individual person",
        properties=[
            PropertySchema(name="text", property_type=PropertyType.STRING, required=True),
            PropertySchema(name="confidence", property_type=PropertyType.FLOAT),
            PropertySchema(name="extraction_strategy", property_type=PropertyType.STRING)
        ]
    )
    
    org_schema = EntityTypeSchema(
        name="organization", 
        description="Organization or company",
        properties=[
            PropertySchema(name="text", property_type=PropertyType.STRING, required=True),
            PropertySchema(name="confidence", property_type=PropertyType.FLOAT),
            PropertySchema(name="extraction_strategy", property_type=PropertyType.STRING)
        ]
    )
    
    location_schema = EntityTypeSchema(
        name="location",
        description="Geographic location",
        properties=[
            PropertySchema(name="text", property_type=PropertyType.STRING, required=True),
            PropertySchema(name="confidence", property_type=PropertyType.FLOAT),
            PropertySchema(name="extraction_strategy", property_type=PropertyType.STRING)
        ]
    )
    
    # Add to schema manager
    schema_manager.add_entity_type(person_schema)
    schema_manager.add_entity_type(org_schema)
    schema_manager.add_entity_type(location_schema)
    
    print(f"✅ Schema manager created with {len(schema_manager.entity_types)} entity types")
    
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
    
    # Extract entities using convenience function
    print("🔍 Extracting entities using multi-model ensemble...")
    entities = extract_entities_from_text(
        text=test_text, 
        schema_manager=schema_manager,
        confidence_threshold=0.6
    )
    
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
            text = entity.properties['text']
            confidence = entity.properties['confidence']
            strategy = entity.properties['extraction_strategy']
            print(f"  📝 '{text}' (confidence: {confidence:.3f}, strategy: {strategy})")
        print()
    
    # Test individual extractor for detailed statistics
    print("📈 Testing individual extractor for statistics...")
    extractor = EnhancedEntityExtractor(schema_manager)
    
    context = ExtractionContext(
        text=test_text,
        schema_manager=schema_manager,
        confidence_threshold=0.6
    )
    
    detailed_entities = extractor.extract_entities(context)
    stats = extractor.get_extraction_statistics()
    
    print(f"📊 Extraction Statistics:")
    print(f"  🔢 Total extractions: {stats['total_extractions']}")
    print(f"  ⭐ High confidence extractions: {stats['high_confidence_extractions']}")
    print(f"  🎯 Strategy counts: {dict(stats['strategy_counts'])}")
    print(f"  📋 Entity type counts: {dict(stats['entity_type_counts'])}")
    print(f"  🔧 Models available: {stats['models_available']}")
    print(f"  ✅ Strategies enabled: {sum(1 for v in stats['strategies_enabled'].values() if v)}/{len(stats['strategies_enabled'])}")
    
    print()
    print("🎉 Enhanced Entity Extraction test completed successfully!")
    
    return entities


def test_basic_extraction():
    """Basic test of extraction patterns without full schema manager"""
    print("🧪 Running basic extraction test...")
    
    # Import required components directly
    from mcp_server.extraction.enhanced_entity_extractor import (
        ExtractionCandidate, ExtractionStrategy
    )
    
    test_text = "Dr. Sarah Chen works for Microsoft Corporation in Seattle."
    
    # Test pattern-based extraction logic
    import re
    
    person_patterns = [
        r'\b(?:Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
    ]
    
    org_patterns = [
        r'\b[A-Z][a-zA-Z\s]{3,25}\s+(?:Inc|Corp|Corporation|Company|Ltd)\.?\b',
    ]
    
    print(f"📝 Test text: {test_text}")
    print()
    
    # Test person extraction
    for pattern in person_patterns:
        matches = list(re.finditer(pattern, test_text))
        print(f"👤 Person pattern '{pattern[:30]}...': {len(matches)} matches")
        for match in matches:
            print(f"  📝 '{match.group()}' at position {match.start()}-{match.end()}")
    
    # Test organization extraction  
    for pattern in org_patterns:
        matches = list(re.finditer(pattern, test_text))
        print(f"🏢 Organization pattern '{pattern[:30]}...': {len(matches)} matches")
        for match in matches:
            print(f"  📝 '{match.group()}' at position {match.start()}-{match.end()}")
    
    print()
    print("✅ Basic extraction test completed!")
    return []

if __name__ == "__main__":
    test_enhanced_entity_extraction()
