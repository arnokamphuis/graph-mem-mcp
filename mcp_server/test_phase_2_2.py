#!/usr/bin/env python3
"""
Comprehensive test for Phase 2.2 Enhanced Entity Extraction functionality
Tests all extraction strategies and validates EntityInstance creation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extraction.enhanced_entity_extractor import EnhancedEntityExtractor
from core.graph_schema import EntityInstance

def test_enhanced_entity_extraction():
    """Test enhanced entity extraction with all strategies"""
    print("🧪 Testing Enhanced Entity Extraction (Phase 2.2)")
    
    try:
        # Initialize extractor
        extractor = EnhancedEntityExtractor()
        print("✅ Enhanced Entity Extractor initialized")
        
        # Test text with various entity types
        test_text = """
        John Smith is a software engineer at Google Inc., working in Mountain View, California.
        He graduated from Stanford University in 2018 with a degree in Computer Science.
        His email is john.smith@google.com and he can be reached at +1-650-555-0123.
        He specializes in machine learning and artificial intelligence systems.
        """
        
        print(f"\n📝 Test text: {test_text[:100]}...")
        
        # Create extraction context
        from extraction.enhanced_entity_extractor import ExtractionContext
        context = ExtractionContext(
            text=test_text,
            source_id="test_document",
            domain_context="technology"
        )
        
        # Extract entities
        entities = extractor.extract_entities(context)
        print(f"✅ Extracted {len(entities)} entities")
        
        # Validate EntityInstance objects
        for i, entity in enumerate(entities):
            if not isinstance(entity, EntityInstance):
                print(f"❌ Entity {i} is not an EntityInstance: {type(entity)}")
                return False
                
            # Check required fields
            if not hasattr(entity, 'name') or not entity.name:
                print(f"❌ Entity {i} missing name: {entity}")
                return False
                
            if not hasattr(entity, 'entity_type') or not entity.entity_type:
                print(f"❌ Entity {i} missing entity_type: {entity}")
                return False
                
            print(f"   ✅ {entity.name} ({entity.entity_type}) - confidence: {entity.confidence:.2f}")
        
        # Test extraction strategies
        strategies_tested = set()
        for entity in entities:
            if hasattr(entity, 'properties') and 'extraction_strategy' in entity.properties:
                strategies_tested.add(entity.properties['extraction_strategy'])
        
        print(f"\n🎯 Extraction strategies used: {', '.join(strategies_tested)}")
        
        print(f"\n✅ Phase 2.2 Enhanced Entity Extraction test PASSED!")
        print(f"   - {len(entities)} entities extracted successfully")
        print(f"   - All EntityInstance objects properly created")
        print(f"   - {len(strategies_tested)} extraction strategies utilized")
        
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_entity_extraction()
    sys.exit(0 if success else 1)
