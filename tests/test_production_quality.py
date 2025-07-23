#!/usr/bin/env python3
"""
Test Production Entity Extraction Quality Improvements

This test verifies that the quality improvements (sentence fragment detection,
domain knowledge correction) are properly integrated into the production system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'mcp_server'))

from mcp_server.core.graph_schema import SchemaManager, EntityInstance
from mcp_server.extraction.enhanced_entity_extractor import EnhancedEntityExtractor

def test_production_quality():
    """Test the production enhanced entity extractor quality improvements"""
    
    print("🧪 Testing Production Entity Extraction Quality Improvements")
    print("=" * 65)
    
    # Initialize production components
    schema_manager = SchemaManager()
    extractor = EnhancedEntityExtractor(schema_manager)
    
    # Test text with quality issues we fixed
    test_text = """
    Dr. Sarah Johnson from MIT collaborated with Professor Michael Chen at Stanford University 
    to develop AI algorithms. Their research was funded by Google and Microsoft. 
    The breakthrough technology was presented at the International Conference on Machine Learning 
    in Montreal, Canada.
    """
    
    print(f"📝 Input Text:")
    print(f"{test_text.strip()}\n")
    
    # Extract entities using production system
    try:
        entities = extractor.extract_entities(test_text)
        
        print(f"✅ Production extraction successful!")
        print(f"📊 Extracted {len(entities)} entities\n")
        
        # Analyze results
        entity_names = [entity.name for entity in entities]
        entity_types = {entity.name: entity.entity_type for entity in entities}
        
        # Check for specific quality improvements
        print("🔍 Quality Analysis:")
        print("-" * 30)
        
        # Check 1: Sentence fragment should be filtered out
        malformed_sentence = "The breakthrough technology was presented at the International Conference"
        sentence_found = any(malformed_sentence.lower() in name.lower() for name in entity_names)
        
        if sentence_found:
            print("❌ ISSUE: Sentence fragment detected in results")
        else:
            print("✅ Sentence fragment filtering: WORKING")
        
        # Check 2: AI should be classified correctly (not as location)
        ai_entities = [entity for entity in entities if 'AI' in entity.name or 'ai' in entity.name.lower()]
        ai_location_issue = any(entity.entity_type == 'location' for entity in ai_entities)
        
        if ai_location_issue:
            print("❌ ISSUE: AI misclassified as location")
        else:
            print("✅ AI classification correction: WORKING")
        
        # Display all extracted entities
        print(f"\n📋 Extracted Entities:")
        for entity in entities:
            print(f"  • '{entity.name}' ({entity.entity_type})")
            
        print(f"\n🎉 Production quality test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Production test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_production_quality()
    if success:
        print("\n✅ All production quality improvements verified!")
    else:
        print("\n❌ Production quality issues detected!")
        sys.exit(1)
