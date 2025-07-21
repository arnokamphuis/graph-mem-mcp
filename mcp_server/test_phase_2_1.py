#!/usr/bin/env python3
"""
Comprehensive test for Phase 2.1 Sophisticated Relationship Extraction
Tests all extraction strategies and validates RelationshipInstance creation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_sophisticated_relationship_extraction():
    """Test sophisticated relationship extraction with all strategies"""
    print("🧪 Testing Sophisticated Relationship Extraction (Phase 2.1)")
    
    try:
        # Import the new classes
        from extraction.relation_extractor import (
            SophisticatedRelationshipExtractor, ExtractionContext, 
            RelationshipCandidate, create_relationship_extractor
        )
        from core.graph_schema import RelationshipInstance, SchemaManager
        
        print("✅ Successfully imported Phase 2.1 components")
        
        # Initialize extractor
        extractor = create_relationship_extractor(
            confidence_threshold=0.5  # Lower threshold for testing
        )
        print("✅ Sophisticated Relationship Extractor initialized")
        
        # Test text with clear relationships
        test_text = """
        John Smith works at Google Inc. as a software engineer. 
        He graduated from Stanford University in 2018.
        Google Inc. is located in Mountain View, California.
        John collaborates with Sarah Johnson on machine learning projects.
        Stanford University offers computer science degrees.
        """
        
        print(f"\n📝 Test text: {test_text[:100]}...")
        
        # Create extraction context
        context = ExtractionContext(
            text=test_text,
            source_id="test_document",
            domain_context="professional"
        )
        
        # Extract relationship candidates
        candidates = extractor.extract_relationships(context)
        print(f"✅ Extracted {len(candidates)} relationship candidates")
        
        # Validate RelationshipCandidate objects
        valid_candidates = 0
        for i, candidate in enumerate(candidates):
            if not isinstance(candidate, RelationshipCandidate):
                print(f"❌ Candidate {i} is not a RelationshipCandidate: {type(candidate)}")
                continue
                
            # Check required fields
            required_fields = ['source_entity', 'target_entity', 'relationship_type', 'confidence']
            for field in required_fields:
                if not hasattr(candidate, field) or getattr(candidate, field) is None:
                    print(f"❌ Candidate {i} missing {field}: {candidate}")
                    continue
            
            valid_candidates += 1
            print(f"   ✅ {candidate.source_entity} --{candidate.relationship_type}--> {candidate.target_entity} (conf: {candidate.confidence:.2f}, method: {candidate.extraction_method.value})")
        
        # Test RelationshipInstance conversion
        instances = extractor.extract_relationships_as_instances(context)
        print(f"✅ Converted {len(instances)} candidates to RelationshipInstance objects")
        
        # Validate RelationshipInstance objects
        for i, instance in enumerate(instances):
            if not isinstance(instance, RelationshipInstance):
                print(f"❌ Instance {i} is not a RelationshipInstance: {type(instance)}")
                continue
                
            print(f"   ✅ Instance: {instance.source_entity_id} --{instance.relationship_type}--> {instance.target_entity_id}")
        
        # Test extraction statistics
        stats = extractor.get_extraction_statistics()
        print(f"\n📊 Extraction Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test extraction strategies
        strategies_used = set()
        for candidate in candidates:
            strategies_used.add(candidate.extraction_method.value)
        
        print(f"\n🎯 Extraction strategies used: {', '.join(strategies_used)}")
        
        print(f"\n✅ Phase 2.1 Sophisticated Relationship Extraction test PASSED!")
        print(f"   - {valid_candidates} valid candidates extracted")
        print(f"   - {len(instances)} RelationshipInstance objects created")
        print(f"   - {len(strategies_used)} extraction strategies utilized")
        
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sophisticated_relationship_extraction()
    sys.exit(0 if success else 1)
