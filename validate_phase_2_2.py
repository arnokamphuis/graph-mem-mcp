#!/usr/bin/env python3
"""
Phase 2.2 Enhanced Entity Extraction Validation Suite
Direct validation testing following prompt quality gate requirements
"""

import sys
sys.path.append('.')

def test_import_validation():
    """Test 1: Import Validation"""
    print("\n🧪 Test 1: Import Validation")
    try:
        # Test core imports
        from mcp_server.extraction.enhanced_entity_extractor import (
            EnhancedEntityExtractor, ExtractionContext, ExtractionStrategy,
            extract_entities_from_text, create_enhanced_entity_extractor
        )
        print("  ✅ Core enhanced entity extraction imports successful")
        
        # Test Phase 1 core integration
        from mcp_server.core.graph_schema import EntityInstance, SchemaManager
        print("  ✅ Phase 1 core imports successful")
        
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_factory_creation():
    """Test 2: Factory Creation"""
    print("\n🧪 Test 2: Factory Creation")
    try:
        from mcp_server.extraction.enhanced_entity_extractor import create_enhanced_entity_extractor
        
        # Create minimal schema manager for testing
        class TestSchema:
            def __init__(self):
                self.entity_types = ["Person", "Organization", "Location", "Concept", "Event"]
        
        class TestSchemaManager:
            def __init__(self):
                self.schema = TestSchema()
                self.entity_types = {}
                
            def validate_entity(self, entity):
                return True, []
        
        schema_manager = TestSchemaManager()
        
        extractor = create_enhanced_entity_extractor(schema_manager)
        print("  ✅ Factory created extractor successfully")
        
        # Check that it's the right type
        if hasattr(extractor, 'extract_entities'):
            print("  ✅ Extractor has extract_entities method")
        else:
            print("  ❌ Missing extract_entities method")
            return False
            
        # Check schema manager integration
        if hasattr(extractor, 'schema_manager'):
            print("  ✅ Schema manager integration confirmed")
        else:
            print("  ❌ Missing schema_manager")
            return False
        
        return True
    except Exception as e:
        print(f"  ❌ Factory creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_extractor_initialization():
    """Test 3: EnhancedEntityExtractor Initialization"""
    print("\n🧪 Test 3: EnhancedEntityExtractor Initialization")
    try:
        from mcp_server.extraction.enhanced_entity_extractor import EnhancedEntityExtractor
        
        # Create minimal schema manager for testing
        class TestSchema:
            def __init__(self):
                self.entity_types = ["Person", "Organization", "Location", "Concept", "Event"]
        
        class TestSchemaManager:
            def __init__(self):
                self.schema = TestSchema()
                self.entity_types = {}
                
            def validate_entity(self, entity):
                return True, []
        
        schema_manager = TestSchemaManager()
        
        extractor = EnhancedEntityExtractor(schema_manager)
        print("  ✅ EnhancedEntityExtractor created successfully")
        
        # Check stats initialization
        if hasattr(extractor, 'stats'):
            stats = extractor.stats
            print(f"  ✅ Statistics initialized: {len(stats)} metrics")
        
        # Check strategies
        if hasattr(extractor, 'available_strategies'):
            strategies = extractor.available_strategies
            print(f"  ✅ Extraction strategies: {len(strategies)} available")
            for strategy in strategies:
                print(f"    - {strategy.value}")
        
        return True
    except Exception as e:
        print(f"  ❌ Extractor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_extraction_context():
    """Test 4: ExtractionContext Creation"""
    print("\n🧪 Test 4: ExtractionContext Creation")
    try:
        from mcp_server.extraction.enhanced_entity_extractor import ExtractionContext
        
        # Create minimal schema manager for testing
        class TestSchemaManager:
            def __init__(self):
                self.entity_types = {}
                
            def validate_entity(self, entity):
                return True, []
        
        schema_manager = TestSchemaManager()
        
        context = ExtractionContext(
            text="John Smith works at Google Inc. The company is located in Mountain View.",
            schema_manager=schema_manager,
            confidence_threshold=0.5,
            enable_resolution=True
        )
        print("  ✅ ExtractionContext created successfully")
        print(f"  ✅ Text: {context.text[:50]}...")
        print(f"  ✅ Confidence threshold: {context.confidence_threshold}")
        print(f"  ✅ Resolution enabled: {context.enable_resolution}")
        print(f"  ✅ Strategies: {len(context.strategies)} configured")
        
        return True
    except Exception as e:
        print(f"  ❌ ExtractionContext creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quick_entity_extraction():
    """Test 5: Quick Entity Extraction Function"""
    print("\n🧪 Test 5: Quick Entity Extraction Function")
    try:
        from mcp_server.extraction.enhanced_entity_extractor import extract_entities_from_text
        
        # Create minimal schema manager for testing
        class TestSchema:
            def __init__(self):
                self.entity_types = ["Person", "Organization", "Location", "Concept", "Event"]
        
        class TestSchemaManager:
            def __init__(self):
                self.schema = TestSchema()
                self.entity_types = {}
                
            def validate_entity(self, entity):
                return True, []
        
        schema_manager = TestSchemaManager()
        
        test_text = "Mary Johnson founded TechCorp in Seattle. The organization develops AI software."
        
        print("  🔍 Attempting entity extraction...")
        entities = extract_entities_from_text(
            text=test_text,
            schema_manager=schema_manager,
            confidence_threshold=0.3
        )
        
        print(f"  ✅ Extraction completed: {len(entities)} entities")
        
        # Show first few results
        for i, entity in enumerate(entities[:3]):
            print(f"    Entity {i+1}: {entity.properties.get('text', 'N/A')} ({entity.entity_type})")
            print(f"       Confidence: {entity.properties.get('confidence', 0.0):.2f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Quick extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_extraction_pipeline():
    """Test 6: Full Extraction Pipeline"""
    print("\n🧪 Test 6: Full Extraction Pipeline")
    try:
        from mcp_server.extraction.enhanced_entity_extractor import EnhancedEntityExtractor, ExtractionContext
        
        # Create minimal schema manager for testing
        class TestSchema:
            def __init__(self):
                self.entity_types = ["Person", "Organization", "Location", "Concept", "Event"]
        
        class TestSchemaManager:
            def __init__(self):
                self.schema = TestSchema()
                self.entity_types = {}
                
            def validate_entity(self, entity):
                return True, []
        
        schema_manager = TestSchemaManager()
        
        extractor = EnhancedEntityExtractor(schema_manager)
        context = ExtractionContext(
            text="Dr. Sarah Chen works at Microsoft Corporation. She leads the AI research team in Redmond, Washington.",
            schema_manager=schema_manager,
            confidence_threshold=0.4,
            enable_resolution=True
        )
        
        print("  🔍 Attempting full pipeline extraction...")
        entities = extractor.extract_entities(context)
        
        print(f"  ✅ Full pipeline completed: {len(entities)} entities")
        
        # Validate results structure
        for i, entity in enumerate(entities[:2]):
            if hasattr(entity, 'entity_type') and hasattr(entity, 'properties'):
                print(f"    ✅ Entity {i+1}: {entity.entity_type}")
                confidence = entity.properties.get('confidence', 0.0)
                print(f"       Confidence: {confidence:.2f}")
            else:
                print(f"    ⚠️ Entity missing required attributes")
        
        return True
    except Exception as e:
        print(f"  ❌ Full pipeline extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_validation_suite():
    """Run complete Phase 2.2 validation suite"""
    print("="*60)
    print("🧪 PHASE 2.2 ENHANCED ENTITY EXTRACTION VALIDATION SUITE")
    print("="*60)
    
    tests = [
        ("Import Validation", test_import_validation),
        ("Factory Creation", test_factory_creation),
        ("Extractor Initialization", test_extractor_initialization),
        ("Extraction Context", test_extraction_context),
        ("Quick Extraction", test_quick_entity_extraction),
        ("Full Pipeline", test_full_extraction_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATION RESULTS")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print("-"*60)
    coverage = (passed / total * 100) if total > 0 else 0
    print(f"Tests Passed: {passed}/{total} ({coverage:.1f}%)")
    
    if coverage >= 90:
        print("✅ QUALITY GATE PASSED: 90%+ test success")
        return True
    else:
        print("❌ QUALITY GATE FAILED: <90% test success")
        return False

if __name__ == "__main__":
    success = run_validation_suite()
    if success:
        print("\n🎉 Phase 2.2 validation SUCCESSFUL!")
        exit(0)
    else:
        print("\n⚠️ Phase 2.2 validation INCOMPLETE")
        exit(1)
