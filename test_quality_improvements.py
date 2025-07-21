#!/usr/bin/env python3
"""
Enhanced Entity Extraction Quality Test
Tests the improved deduplication and validation fixes
"""

import os
import sys

# Add the mcp_server directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp_server'))

def test_quality_improvements():
    """Test the quality improvements in enhanced entity extraction"""
    
    print("🧪 Enhanced Entity Extraction - Quality Improvements Test")
    print("=" * 65)
    
    try:
        from core.graph_schema import GraphSchema
        from extraction.enhanced_entity_extractor import EnhancedEntityExtractor
        
        # Initialize
        schema = GraphSchema()
        extractor = EnhancedEntityExtractor()
        
        print("✅ Successfully imported enhanced extractor")
        
        # Test text that previously caused issues
        test_text = """
        Dr. Sarah Johnson from MIT collaborated with Professor Michael Chen at Stanford University 
        to develop AI algorithms. Their research was funded by Google and Microsoft, with additional 
        support from the National Science Foundation. The team worked in Silicon Valley and Boston, 
        publishing their findings in the Journal of AI Research. Sarah's startup, NeuralTech Inc., 
        was later acquired by Apple for $50 million. The breakthrough technology was presented at 
        the International Conference on Machine Learning in Montreal, Canada, where it received 
        widespread acclaim from experts like Dr. Elena Rodriguez from Cambridge University.
        """
        
        print(f"📝 Test Text: {len(test_text)} characters")
        
        # Extract entities using proper enhanced extractor (not demo)
        entities = extractor.extract_entities(test_text, schema)
        
        print(f"\n🎯 EXTRACTION RESULTS")
        print("-" * 40)
        print(f"Total entities extracted: {len(entities)}")
        
        if not entities:
            print("⚠️  No entities extracted!")
            return
        
        # Check for specific quality issues
        print(f"\n🔍 QUALITY ANALYSIS")
        print("-" * 40)
        
        # Test 1: Check for duplicate names like "Sarah" and "Sarah Johnson"
        person_names = [e['name'] for e in entities if e.get('entity_type') == 'person']
        print(f"Person entities found: {person_names}")
        
        # Check for substring containment issues
        duplicate_issues = []
        for i, name1 in enumerate(person_names):
            for j, name2 in enumerate(person_names):
                if i != j and (name1.lower() in name2.lower() or name2.lower() in name1.lower()):
                    duplicate_issues.append((name1, name2))
        
        if duplicate_issues:
            print(f"❌ Found {len(duplicate_issues)} potential duplicates:")
            for name1, name2 in duplicate_issues:
                print(f"  - '{name1}' vs '{name2}'")
        else:
            print("✅ No duplicate person names found")
        
        # Test 2: Check for malformed extractions like "Apple for"
        org_names = [e['name'] for e in entities if e.get('entity_type') == 'organization']
        print(f"\nOrganization entities found: {org_names}")
        
        malformed_orgs = []
        for name in org_names:
            if name.lower().endswith(' for') or name.lower().endswith(' by') or name.lower().endswith(' with'):
                malformed_orgs.append(name)
        
        if malformed_orgs:
            print(f"❌ Found {len(malformed_orgs)} malformed organizations:")
            for name in malformed_orgs:
                print(f"  - '{name}'")
        else:
            print("✅ No malformed organization names found")
        
        # Test 3: Check overall quality metrics
        print(f"\n📊 QUALITY METRICS")
        print("-" * 40)
        
        # Group by type
        entity_types = {}
        for entity in entities:
            entity_type = entity.get('entity_type', 'unknown')
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity)
        
        for entity_type, type_entities in entity_types.items():
            print(f"  {entity_type}: {len(type_entities)} entities")
            
            # Show confidence distribution
            confidences = [e.get('confidence', 0) for e in type_entities]
            if confidences:
                avg_conf = sum(confidences) / len(confidences)
                min_conf = min(confidences)
                max_conf = max(confidences)
                print(f"    Confidence: {min_conf:.2f} - {max_conf:.2f} (avg: {avg_conf:.2f})")
        
        # Test 4: Show extraction strategies used
        strategies = {}
        for entity in entities:
            strategy = entity.get('extracted_by', 'unknown')
            strategies[strategy] = strategies.get(strategy, 0) + 1
        
        print(f"\n🔧 EXTRACTION STRATEGIES")
        print("-" * 40)
        for strategy, count in strategies.items():
            percentage = (count / len(entities)) * 100
            print(f"  {strategy}: {count} entities ({percentage:.1f}%)")
        
        # Test 5: Detailed entity listing
        print(f"\n📋 DETAILED ENTITY LISTING")
        print("-" * 40)
        
        for entity_type, type_entities in sorted(entity_types.items()):
            print(f"\n{entity_type.upper()} ({len(type_entities)} entities):")
            for entity in type_entities:
                name = entity['name']
                confidence = entity.get('confidence', 0)
                strategy = entity.get('extracted_by', 'unknown')
                print(f"  - '{name}' (conf: {confidence:.2f}, by: {strategy})")
        
        print(f"\n🎉 Quality test completed!")
        
        # Summary
        total_issues = len(duplicate_issues) + len(malformed_orgs)
        if total_issues == 0:
            print("✅ All quality checks passed - no duplicates or malformed extractions found!")
        else:
            print(f"⚠️  Found {total_issues} quality issues that need attention")
        
        return total_issues == 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This may be due to the Phase 1/Phase 2 integration issue.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quality_improvements()
    exit(0 if success else 1)
