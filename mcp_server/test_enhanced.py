#!/usr/bin/env python3
"""
Test script to verify the enhanced knowledge graph processor
"""

import sys
sys.path.append('/app')

try:
    from enhanced_knowledge_processor import create_enhanced_knowledge_graph_processor
    
    # Create processor
    processor = create_enhanced_knowledge_graph_processor()
    print("✅ Enhanced processor loaded successfully")
    
    # Test with Abel's text
    test_text = """Hi, I'm Abel. I'm in my mid-thirties and I work as an urban planner for the city council — I spend my days figuring out how to make neighborhoods more livable, greener, and better connected. It's rewarding work, especially when I get to see new parks or bike paths come to life."""
    
    entities, relationships = processor.process_text_enhanced(test_text)
    
    print(f"📊 Enhanced Results:")
    print(f"   Entities: {len(entities)}")
    print(f"   Relationships: {len(relationships)}")
    
    print(f"\n🏷️  Entities:")
    for entity in entities:
        print(f"   - {entity.name} ({entity.entity_type}) [score: {entity.importance_score:.2f}]")
    
    print(f"\n🔗 Relationships:")
    for rel in relationships:
        print(f"   - {rel.source} --{rel.relation_type}--> {rel.target} [confidence: {rel.confidence:.2f}]")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
