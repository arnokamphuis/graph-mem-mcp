#!/usr/bin/env python3
"""
Test script to verify the cleaned up main.py and modern knowledge graph processor work correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'mcp_server'))

# Test the cleaned functions without starting the server
def test_extract_relationships():
    """Test the cleaned up extract_relationships function"""
    from mcp_server.main import extract_relationships, extract_advanced_entities
    
    # Test text about quantum computing relationships
    test_text = """
    Quantum computing uses quantum bits called qubits to process information. 
    IBM developed quantum computers that leverage quantum superposition.
    Quantum algorithms like Shor's algorithm can factor large numbers efficiently.
    """
    
    print("Testing entity extraction...")
    entities = extract_advanced_entities(test_text)
    print(f"Extracted entities: {list(entities.keys())}")
    
    print("\nTesting relationship extraction...")
    relationships = extract_relationships(test_text, entities)
    print(f"Extracted {len(relationships)} relationships:")
    
    for rel in relationships[:5]:  # Show first 5 relationships
        print(f"  {rel['from']} --{rel['type']}--> {rel['to']} (confidence: {rel['confidence']})")
        if 'context' in rel:
            print(f"    Context: {rel['context'][:100]}...")
        
    return entities, relationships

def test_modern_kg_processor():
    """Test the modern knowledge graph processor if available"""
    try:
        from mcp_server.knowledge_graph_processor import ModernKnowledgeGraphProcessor
        
        print("\n=== Testing Modern Knowledge Graph Processor ===")
        processor = ModernKnowledgeGraphProcessor()
        
        test_text = """
        Quantum entanglement enables quantum computers to perform complex calculations.
        Google's Sycamore processor achieved quantum supremacy in 2019.
        Quantum error correction protects quantum information from decoherence.
        """
        
        result = processor.construct_knowledge_graph(test_text, [])
        
        print(f"Modern processor extracted {len(result['entities'])} entities:")
        for entity in result['entities'][:5]:  # Show first 5
            print(f"  {entity.name} ({entity.entity_type}) - confidence: {entity.confidence}")
            
        print(f"\nModern processor extracted {len(result['relationships'])} relationships:")
        for rel in result['relationships'][:5]:  # Show first 5
            print(f"  {rel.source} --{rel.relation_type}--> {rel.target} (confidence: {rel.confidence})")
            
        print(f"\nStats: {result['stats']}")
        return result
        
    except ImportError as e:
        print(f"\nModern KG processor not available: {e}")
        return None

if __name__ == "__main__":
    print("Testing cleaned up knowledge graph code...")
    
    try:
        entities, relationships = test_extract_relationships()
        
        # Check for the dreaded "related_to" dominance
        relationship_types = [rel['type'] for rel in relationships]
        related_to_count = relationship_types.count('related_to')
        total_relationships = len(relationship_types)
        
        print(f"\n=== Relationship Type Analysis ===")
        print(f"Total relationships: {total_relationships}")
        print(f"'related_to' relationships: {related_to_count}")
        if total_relationships > 0:
            percentage = (related_to_count / total_relationships) * 100
            print(f"Percentage of 'related_to': {percentage:.1f}%")
            
            if percentage > 80:
                print("⚠️  WARNING: Still too many 'related_to' relationships!")
            elif percentage > 50:
                print("⚠️  CAUTION: Many 'related_to' relationships")
            else:
                print("✅ Good: Diverse relationship types")
        
        # Show unique relationship types
        unique_types = set(relationship_types)
        print(f"Unique relationship types: {unique_types}")
        
        # Test modern processor
        modern_result = test_modern_kg_processor()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
