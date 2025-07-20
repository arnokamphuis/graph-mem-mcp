#!/usr/bin/env python3
"""
Test the modern knowledge graph processor directly.
"""

import sys
import os

# Test the modern KG processor without importing main.py
def test_modern_kg_processor_direct():
    """Test the modern knowledge graph processor directly"""
    try:
        # Try importing the processor
        sys.path.append('mcp_server')
        from knowledge_graph_processor import ModernKnowledgeGraphProcessor
        
        print("=== Testing Modern Knowledge Graph Processor ===")
        
        # Test without spacy for now
        try:
            processor = ModernKnowledgeGraphProcessor()
            print("✅ ModernKnowledgeGraphProcessor initialized successfully")
        except Exception as e:
            print(f"⚠️  ModernKnowledgeGraphProcessor initialization failed: {e}")
            return None
        
        test_text = """
        Quantum computing uses quantum bits called qubits to process information.
        IBM developed quantum computers that leverage quantum superposition.
        Quantum algorithms like Shor's algorithm can factor large numbers efficiently.
        Google's Sycamore processor achieved quantum supremacy in 2019.
        """
        
        print(f"Processing text: {test_text[:100]}...")
        
        try:
            result = processor.construct_knowledge_graph(test_text, [])
            
            print(f"✅ Successfully processed text")
            print(f"Extracted {len(result['entities'])} entities:")
            for entity in result['entities'][:5]:  # Show first 5
                print(f"  {entity.name} ({entity.entity_type}) - confidence: {entity.confidence:.2f}")
                
            print(f"\nExtracted {len(result['relationships'])} relationships:")
            for rel in result['relationships'][:5]:  # Show first 5
                print(f"  {rel.source} --{rel.relation_type}--> {rel.target} (confidence: {rel.confidence:.2f})")
                
            print(f"\nStats: {result['stats']}")
            
            # Check relationship diversity
            relationship_types = [rel.relation_type for rel in result['relationships']]
            related_to_count = relationship_types.count('related_to')
            total_relationships = len(relationship_types)
            
            print(f"\n=== Relationship Quality Analysis ===")
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
            print(f"Unique relationship types: {sorted(unique_types)}")
            
            return result
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    except ImportError as e:
        print(f"❌ Could not import ModernKnowledgeGraphProcessor: {e}")
        return None

def test_relationship_diversity():
    """Test that we get diverse relationships, not just 'related_to'"""
    print("\n=== Testing Relationship Diversity ===")
    
    # Test with clear semantic relationships
    test_cases = [
        "IBM created quantum computers in their research labs.",
        "Quantum computers use qubits to perform calculations.",
        "Shor's algorithm is a type of quantum algorithm.",
        "Google achieved quantum supremacy with their Sycamore processor.",
        "Quantum entanglement enables quantum computing applications."
    ]
    
    try:
        sys.path.append('mcp_server')
        from knowledge_graph_processor import ModernKnowledgeGraphProcessor
        processor = ModernKnowledgeGraphProcessor()
        
        all_relationships = []
        for text in test_cases:
            print(f"\nTesting: {text}")
            result = processor.construct_knowledge_graph(text, [])
            for rel in result['relationships']:
                all_relationships.append(rel.relation_type)
                print(f"  {rel.source} --{rel.relation_type}--> {rel.target}")
        
        # Analyze overall diversity
        relationship_counts = {}
        for rel_type in all_relationships:
            relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
        
        print(f"\n=== Overall Relationship Analysis ===")
        for rel_type, count in sorted(relationship_counts.items()):
            percentage = (count / len(all_relationships)) * 100
            print(f"{rel_type}: {count} ({percentage:.1f}%)")
        
        related_to_percentage = (relationship_counts.get('related_to', 0) / len(all_relationships)) * 100
        if related_to_percentage < 30:
            print("✅ EXCELLENT: Good relationship diversity!")
        elif related_to_percentage < 60:
            print("✅ GOOD: Acceptable relationship diversity")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Too many generic relationships")
            
    except Exception as e:
        print(f"❌ Diversity test failed: {e}")

if __name__ == "__main__":
    print("Testing Modern Knowledge Graph Processor...")
    
    result = test_modern_kg_processor_direct()
    
    if result:
        test_relationship_diversity()
        print("\n🎉 Modern Knowledge Graph Processor working successfully!")
    else:
        print("\n❌ Tests failed - please check dependencies")
