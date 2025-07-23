#!/usr/bin/env python3
"""
Enhanced Entity Extractor Test with Detailed Output
Tests all extraction strategies and shows complete entity information
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'mcp_server'))

from extraction.enhanced_entity_extractor import EnhancedEntityExtractor
from core.graph_schema import GraphSchema

def test_enhanced_extraction():
    """Test enhanced entity extraction with detailed output"""
    
    # Initialize
    schema = GraphSchema()
    extractor = EnhancedEntityExtractor()
    
    # Test text with diverse entities
    test_text = """
    Dr. Sarah Johnson from MIT collaborated with Professor Michael Chen at Stanford University 
    to develop AI algorithms. Their research was funded by Google and Microsoft, with additional 
    support from the National Science Foundation. The team worked in Silicon Valley and Boston, 
    publishing their findings in the Journal of AI Research. Sarah's startup, NeuralTech Inc., 
    was later acquired by Apple for $50 million. The breakthrough technology was presented at 
    the International Conference on Machine Learning in Montreal, Canada, where it received 
    widespread acclaim from experts like Dr. Elena Rodriguez from Cambridge University.
    """
    
    print("Enhanced Entity Extraction Test - Detailed Output")
    print("=" * 60)
    print(f"Input Text:\n{test_text}\n")
    
    # Extract entities
    entities = extractor.extract_entities(test_text, schema)
    
    print(f"EXTRACTION RESULTS")
    print("-" * 30)
    print(f"Total entities extracted: {len(entities)}")
    
    if not entities:
        print("⚠️  No entities extracted!")
        return
    
    # Group entities by type
    entity_groups = {}
    for entity in entities:
        entity_type = entity.get('entity_type', 'unknown')
        if entity_type not in entity_groups:
            entity_groups[entity_type] = []
        entity_groups[entity_type].append(entity)
    
    # Display by type with detailed information
    for entity_type, type_entities in entity_groups.items():
        print(f"\n📋 {entity_type.upper()} ({len(type_entities)} entities)")
        print("-" * 40)
        
        for i, entity in enumerate(type_entities, 1):
            print(f"  {i}. Name: '{entity['name']}'")
            print(f"     Type: {entity['entity_type']}")
            print(f"     Confidence: {entity.get('confidence', 'N/A')}")
            
            # Show extraction evidence
            if 'extracted_by' in entity:
                print(f"     Extracted by: {entity['extracted_by']}")
            
            if 'context' in entity:
                context = entity['context']
                if len(context) > 60:
                    context = context[:60] + "..."
                print(f"     Context: \"{context}\"")
            
            # Show additional metadata
            if 'attributes' in entity and entity['attributes']:
                print(f"     Attributes: {entity['attributes']}")
            
            print()
    
    # Show extraction strategy statistics
    print(f"\nEXTRACTION STRATEGY ANALYSIS")
    print("-" * 30)
    
    strategy_counts = {}
    for entity in entities:
        strategy = entity.get('extracted_by', 'unknown')
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    total_entities = len(entities)
    for strategy, count in strategy_counts.items():
        percentage = (count / total_entities) * 100
        print(f"  {strategy}: {count} entities ({percentage:.1f}%)")
    
    # Test individual strategies
    print(f"\nSTRATEGY AVAILABILITY CHECK")
    print("-" * 30)
    
    strategies = [
        ('schema_guided', 'Schema-guided extraction'),
        ('pattern_based', 'Pattern-based extraction'),
        ('contextual', 'Contextual extraction'),
        ('transformer_ner', 'Transformer NER'),
        ('spacy_ner', 'spaCy NER')
    ]
    
    for strategy_key, strategy_name in strategies:
        if hasattr(extractor, f'_{strategy_key}_extraction'):
            method = getattr(extractor, f'_{strategy_key}_extraction')
            try:
                test_result = method(test_text, schema)
                status = f"✅ Available ({len(test_result)} entities)"
            except Exception as e:
                status = f"❌ Error: {str(e)[:50]}..."
        else:
            status = "❌ Method not found"
        
        print(f"  {strategy_name}: {status}")
    
    # Quality metrics
    print(f"\nQUALITY METRICS")
    print("-" * 30)
    
    # Entity diversity
    unique_names = set(entity['name'] for entity in entities)
    print(f"  Unique entities: {len(unique_names)} out of {len(entities)}")
    
    # Confidence distribution
    confidences = [entity.get('confidence', 0) for entity in entities if entity.get('confidence') is not None]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        print(f"  Confidence range: {min_confidence:.2f} - {max_confidence:.2f} (avg: {avg_confidence:.2f})")
    
    # Context coverage
    entities_with_context = sum(1 for entity in entities if entity.get('context'))
    print(f"  Entities with context: {entities_with_context}/{len(entities)} ({entities_with_context/len(entities)*100:.1f}%)")
    
    print(f"\nTest completed successfully! 🎉")

if __name__ == "__main__":
    test_enhanced_extraction()
