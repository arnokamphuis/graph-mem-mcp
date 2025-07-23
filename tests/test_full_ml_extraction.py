#!/usr/bin/env python3
"""
Standalone Enhanced Entity Extractor Test
Works around import issues to test the full multi-model ensemble
"""

import os
import sys

# Add the mcp_server directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp_server'))

# Import core components first
from core.graph_schema import GraphSchema

# Create minimal test for enhanced extractor
def test_with_all_dependencies():
    """Test extraction with all ML dependencies available"""
    
    print("🧪 Enhanced Entity Extraction Test with All Dependencies")
    print("=" * 65)
    
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
    
    print(f"📝 Input Text:")
    print(f"{test_text.strip()}\n")
    
    # Test dependency availability
    print("🔍 DEPENDENCY CHECK")
    print("-" * 30)
    
    dependencies = [
        ('transformers', 'Hugging Face Transformers'),
        ('spacy', 'spaCy NLP'),
        ('torch', 'PyTorch'),
        ('sentence_transformers', 'Sentence Transformers')
    ]
    
    all_available = True
    for dep_name, dep_description in dependencies:
        try:
            __import__(dep_name)
            print(f"  ✅ {dep_description}: Available")
        except ImportError:
            print(f"  ❌ {dep_description}: Missing")
            all_available = False
    
    if not all_available:
        print("\n⚠️  Some dependencies are missing. Install with:")
        print("pip install transformers spacy torch sentence-transformers")
        return
    
    # Now try to import the enhanced extractor
    print(f"\n🚀 ENHANCED EXTRACTION TEST")
    print("-" * 30)
    
    try:
        # Import the enhanced extractor directly
        from extraction.enhanced_entity_extractor import EnhancedEntityExtractor
        
        # Initialize
        schema = GraphSchema()
        extractor = EnhancedEntityExtractor()
        
        print("✅ Enhanced Entity Extractor initialized successfully")
        
        # Extract entities
        entities = extractor.extract_entities(test_text, schema)
        
        print(f"✅ Extraction completed: {len(entities)} entities found\n")
        
        # Display results grouped by type
        if entities:
            entity_groups = {}
            for entity in entities:
                entity_type = entity.get('entity_type', 'unknown')
                if entity_type not in entity_groups:
                    entity_groups[entity_type] = []
                entity_groups[entity_type].append(entity)
            
            for entity_type, type_entities in entity_groups.items():
                print(f"📋 {entity_type.upper()} ({len(type_entities)} entities)")
                print("-" * 40)
                
                for i, entity in enumerate(type_entities, 1):
                    name = entity['name']
                    confidence = entity.get('confidence', 'N/A')
                    extracted_by = entity.get('extracted_by', 'unknown')
                    
                    print(f"  {i}. '{name}' (confidence: {confidence}) - by {extracted_by}")
                    
                    if 'context' in entity:
                        context = entity['context']
                        if len(context) > 80:
                            context = context[:80] + "..."
                        print(f"     Context: \"{context}\"")
                    
                    if 'attributes' in entity and entity['attributes']:
                        print(f"     Attributes: {entity['attributes']}")
                    
                    print()
            
            # Strategy analysis
            print("📊 EXTRACTION STRATEGY ANALYSIS")
            print("-" * 30)
            
            strategy_counts = {}
            for entity in entities:
                strategy = entity.get('extracted_by', 'unknown')
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            total_entities = len(entities)
            for strategy, count in strategy_counts.items():
                percentage = (count / total_entities) * 100
                print(f"  {strategy}: {count} entities ({percentage:.1f}%)")
            
            # Test each strategy individually
            print(f"\n🔧 INDIVIDUAL STRATEGY TESTING")
            print("-" * 30)
            
            strategies = [
                ('_schema_guided_extraction', 'Schema-guided extraction'),
                ('_pattern_based_extraction', 'Pattern-based extraction'),
                ('_contextual_extraction', 'Contextual extraction'),
                ('_transformer_ner_extraction', 'Transformer NER'),
                ('_spacy_ner_extraction', 'spaCy NER')
            ]
            
            for method_name, strategy_name in strategies:
                if hasattr(extractor, method_name):
                    method = getattr(extractor, method_name)
                    try:
                        test_result = method(test_text, schema)
                        status = f"✅ Available ({len(test_result)} entities)"
                    except Exception as e:
                        status = f"❌ Error: {str(e)[:50]}..."
                else:
                    status = "❌ Method not found"
                
                print(f"  {strategy_name}: {status}")
        
        else:
            print("⚠️  No entities were extracted!")
        
        print(f"\n🎉 Test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Failed to import enhanced extractor: {e}")
        print("This may be due to the Phase 1/Phase 2 integration issue.")
        
        # Try manual extraction test
        print(f"\n🔧 MANUAL EXTRACTION TEST")
        print("-" * 30)
        manual_test_extraction(test_text)
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

def manual_test_extraction(text):
    """Manually test extraction strategies"""
    
    # Test transformers
    print("Testing Transformers NER...")
    try:
        from transformers import pipeline
        
        # Use a pre-trained NER model
        ner_pipeline = pipeline("ner", 
                               model="dbmdz/bert-large-cased-finetuned-conll03-english",
                               aggregation_strategy="simple")
        
        entities = ner_pipeline(text)
        print(f"  ✅ Transformers NER: {len(entities)} entities")
        
        for entity in entities[:3]:  # Show first 3
            print(f"    - {entity['word']} ({entity['entity_group']}, score: {entity['score']:.2f})")
        
    except Exception as e:
        print(f"  ❌ Transformers NER failed: {e}")
    
    # Test spaCy
    print("\nTesting spaCy NER...")
    try:
        import spacy
        
        # Try to load a model
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("  ⚠️  en_core_web_sm model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"  ✅ spaCy NER: {len(entities)} entities")
        
        for text, label in entities[:3]:  # Show first 3
            print(f"    - {text} ({label})")
            
    except Exception as e:
        print(f"  ❌ spaCy NER failed: {e}")

if __name__ == "__main__":
    test_with_all_dependencies()
