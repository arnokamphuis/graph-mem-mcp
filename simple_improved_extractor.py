"""
Simplified Improved Relationship Extraction Demo

This demonstrates how to fix the "related_to" dominance problem without external dependencies.
"""

import re
from typing import Dict, List, Tuple

class SimpleImprovedRelationshipExtractor:
    """Simplified relationship extractor that fixes the 'related_to' dominance problem"""
    
    def __init__(self, domain: str = "water_cycle"):
        self.domain = domain
        self.relationship_patterns = self._get_enhanced_relationship_patterns()
        self.process_verbs = self._get_process_verb_patterns()
        self.causal_patterns = self._get_causal_patterns()
        
    def _get_enhanced_relationship_patterns(self) -> Dict[str, List[str]]:
        """Enhanced relationship patterns with broader matching"""
        if self.domain == "water_cycle":
            return {
                'evaporates_from': [
                    r'evaporate[s]?\s+from',
                    r'evaporation\s+from',
                    r'rising\s+from',
                    r'vapor\s+from',
                    r'heat[s]?\s+.*evaporat'
                ],
                'flows_into': [
                    r'flow[s]?\s+into',
                    r'drain[s]?\s+into',
                    r'empty\s+into',
                    r'join[s]?',
                    r'merge[s]?\s+into',
                    r'flow[s]?\s+to'
                ],
                'condenses_into': [
                    r'condense[s]?\s+into',
                    r'form[s]?\s+into',
                    r'become[s]?\s+clouds',
                    r'transform[s]?\s+into\s+droplets'
                ],
                'precipitates_as': [
                    r'fall[s]?\s+as',
                    r'precipitate[s]?\s+as',
                    r'become[s]?\s+rain',
                    r'become[s]?\s+snow'
                ],
                'powered_by': [
                    r'powered\s+by',
                    r'driven\s+by',
                    r'caused\s+by',
                    r'energy\s+from'
                ],
                'produces': [
                    r'produces?',
                    r'creates?',
                    r'generates?',
                    r'causes?'
                ]
            }
        return {}
    
    def _get_process_verb_patterns(self) -> Dict[str, str]:
        """Process-specific verb patterns"""
        return {
            'evaporate': 'evaporates_from',
            'evaporates': 'evaporates_from', 
            'evaporating': 'evaporates_from',
            'evaporation': 'evaporates_from',
            'condense': 'condenses_into',
            'condenses': 'condenses_into',
            'precipitation': 'precipitates_as',
            'heat': 'powered_by',
            'heats': 'powered_by'
        }
    
    def _get_causal_patterns(self) -> List[Tuple[str, str]]:
        """Causal relationship patterns"""
        return [
            (r'(.+?)\s+causes?\s+(.+)', 'causes'),
            (r'(.+?)\s+leads?\s+to\s+(.+)', 'leads_to'),
            (r'(.+?)\s+results?\s+in\s+(.+)', 'results_in'),
            (r'(.+?)\s+produces?\s+(.+)', 'produces')
        ]
    
    def extract_relationships_improved(self, text: str, concepts: List[str]) -> List[Dict]:
        """Extract relationships using multiple improved strategies"""
        relationships = []
        
        # Strategy 1: Enhanced pattern-based extraction
        relationships.extend(self._extract_pattern_based_relationships(text, concepts))
        
        # Strategy 2: Process verb extraction
        relationships.extend(self._extract_process_relationships(text, concepts))
        
        # Strategy 3: Causal relationship extraction
        relationships.extend(self._extract_causal_relationships(text, concepts))
        
        return relationships
    
    def _extract_pattern_based_relationships(self, text: str, concepts: List[str]) -> List[Dict]:
        """Extract relationships using enhanced pattern matching"""
        relationships = []
        
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
                
            # Find concepts in this sentence
            concepts_found = []
            for concept in concepts:
                if concept.lower() in sentence.lower():
                    concepts_found.append(concept)
            
            if len(concepts_found) >= 2:
                # Check for relationship patterns
                for rel_type, patterns in self.relationship_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, sentence, re.IGNORECASE):
                            # Create relationships between concept pairs
                            for i, concept1 in enumerate(concepts_found):
                                for concept2 in concepts_found[i+1:]:
                                    relationships.append({
                                        'source': concept1,
                                        'target': concept2,
                                        'relation_type': rel_type,
                                        'confidence': 0.8,
                                        'context': sentence.strip()
                                    })
        
        return relationships
    
    def _extract_process_relationships(self, text: str, concepts: List[str]) -> List[Dict]:
        """Extract relationships based on process verbs"""
        relationships = []
        
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
                
            # Find concepts in sentence
            concepts_in_sent = []
            for concept in concepts:
                if concept.lower() in sentence.lower():
                    concepts_in_sent.append(concept)
            
            if len(concepts_in_sent) >= 2:
                # Look for process verbs
                words = re.findall(r'\b\w+\b', sentence.lower())
                for word in words:
                    if word in self.process_verbs:
                        rel_type = self.process_verbs[word]
                        
                        # Create relationships between concepts
                        for i, concept1 in enumerate(concepts_in_sent):
                            for concept2 in concepts_in_sent[i+1:]:
                                relationships.append({
                                    'source': concept1,
                                    'target': concept2,
                                    'relation_type': rel_type,
                                    'confidence': 0.9,
                                    'context': sentence
                                })
        
        return relationships
    
    def _extract_causal_relationships(self, text: str, concepts: List[str]) -> List[Dict]:
        """Extract causal relationships using pattern matching"""
        relationships = []
        
        for pattern, rel_type in self.causal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                cause_text = match.group(1).strip()
                effect_text = match.group(2).strip()
                
                # Find concepts in cause and effect
                cause_concepts = [c for c in concepts if c.lower() in cause_text.lower()]
                effect_concepts = [c for c in concepts if c.lower() in effect_text.lower()]
                
                # Create relationships
                for cause_concept in cause_concepts:
                    for effect_concept in effect_concepts:
                        relationships.append({
                            'source': cause_concept,
                            'target': effect_concept,
                            'relation_type': rel_type,
                            'confidence': 0.85,
                            'context': match.group(0)
                        })
        
        return relationships

# Test the improved extractor
if __name__ == "__main__":
    extractor = SimpleImprovedRelationshipExtractor("water_cycle")
    
    # Test with water cycle sentences
    test_text = """Solar energy heats the ocean surface, causing water to evaporate into the atmosphere. 
    The water vapor then condenses into clouds, which eventually precipitate as rain. 
    Rain flows into rivers that drain into the ocean."""
    
    test_concepts = ["solar energy", "ocean", "water", "atmosphere", "water vapor", "clouds", "rain", "rivers"]
    
    relationships = extractor.extract_relationships_improved(test_text, test_concepts)
    
    print("=== IMPROVED RELATIONSHIP EXTRACTION RESULTS ===")
    print(f"Input text: {test_text[:100]}...")
    print(f"Concepts: {test_concepts}")
    print(f"\\nExtracted {len(relationships)} relationships:")
    
    # Group by relationship type
    by_type = {}
    for rel in relationships:
        rel_type = rel['relation_type']
        if rel_type not in by_type:
            by_type[rel_type] = []
        by_type[rel_type].append(rel)
    
    for rel_type, rels in by_type.items():
        print(f"\\n{rel_type.upper()}:")
        for rel in rels:
            print(f"  {rel['source']} --> {rel['target']} (conf: {rel['confidence']})")
            print(f"    Context: '{rel['context'][:80]}...'")
    
    print(f"\\n=== SUMMARY ===")
    print(f"Total relationships: {len(relationships)}")
    print(f"Relationship types found: {list(by_type.keys())}")
    print(f"'related_to' relationships: {len(by_type.get('related_to', []))}")
    print(f"Specific relationships: {len(relationships) - len(by_type.get('related_to', []))}")
