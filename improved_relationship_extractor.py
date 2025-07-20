"""
Domain-Agnostic Relationship Extraction for Enhanced Knowledge Graph Construction

This module extracts meaningful relationships from text using pure linguistic analysis,
without any hardcoded domain-specific patterns. Works for any content domain.
"""

import re
from typing import Dict, List, Tuple
import spacy
from collections import defaultdict

class UniversalRelationshipExtractor:
    """Domain-agnostic relationship extractor using pure linguistic analysis"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        # Universal linguistic patterns (no domain-specific content)
        self.universal_patterns = self._get_universal_linguistic_patterns()
        
        # Universal causal patterns
        self.causal_patterns = self._get_universal_causal_patterns()
        
    def _get_universal_linguistic_patterns(self) -> Dict[str, List[str]]:
        """Universal relationship patterns based on linguistic structure only"""
        return {
            # Movement and flow patterns
            'flows_into': [r'\w+\s+flow[s]?\s+into\s+\w+', r'\w+\s+drain[s]?\s+into\s+\w+', r'\w+\s+empty\s+into\s+\w+'],
            'moves_to': [r'\w+\s+move[s]?\s+to\s+\w+', r'\w+\s+travel[s]?\s+to\s+\w+', r'\w+\s+go[es]?\s+to\s+\w+'],
            
            # Transformation patterns
            'transforms_into': [r'\w+\s+transform[s]?\s+into\s+\w+', r'\w+\s+become[s]?\s+\w+', r'\w+\s+turn[s]?\s+into\s+\w+'],
            'converts_to': [r'\w+\s+convert[s]?\s+to\s+\w+', r'\w+\s+change[s]?\s+to\s+\w+'],
            
            # Creation patterns
            'creates': [r'\w+\s+create[s]?\s+\w+', r'\w+\s+produce[s]?\s+\w+', r'\w+\s+generate[s]?\s+\w+'],
            'forms': [r'\w+\s+form[s]?\s+\w+', r'\w+\s+make[s]?\s+\w+'],
            
            # Containment patterns
            'contains': [r'\w+\s+contain[s]?\s+\w+', r'\w+\s+hold[s]?\s+\w+', r'\w+\s+store[s]?\s+\w+'],
            'located_in': [r'\w+\s+in\s+\w+', r'\w+\s+within\s+\w+', r'\w+\s+inside\s+\w+'],
            
            # Influence patterns
            'affects': [r'\w+\s+affect[s]?\s+\w+', r'\w+\s+influence[s]?\s+\w+', r'\w+\s+impact[s]?\s+\w+'],
            'controls': [r'\w+\s+control[s]?\s+\w+', r'\w+\s+regulate[s]?\s+\w+', r'\w+\s+manage[s]?\s+\w+'],
            
            # Support patterns  
            'enables': [r'\w+\s+enable[s]?\s+\w+', r'\w+\s+allow[s]?\s+\w+', r'\w+\s+support[s]?\s+\w+'],
            'helps': [r'\w+\s+help[s]?\s+\w+', r'\w+\s+assist[s]?\s+\w+'],
            
            # Possession patterns
            'has': [r'\w+\s+ha[s|ve]\s+\w+', r'\w+\s+own[s]?\s+\w+', r'\w+\s+possess[es]?\s+\w+'],
            
            # Temporal patterns
            'follows': [r'\w+\s+follow[s]?\s+\w+', r'\w+\s+come[s]?\s+after\s+\w+'],
            'precedes': [r'\w+\s+precede[s]?\s+\w+', r'\w+\s+come[s]?\s+before\s+\w+']
        }
    
    def _get_universal_causal_patterns(self) -> List[Tuple[str, str]]:
        """Universal causal relationship patterns"""
        return [
            (r'(.+?)\s+cause[s]?\s+(.+)', 'causes'),
            (r'(.+?)\s+lead[s]?\s+to\s+(.+)', 'leads_to'),
            (r'(.+?)\s+result[s]?\s+in\s+(.+)', 'results_in'),
            (r'(.+?)\s+trigger[s]?\s+(.+)', 'triggers'),
            (r'due\s+to\s+(.+?),?\s+(.+)', 'caused_by'),
            (r'because\s+of\s+(.+?),?\s+(.+)', 'caused_by'),
            (r'when\s+(.+?),?\s+(.+)', 'when_then'),
            (r'if\s+(.+?),?\s+then\s+(.+)', 'if_then')
        ]
    
    
    def extract_relationships_universal(self, text: str, concepts: List[str]) -> List[Dict]:
        """
        Extract relationships using universal linguistic analysis
        
        Args:
            text: Input text
            concepts: List of concept names to look for
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # Strategy 1: Universal pattern-based extraction
        relationships.extend(self._extract_universal_patterns(text, concepts))
        
        # Strategy 2: Dependency parsing for semantic relationships
        relationships.extend(self._extract_dependency_relationships(text, concepts))
        
        # Strategy 3: Universal causal relationship extraction
        relationships.extend(self._extract_universal_causal_relationships(text, concepts))
        
        # Strategy 4: Verb-based relationship inference
        relationships.extend(self._extract_verb_relationships(text, concepts))
        
        return relationships
    
    def _extract_universal_patterns(self, text: str, concepts: List[str]) -> List[Dict]:
        """Extract relationships using universal linguistic patterns"""
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
                # Check for universal relationship patterns
                for rel_type, patterns in self.universal_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, sentence, re.IGNORECASE):
                            # Extract specific entities from the pattern match
                            relationships.extend(self._extract_entities_from_pattern(
                                sentence, pattern, concepts_found, rel_type))
        
        return relationships
    
    def _extract_entities_from_pattern(self, sentence: str, pattern: str, concepts: List[str], rel_type: str) -> List[Dict]:
        """Extract specific entity relationships from pattern matches"""
        relationships = []
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            # For each concept pair in the sentence, create a relationship
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    relationships.append({
                        'source': concept1,
                        'target': concept2,
                        'relation_type': rel_type,
                        'confidence': 0.8,
                        'context': sentence.strip()
                    })
        return relationships
    
    def _extract_dependency_relationships(self, text: str, concepts: List[str]) -> List[Dict]:
        """Extract relationships using spaCy dependency parsing"""
        relationships = []
        doc = self.nlp(text)
        
        for sent in doc.sents:
            concepts_in_sent = []
            concept_tokens = {}
            
            # Find concept tokens in sentence
            for concept in concepts:
                for token in sent:
                    if concept.lower() in token.text.lower() or token.text.lower() in concept.lower():
                        concepts_in_sent.append(concept)
                        concept_tokens[concept] = token
                        break
            
            if len(concepts_in_sent) >= 2:
                # Analyze dependency relationships between concept tokens
                for concept1 in concepts_in_sent:
                    for concept2 in concepts_in_sent:
                        if concept1 != concept2:
                            token1 = concept_tokens.get(concept1)
                            token2 = concept_tokens.get(concept2)
                            
                            if token1 and token2:
                                rel_type = self._classify_dependency_relationship(token1, token2, sent)
                                if rel_type and rel_type != 'related_to':
                                    relationships.append({
                                        'source': concept1,
                                        'target': concept2,
                                        'relation_type': rel_type,
                                        'confidence': 0.7,
                                        'context': sent.text
                                    })
        
        return relationships
    
    def _classify_dependency_relationship(self, token1, token2, sent) -> str:
        """Classify relationship based on dependency structure"""
        # Find connecting verb or preposition
        for token in sent:
            if token.pos_ == 'VERB':
                # Check if this verb connects our entities
                if (token1 in token.subtree or token2 in token.subtree):
                    verb_lemma = token.lemma_.lower()
                    
                    # Use verb semantic meaning to determine relationship
                    if verb_lemma in ['flow', 'drain', 'empty', 'pour']:
                        return 'flows_into'
                    elif verb_lemma in ['transform', 'change', 'become', 'turn']:
                        return 'transforms_into'
                    elif verb_lemma in ['create', 'produce', 'generate', 'make']:
                        return 'creates'
                    elif verb_lemma in ['cause', 'trigger', 'induce']:
                        return 'causes'
                    elif verb_lemma in ['contain', 'hold', 'store']:
                        return 'contains'
                    elif verb_lemma in ['affect', 'influence', 'impact']:
                        return 'affects'
                    elif verb_lemma in ['control', 'regulate', 'manage']:
                        return 'controls'
                    elif verb_lemma in ['enable', 'allow', 'support']:
                        return 'enables'
                    elif verb_lemma in ['move', 'transport', 'carry']:
                        return 'transports'
                    else:
                        # Create relationship type from verb
                        return f'{verb_lemma}s'
        
        return 'related_to'
    
    def _extract_universal_causal_relationships(self, text: str, concepts: List[str]) -> List[Dict]:
        """Extract causal relationships using universal patterns"""
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
    
    def _extract_verb_relationships(self, text: str, concepts: List[str]) -> List[Dict]:
        """Extract relationships by analyzing all verbs in context"""
        relationships = []
        doc = self.nlp(text)
        
        for sent in doc.sents:
            concepts_in_sent = [c for c in concepts if c.lower() in sent.text.lower()]
            
            if len(concepts_in_sent) >= 2:
                # Find all verbs in sentence
                verbs = [token for token in sent if token.pos_ == 'VERB']
                
                for verb in verbs:
                    # Create relationship type dynamically from verb
                    verb_lemma = verb.lemma_.lower()
                    
                    # Skip common auxiliary verbs
                    if verb_lemma in ['be', 'have', 'do', 'will', 'would', 'could', 'should']:
                        continue
                    
                    rel_type = f'{verb_lemma}s'
                    
                    # Create relationships between concepts
                    for i, concept1 in enumerate(concepts_in_sent):
                        for concept2 in concepts_in_sent[i+1:]:
                            relationships.append({
                                'source': concept1,
                                'target': concept2,
                                'relation_type': rel_type,
                                'confidence': 0.6,
                                'context': sent.text
                            })
        
        return relationships

# Test the universal extractor
if __name__ == "__main__":
    extractor = UniversalRelationshipExtractor()
    
    # Test with various domain content to prove universality
    test_texts = [
        "Solar energy heats the ocean surface, causing water to evaporate into the atmosphere. The water vapor then condenses into clouds, which eventually precipitate as rain.",
        "The CEO creates a new strategy that transforms the company culture. This innovation leads to increased productivity.",
        "Mitochondria produce ATP that powers cellular processes. The electron transport chain generates energy for the cell."
    ]
    
    test_concepts_sets = [
        ["solar energy", "ocean", "water", "atmosphere", "water vapor", "clouds", "rain"],
        ["CEO", "strategy", "company culture", "innovation", "productivity"],
        ["mitochondria", "ATP", "cellular processes", "electron transport chain", "energy", "cell"]
    ]
    
    for i, (test_text, test_concepts) in enumerate(zip(test_texts, test_concepts_sets)):
        print(f"\n=== Test {i+1}: Universal Relationship Extraction ===")
        print(f"Text: {test_text}")
        
        relationships = extractor.extract_relationships_universal(test_text, test_concepts)
        
        print(f"Extracted {len(relationships)} relationships:")
        for rel in relationships:
            print(f"  {rel['source']} --[{rel['relation_type']}]--> {rel['target']} (confidence: {rel['confidence']})")
