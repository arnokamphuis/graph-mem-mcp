"""
Quick Fix for Generic Relationship Problem

This module provides a simple relationship extractor that can replace
the generic "related_to" relationships with more meaningful ones.
"""

import re
from typing import List, Tuple, Dict

class QuickRelationshipExtractor:
    """Simple relationship extractor to fix generic 'related_to' dominance"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize relationship patterns for common scientific/technical domains"""
        return {
            # Scientific processes
            'causes': [
                r'(.+?)\s+causes?\s+(.+)',
                r'(.+?)\s+leads?\s+to\s+(.+)',
                r'(.+?)\s+results?\s+in\s+(.+)',
                r'(.+?)\s+triggers?\s+(.+)',
                r'(.+?)\s+induces?\s+(.+)',
            ],
            
            'creates': [
                r'(.+?)\s+creates?\s+(.+)',
                r'(.+?)\s+produces?\s+(.+)',
                r'(.+?)\s+generates?\s+(.+)',
                r'(.+?)\s+forms?\s+(.+)',
                r'(.+?)\s+makes?\s+(.+)',
            ],
            
            'transforms_into': [
                r'(.+?)\s+becomes?\s+(.+)',
                r'(.+?)\s+transforms?\s+into\s+(.+)',
                r'(.+?)\s+converts?\s+to\s+(.+)',
                r'(.+?)\s+evolves?\s+into\s+(.+)',
                r'(.+?)\s+changes?\s+to\s+(.+)',
            ],
            
            'flows_into': [
                r'(.+?)\s+flows?\s+into\s+(.+)',
                r'(.+?)\s+drains?\s+into\s+(.+)',
                r'(.+?)\s+empties?\s+into\s+(.+)',
                r'(.+?)\s+enters?\s+(.+)',
                r'(.+?)\s+joins?\s+(.+)',
            ],
            
            'contains': [
                r'(.+?)\s+contains?\s+(.+)',
                r'(.+?)\s+holds?\s+(.+)',
                r'(.+?)\s+includes?\s+(.+)',
                r'(.+?)\s+has\s+(.+)',
                r'(.+?)\s+stores?\s+(.+)',
            ],
            
            'enables': [
                r'(.+?)\s+enables?\s+(.+)',
                r'(.+?)\s+allows?\s+(.+)',
                r'(.+?)\s+permits?\s+(.+)',
                r'(.+?)\s+facilitates?\s+(.+)',
                r'(.+?)\s+supports?\s+(.+)',
            ],
            
            'utilizes': [
                r'(.+?)\s+uses?\s+(.+)',
                r'(.+?)\s+utilizes?\s+(.+)',
                r'(.+?)\s+employs?\s+(.+)',
                r'(.+?)\s+relies?\s+on\s+(.+)',
                r'(.+?)\s+depends?\s+on\s+(.+)',
            ],
            
            'implements': [
                r'(.+?)\s+implements?\s+(.+)',
                r'(.+?)\s+realizes?\s+(.+)',
                r'(.+?)\s+executes?\s+(.+)',
                r'(.+?)\s+applies?\s+(.+)',
            ],
            
            'threatens': [
                r'(.+?)\s+threatens?\s+(.+)',
                r'(.+?)\s+endangers?\s+(.+)',
                r'(.+?)\s+compromises?\s+(.+)',
                r'(.+?)\s+attacks?\s+(.+)',
            ],
            
            'demonstrates': [
                r'(.+?)\s+demonstrates?\s+(.+)',
                r'(.+?)\s+shows?\s+(.+)',
                r'(.+?)\s+exhibits?\s+(.+)',
                r'(.+?)\s+proves?\s+(.+)',
            ],
            
            'developed_by': [
                r'(.+?)\s+developed\s+by\s+(.+)',
                r'(.+?)\s+created\s+by\s+(.+)',
                r'(.+?)\s+invented\s+by\s+(.+)',
                r'(.+?)\s+designed\s+by\s+(.+)',
            ],
            
            'part_of': [
                r'(.+?)\s+is\s+part\s+of\s+(.+)',
                r'(.+?)\s+belongs\s+to\s+(.+)',
                r'(.+?)\s+component\s+of\s+(.+)',
                r'(.+?)\s+element\s+of\s+(.+)',
            ],
            
            'type_of': [
                r'(.+?)\s+is\s+a\s+type\s+of\s+(.+)',
                r'(.+?)\s+is\s+a\s+kind\s+of\s+(.+)',
                r'(.+?)\s+is\s+an?\s+example\s+of\s+(.+)',
                r'(.+?)\s+represents?\s+(.+)',
            ],
        }
    
    def extract_relationships(self, text: str, entities: List[str]) -> List[Tuple[str, str, str]]:
        """
        Extract relationships from text using pattern matching
        Returns list of (source, relation_type, target) tuples
        """
        relationships = []
        
        # Clean and normalize text
        text = text.lower().strip()
        
        # Try each relationship pattern
        for relation_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    source = match.group(1).strip()
                    target = match.group(2).strip()
                    
                    # Clean the entities
                    source = self._clean_entity(source)
                    target = self._clean_entity(target)
                    
                    # Only add if both entities are meaningful
                    if self._is_valid_entity(source) and self._is_valid_entity(target):
                        relationships.append((source, relation_type, target))
        
        # If no pattern relationships found, try simple co-occurrence with better types
        if not relationships:
            relationships.extend(self._extract_cooccurrence_relationships(text, entities))
        
        return relationships
    
    def _clean_entity(self, entity: str) -> str:
        """Clean and normalize entity names"""
        # Remove articles and common words
        entity = re.sub(r'\b(the|a|an|this|that|these|those)\b', '', entity, flags=re.IGNORECASE)
        entity = re.sub(r'\s+', ' ', entity).strip()
        
        # Remove punctuation at the end
        entity = re.sub(r'[.,;:!?]+$', '', entity)
        
        return entity
    
    def _is_valid_entity(self, entity: str) -> bool:
        """Check if entity is valid (not too short, not stop word)"""
        if len(entity) < 2:
            return False
        
        stop_words = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                     'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                     'and', 'or', 'but', 'so', 'for', 'nor', 'yet', 'to', 'of', 'in', 'on',
                     'at', 'by', 'with', 'from', 'up', 'about', 'into', 'through', 'during',
                     'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your'}
        
        return entity.lower() not in stop_words
    
    def _extract_cooccurrence_relationships(self, text: str, entities: List[str]) -> List[Tuple[str, str, str]]:
        """Extract relationships based on entity co-occurrence with better relationship types"""
        relationships = []
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if not sentence:
                continue
                
            # Find entities in this sentence
            entities_in_sentence = [e for e in entities if e.lower() in sentence]
            
            if len(entities_in_sentence) >= 2:
                # Determine relationship type based on sentence context
                relation_type = self._determine_contextual_relationship(sentence)
                
                # Create relationships between consecutive entities
                for i in range(len(entities_in_sentence) - 1):
                    source = entities_in_sentence[i]
                    target = entities_in_sentence[i + 1]
                    relationships.append((source, relation_type, target))
        
        return relationships
    
    def _determine_contextual_relationship(self, sentence: str) -> str:
        """Determine relationship type based on sentence context"""
        sentence = sentence.lower()
        
        # Check for different types of relationships based on keywords
        if any(word in sentence for word in ['cause', 'lead', 'result', 'trigger']):
            return 'causes'
        elif any(word in sentence for word in ['create', 'produce', 'generate', 'form']):
            return 'creates'
        elif any(word in sentence for word in ['flow', 'drain', 'empty', 'enter']):
            return 'flows_into'
        elif any(word in sentence for word in ['contain', 'hold', 'include', 'store']):
            return 'contains'
        elif any(word in sentence for word in ['use', 'utilize', 'employ', 'apply']):
            return 'utilizes'
        elif any(word in sentence for word in ['enable', 'allow', 'support', 'facilitate']):
            return 'enables'
        elif any(word in sentence for word in ['develop', 'create', 'invent', 'design']):
            return 'developed_by'
        elif any(word in sentence for word in ['part', 'component', 'element']):
            return 'part_of'
        elif any(word in sentence for word in ['type', 'kind', 'example', 'instance']):
            return 'type_of'
        else:
            return 'associated_with'  # Better than generic 'related_to'

# Quick test function
def test_extractor():
    """Test the relationship extractor"""
    extractor = QuickRelationshipExtractor()
    
    test_text = "Water flows into the ocean. The sun heats the water. Evaporation creates water vapor."
    entities = ["water", "ocean", "sun", "evaporation", "water vapor"]
    
    relationships = extractor.extract_relationships(test_text, entities)
    
    print("Test Results:")
    for source, relation, target in relationships:
        print(f"  {source} --[{relation}]--> {target}")
    
    return relationships

if __name__ == "__main__":
    test_extractor()
