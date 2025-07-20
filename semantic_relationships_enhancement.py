"""
Enhanced Semantic Relationship Extraction

This module provides advanced relationship extraction that captures meaningful 
semantic relationships instead of generic "related_to" connections.
"""

import re
from typing import List, Dict, Any, Tuple


def extract_semantic_relationships(text: str, entities: dict) -> List[Dict[str, Any]]:
    """
    Extract semantically meaningful relationships between entities.
    
    Args:
        text: The source text to analyze
        entities: Dictionary of extracted entities with their metadata
        
    Returns:
        List of relationship dictionaries with semantic types
    """
    relationships = []
    
    # Split text into sentences for analysis
    sentences = re.split(r'[.!?;]', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
            
        # Find entities in this sentence
        sentence_entities = []
        entity_positions = {}
        
        for entity in entities.keys():
            if entity.lower() in sentence.lower():
                sentence_entities.append(entity)
                entity_positions[entity] = sentence.lower().find(entity.lower())
        
        # Sort entities by position in sentence for better relationship extraction
        sentence_entities.sort(key=lambda e: entity_positions[e])
        
        # Extract relationships between entity pairs
        for i, entity1 in enumerate(sentence_entities):
            for entity2 in sentence_entities[i+1:]:
                relationship = analyze_entity_relationship(
                    sentence, entity1, entity2, entities
                )
                if relationship:
                    relationships.append(relationship)
    
    return relationships


def analyze_entity_relationship(sentence: str, entity1: str, entity2: str, entities: dict) -> Dict[str, Any]:
    """
    Analyze the semantic relationship between two entities in a sentence.
    
    Args:
        sentence: The sentence containing both entities
        entity1: First entity
        entity2: Second entity
        entities: Entity metadata dictionary
        
    Returns:
        Relationship dictionary or None if no meaningful relationship found
    """
    sentence_lower = sentence.lower()
    entity1_lower = entity1.lower()
    entity2_lower = entity2.lower()
    
    # Get entity types for context
    entity1_type = entities.get(entity1, {}).get("type", "unknown")
    entity2_type = entities.get(entity2, {}).get("type", "unknown")
    
    # Find the text between entities
    pos1 = sentence_lower.find(entity1_lower)
    pos2 = sentence_lower.find(entity2_lower)
    
    if pos1 == -1 or pos2 == -1:
        return None
    
    # Ensure entity1 comes before entity2
    if pos1 > pos2:
        entity1, entity2 = entity2, entity1
        entity1_lower, entity2_lower = entity2_lower, entity1_lower
        entity1_type, entity2_type = entity2_type, entity1_type
        pos1, pos2 = pos2, pos1
    
    # Extract the connecting text between entities
    start = pos1 + len(entity1_lower)
    end = pos2
    connecting_text = sentence_lower[start:end].strip()
    
    # Pattern-based relationship extraction
    relationship_type = extract_pattern_based_relationship(
        sentence_lower, entity1_lower, entity2_lower, connecting_text
    )
    
    # If no pattern match, try contextual analysis
    if relationship_type == "related_to":
        relationship_type = extract_contextual_relationship(
            connecting_text, entity1_type, entity2_type
        )
    
    # If still generic, try domain-specific inference
    if relationship_type == "related_to":
        relationship_type = infer_domain_relationship(entity1_type, entity2_type)
    
    return {
        "from": entity1,
        "to": entity2,
        "type": relationship_type,
        "context": sentence[:200] + "..." if len(sentence) > 200 else sentence,
        "confidence": calculate_confidence(relationship_type, connecting_text),
        "connecting_text": connecting_text
    }


def extract_pattern_based_relationship(sentence: str, entity1: str, entity2: str, connecting_text: str) -> str:
    """Extract relationships using predefined linguistic patterns."""
    
    # Hierarchical/Classification patterns
    is_patterns = [
        r'\bis\s+an?\s+',
        r'\bare\s+',
        r'\bis\s+a\s+type\s+of\s+',
        r'\bserves\s+as\s+',
        r'\bacts\s+as\s+'
    ]
    
    for pattern in is_patterns:
        if re.search(f'{re.escape(entity1)}\s*{pattern}.*{re.escape(entity2)}', sentence):
            return "is_type_of"
        if re.search(f'{re.escape(entity2)}\s*{pattern}.*{re.escape(entity1)}', sentence):
            return "instance_of"
    
    # Possession/Containment patterns
    has_patterns = [
        r'\bhas\s+',
        r'\bhave\s+',
        r'\bcontains\s+',
        r'\bincludes\s+',
        r'\bpossesses\s+',
        r'\bowns\s+'
    ]
    
    for pattern in has_patterns:
        if re.search(f'{re.escape(entity1)}\s*{pattern}.*{re.escape(entity2)}', sentence):
            return "has"
        if re.search(f'{re.escape(entity2)}\s*{pattern}.*{re.escape(entity1)}', sentence):
            return "belongs_to"
    
    # Creation/Development patterns
    creation_patterns = [
        r'\bcreated\s+',
        r'\bdeveloped\s+',
        r'\bbuilt\s+',
        r'\bdesigned\s+',
        r'\bauthored\s+',
        r'\bwrote\s+',
        r'\bmade\s+',
        r'\bfounded\s+'
    ]
    
    for pattern in creation_patterns:
        if re.search(f'{re.escape(entity1)}\s*{pattern}.*{re.escape(entity2)}', sentence):
            return "created"
        if re.search(f'{re.escape(entity2)}\s*{pattern}.*{re.escape(entity1)}', sentence):
            return "created_by"
    
    # Usage/Dependency patterns
    usage_patterns = [
        r'\buses\s+',
        r'\butilizes\s+',
        r'\bemploys\s+',
        r'\brelies\s+on\s+',
        r'\bdepends\s+on\s+',
        r'\brequires\s+'
    ]
    
    for pattern in usage_patterns:
        if re.search(f'{re.escape(entity1)}\s*{pattern}.*{re.escape(entity2)}', sentence):
            return "uses"
        if re.search(f'{re.escape(entity2)}\s*{pattern}.*{re.escape(entity1)}', sentence):
            return "used_by"
    
    # Implementation/Extension patterns
    impl_patterns = [
        r'\bimplements\s+',
        r'\bextends\s+',
        r'\binherits\s+from\s+',
        r'\bderives\s+from\s+',
        r'\bis\s+based\s+on\s+'
    ]
    
    for pattern in impl_patterns:
        if re.search(f'{re.escape(entity1)}\s*{pattern}.*{re.escape(entity2)}', sentence):
            return "implements"
        if re.search(f'{re.escape(entity2)}\s*{pattern}.*{re.escape(entity1)}', sentence):
            return "implemented_by"
    
    # Location patterns
    location_patterns = [
        r'\bin\s+',
        r'\bat\s+',
        r'\bwithin\s+',
        r'\blocated\s+in\s+',
        r'\bsituated\s+in\s+'
    ]
    
    for pattern in location_patterns:
        if re.search(f'{re.escape(entity1)}\s*{pattern}.*{re.escape(entity2)}', sentence):
            return "located_in"
        if re.search(f'{re.escape(entity2)}\s*{pattern}.*{re.escape(entity1)}', sentence):
            return "contains"
    
    # Temporal patterns
    temporal_patterns = [
        r'\bbefore\s+',
        r'\bafter\s+',
        r'\bduring\s+',
        r'\bfollowed\s+by\s+',
        r'\bpreceded\s+by\s+'
    ]
    
    for pattern in temporal_patterns:
        if re.search(f'{re.escape(entity1)}\s*{pattern}.*{re.escape(entity2)}', sentence):
            if 'before' in pattern:
                return "before"
            elif 'after' in pattern or 'followed' in pattern:
                return "after"
            elif 'during' in pattern:
                return "during"
    
    return "related_to"  # Fallback


def extract_contextual_relationship(connecting_text: str, entity1_type: str, entity2_type: str) -> str:
    """Extract relationships based on contextual analysis of connecting words."""
    
    # Action verbs that indicate specific relationships
    action_verbs = {
        'manages': 'manages',
        'leads': 'leads',
        'supervises': 'supervises',
        'reports to': 'reports_to',
        'works for': 'works_for',
        'collaborates with': 'collaborates_with',
        'competes with': 'competes_with',
        'supports': 'supports',
        'enhances': 'enhances',
        'improves': 'improves',
        'modifies': 'modifies',
        'configures': 'configures',
        'processes': 'processes',
        'generates': 'generates',
        'produces': 'produces',
        'consumes': 'consumes',
        'transforms': 'transforms',
        'triggers': 'triggers',
        'initiates': 'initiates',
        'terminates': 'terminates',
        'controls': 'controls',
        'monitors': 'monitors',
        'validates': 'validates',
        'tests': 'tests',
        'debugs': 'debugs',
        'fixes': 'fixes'
    }
    
    for verb, relationship in action_verbs.items():
        if verb in connecting_text:
            return relationship
    
    # Preposition-based relationships
    preposition_relationships = {
        'with': 'associated_with',
        'by': 'performed_by',
        'for': 'intended_for',
        'from': 'derived_from',
        'to': 'directed_to',
        'through': 'through',
        'via': 'via',
        'under': 'under',
        'over': 'over',
        'within': 'within',
        'without': 'without',
        'between': 'between',
        'among': 'among'
    }
    
    for prep, relationship in preposition_relationships.items():
        if f' {prep} ' in connecting_text:
            return relationship
    
    return "related_to"


def infer_domain_relationship(entity1_type: str, entity2_type: str) -> str:
    """Infer relationships based on entity types and domain knowledge."""
    
    # Person-Organization relationships
    if entity1_type in ['person', 'named_entity'] and entity2_type in ['organization', 'company']:
        return "works_for"
    
    if entity1_type in ['organization', 'company'] and entity2_type in ['person', 'named_entity']:
        return "employs"
    
    # Technology relationships
    if entity1_type == 'technical_term' and entity2_type == 'technical_term':
        return "depends_on"
    
    # Concept relationships
    if entity1_type == 'concept' and entity2_type == 'concept':
        return "relates_to"
    
    # Date-Entity relationships
    if entity1_type == 'date' or entity2_type == 'date':
        return "occurred_on"
    
    # Location relationships
    if entity1_type in ['location', 'place'] or entity2_type in ['location', 'place']:
        return "located_in"
    
    # URL/Email relationships
    if entity1_type in ['url', 'email'] or entity2_type in ['url', 'email']:
        return "references"
    
    return "related_to"


def calculate_confidence(relationship_type: str, connecting_text: str) -> float:
    """Calculate confidence score for the extracted relationship."""
    
    if relationship_type == "related_to":
        return 0.3  # Low confidence for generic relationships
    
    # Higher confidence for specific patterns
    specific_patterns = ['is_type_of', 'created_by', 'implements', 'has', 'uses']
    if relationship_type in specific_patterns:
        return 0.9
    
    # Medium confidence for contextual relationships
    contextual_patterns = ['works_for', 'manages', 'supports', 'located_in']
    if relationship_type in contextual_patterns:
        return 0.7
    
    # Default confidence for inferred relationships
    return 0.5


def get_relationship_taxonomy() -> Dict[str, List[str]]:
    """Return a taxonomy of all supported relationship types."""
    
    return {
        "hierarchical": [
            "is_type_of", "instance_of", "subclass_of", "parent_of", "child_of"
        ],
        "possession": [
            "has", "contains", "includes", "part_of", "belongs_to", "owns"
        ],
        "creation": [
            "created", "created_by", "authored_by", "developed_by", "built_by"
        ],
        "usage": [
            "uses", "used_by", "depends_on", "requires", "utilizes"
        ],
        "implementation": [
            "implements", "implemented_by", "extends", "inherits_from"
        ],
        "organizational": [
            "works_for", "employs", "manages", "reports_to", "collaborates_with"
        ],
        "spatial": [
            "located_in", "contains", "within", "under", "over", "between"
        ],
        "temporal": [
            "before", "after", "during", "followed_by", "preceded_by"
        ],
        "functional": [
            "processes", "generates", "transforms", "controls", "monitors"
        ],
        "associative": [
            "associated_with", "relates_to", "connected_to", "similar_to"
        ]
    }
