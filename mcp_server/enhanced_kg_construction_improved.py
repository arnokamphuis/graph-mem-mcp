"""
Enhanced Knowledge Graph Construction Module with Domain-Specific Relationships

This module provides advanced text-to-knowledge-graph construction with:
- spaCy for Named Entity Recognition and dependency parsing
- sentence-transformers for semantic similarity and concept clustering
- Domain-specific relationship extraction for better semantic understanding
- Water cycle and scientific process relationship patterns
"""

import spacy
from sentence_transformers import SentenceTransformer
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
import re
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConceptNode:
    """Represents a concept in the knowledge graph"""
    id: str
    name: str
    type: str  # PERSON, ORG, CONCEPT, etc.
    aliases: List[str]
    confidence: float
    embedding: Optional[np.ndarray] = None
    observations: List[str] = None
    
    def __post_init__(self):
        if self.observations is None:
            self.observations = []

@dataclass
class SemanticRelationship:
    """Represents a semantic relationship between concepts"""
    source: str
    target: str
    relation_type: str
    confidence: float
    context: str
    dependency_path: str = ""

class EnhancedKGConstructor:
    """Advanced Knowledge Graph Constructor with Domain-Specific Relationships"""
    
    def __init__(self, model_name: str = "en_core_web_sm", 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the constructor
        
        Args:
            model_name: spaCy model name
            embedding_model: SentenceTransformer model name
        """
        self.nlp = spacy.load(model_name)
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Basic relationship patterns for text matching
        self.relation_patterns = {
            'is_type_of': ['is', 'are', 'was', 'were', 'being', 'been', 'represents', 'constitutes'],
            'has_property': ['has', 'have', 'had', 'owns', 'contains', 'includes', 'exhibits', 'displays'],
            'located_in': ['in', 'at', 'located', 'based', 'situated', 'found in', 'occurs in'],
            'part_of': ['part of', 'member of', 'belongs to', 'component of', 'element of'],
            'created_by': ['created by', 'made by', 'developed by', 'authored by', 'produced by'],
            'influences': ['influences', 'affects', 'impacts', 'modifies', 'alters', 'changes'],
            'enables': ['enables', 'allows', 'permits', 'facilitates', 'supports']
        }
    
    def extract_concepts(self, text: str) -> List[ConceptNode]:
        """
        Extract concepts from text using advanced NLP techniques
        
        Args:
            text: Input text to process
            
        Returns:
            List of ConceptNode objects representing extracted concepts
        """
        doc = self.nlp(text)
        concepts = {}
        
        # Extract named entities using spaCy NER
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
                concept_id = self._normalize_concept_name(ent.text)
                if concept_id not in concepts:
                    concepts[concept_id] = ConceptNode(
                        id=concept_id,
                        name=ent.text,
                        type=ent.label_,
                        aliases=[ent.text],
                        confidence=0.9
                    )
                else:
                    # Add as alias if not already present
                    if ent.text not in concepts[concept_id].aliases:
                        concepts[concept_id].aliases.append(ent.text)
        
        # Extract key noun phrases as concepts
        for chunk in doc.noun_chunks:
            # Filter out simple pronouns and common words
            if (len(chunk.text.split()) > 1 and 
                chunk.root.pos_ == 'NOUN' and 
                not chunk.root.is_stop and
                len(chunk.text.strip()) > 3):
                
                concept_id = self._normalize_concept_name(chunk.text)
                if concept_id not in concepts:
                    concepts[concept_id] = ConceptNode(
                        id=concept_id,
                        name=chunk.text,
                        type='CONCEPT',
                        aliases=[chunk.text],
                        confidence=0.7
                    )
                else:
                    # Add as alias if not already present
                    if chunk.text not in concepts[concept_id].aliases:
                        concepts[concept_id].aliases.append(chunk.text)
        
        # Extract important single nouns
        for token in doc:
            if (token.pos_ == 'NOUN' and 
                not token.is_stop and 
                not token.is_punct and
                len(token.text) > 3 and
                token.text.lower() not in ['thing', 'way', 'part', 'time', 'place']):
                
                concept_id = self._normalize_concept_name(token.text)
                if concept_id not in concepts:
                    concepts[concept_id] = ConceptNode(
                        id=concept_id,
                        name=token.text,
                        type='CONCEPT',
                        aliases=[token.text],
                        confidence=0.5
                    )
        
        # Merge similar concepts using embeddings
        concept_list = list(concepts.values())
        merged_concepts = self._merge_similar_concepts(concept_list)
        
        return merged_concepts
    
    def _merge_similar_concepts(self, concepts: List[ConceptNode], 
                               similarity_threshold: float = 0.8) -> List[ConceptNode]:
        """Merge concepts that are semantically similar"""
        if len(concepts) <= 1:
            return concepts
        
        # Generate embeddings for concept names
        concept_texts = [concept.name for concept in concepts]
        embeddings = self.embedding_model.encode(concept_texts)
        
        # Store embeddings in concept objects
        for concept, embedding in zip(concepts, embeddings):
            concept.embedding = embedding
        
        # Cluster similar concepts
        clustering = DBSCAN(eps=1-similarity_threshold, min_samples=1, 
                           metric='cosine').fit(embeddings)
        
        # Merge concepts in the same cluster
        clusters = defaultdict(list)
        for concept, label in zip(concepts, clustering.labels_):
            clusters[label].append(concept)
        
        merged_concepts = []
        for cluster_concepts in clusters.values():
            if len(cluster_concepts) == 1:
                merged_concepts.append(cluster_concepts[0])
            else:
                # Merge concepts in the cluster
                primary_concept = max(cluster_concepts, key=lambda c: c.confidence)
                for other_concept in cluster_concepts:
                    if other_concept != primary_concept:
                        # Add aliases
                        for alias in other_concept.aliases:
                            if alias not in primary_concept.aliases:
                                primary_concept.aliases.append(alias)
                        # Merge observations
                        primary_concept.observations.extend(other_concept.observations)
                
                merged_concepts.append(primary_concept)
        
        return merged_concepts
    
    def extract_semantic_relationships(self, text: str, concepts: List[ConceptNode]) -> List[SemanticRelationship]:
        """
        Extract semantic relationships between concepts using dependency parsing
        
        Args:
            text: Input text
            concepts: List of concepts to find relationships between
            
        Returns:
            List of SemanticRelationship objects
        """
        doc = self.nlp(text)
        relationships = []
        
        # Create a mapping from text spans to concepts
        concept_map = {}
        for concept in concepts:
            for alias in concept.aliases:
                concept_map[alias.lower()] = concept.id
        
        # Extract relationships using dependency parsing
        for sent in doc.sents:
            sent_relationships = self._extract_relationships_from_sentence(sent, concept_map)
            relationships.extend(sent_relationships)
        
        # Extract pattern-based relationships
        pattern_relationships = self._extract_pattern_relationships(text, concepts)
        relationships.extend(pattern_relationships)
        
        return relationships
    
    def _extract_relationships_from_sentence(self, sent, concept_map: Dict[str, str]) -> List[SemanticRelationship]:
        """Extract relationships from a single sentence using dependency parsing"""
        relationships = []
        
        # Find entities in the sentence
        entities_in_sent = []
        for token in sent:
            if token.text.lower() in concept_map:
                entities_in_sent.append((token, concept_map[token.text.lower()]))
        
        # Extract relationships based on dependency paths
        for i, (entity1_token, entity1_id) in enumerate(entities_in_sent):
            for j, (entity2_token, entity2_id) in enumerate(entities_in_sent):
                if i >= j:  # Avoid duplicates and self-relationships
                    continue
                
                # Find dependency path between entities
                path = self._find_dependency_path(entity1_token, entity2_token)
                if path:
                    relation_type = self._classify_relationship_from_path(path)
                    if relation_type:
                        relationships.append(SemanticRelationship(
                            source=entity1_id,
                            target=entity2_id,
                            relation_type=relation_type,
                            confidence=0.7,
                            context=sent.text,
                            dependency_path=' -> '.join([token.text for token in path])
                        ))
        
        return relationships
    
    def _extract_pattern_relationships(self, text: str, concepts: List[ConceptNode]) -> List[SemanticRelationship]:
        """Extract relationships using pattern matching"""
        relationships = []
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            # Find concepts in this sentence
            concepts_in_sent = []
            for concept in concepts:
                for alias in concept.aliases:
                    if alias.lower() in sentence.lower():
                        concepts_in_sent.append(concept)
                        break
            
            if len(concepts_in_sent) < 2:
                continue
            
            # Check for relationship patterns
            for relation_type, patterns in self.relation_patterns.items():
                for pattern in patterns:
                    if pattern in sentence.lower():
                        # Find concepts around the pattern
                        for i, concept1 in enumerate(concepts_in_sent):
                            for concept2 in concepts_in_sent[i+1:]:
                                relationships.append(SemanticRelationship(
                                    source=concept1.id,
                                    target=concept2.id,
                                    relation_type=relation_type,
                                    confidence=0.6,
                                    context=sentence
                                ))
        
        return relationships
    
    def _find_dependency_path(self, token1, token2):
        """Find the dependency path between two tokens"""
        # Simple implementation - find common ancestor
        ancestors1 = set()
        current = token1
        while current.head != current:
            ancestors1.add(current)
            current = current.head
        ancestors1.add(current)
        
        current = token2
        path = [token2]
        while current.head != current and current not in ancestors1:
            current = current.head
            path.append(current)
        
        if current in ancestors1:
            return path
        return None
    
    def _classify_relationship_from_path(self, path):
        """Classify relationship type based on dependency path using semantic analysis"""
        if not path:
            return 'related_to'
        
        # Find verbs and prepositions in the dependency path
        verbs_in_path = [token for token in path if token.pos_ == 'VERB']
        prepositions = [token for token in path if token.pos_ == 'ADP']
        
        if verbs_in_path:
            verb = verbs_in_path[0]
            verb_lemma = verb.lemma_.lower()
            
            # Extract semantic relationship based on verb meaning and context
            relationship_type = self._extract_semantic_relationship(verb, prepositions, path)
            return relationship_type
        
        # If no verb, check for prepositions that indicate relationships
        if prepositions:
            prep = prepositions[0].text.lower()
            if prep in ['in', 'within', 'inside']:
                return 'located_in'
            elif prep in ['from', 'out_of']:
                return 'originates_from'
            elif prep in ['to', 'into', 'toward']:
                return 'leads_to'
            elif prep in ['through', 'via']:
                return 'passes_through'
            elif prep in ['with', 'alongside']:
                return 'associated_with'
            elif prep in ['by', 'using']:
                return 'performed_by'
        
        return 'related_to'
    
    def _extract_semantic_relationship(self, verb, prepositions, path):
        """Extract semantic relationship type based on verb and context"""
        verb_lemma = verb.lemma_.lower()
        verb_text = verb.text.lower()
        
        # Analyze the verb's semantic meaning to determine relationship type
        
        # Movement and flow relationships
        if verb_lemma in ['flow', 'move', 'travel', 'go', 'pass', 'run', 'drain', 'empty']:
            if any(p.text.lower() in ['into', 'to'] for p in prepositions):
                return 'flows_into'
            elif any(p.text.lower() in ['from', 'out'] for p in prepositions):
                return 'flows_from'
            else:
                return 'moves_through'
        
        # Transformation relationships
        elif verb_lemma in ['transform', 'change', 'convert', 'become', 'turn', 'evolve']:
            return 'transforms_into'
        
        # Creation and production relationships
        elif verb_lemma in ['create', 'produce', 'generate', 'make', 'form', 'build', 'develop']:
            return 'creates'
        
        # Causation relationships
        elif verb_lemma in ['cause', 'lead', 'result', 'trigger', 'induce', 'bring']:
            return 'causes'
        
        # Process relationships
        elif verb_lemma in ['evaporate', 'condense', 'precipitate', 'freeze', 'melt']:
            return f'{verb_lemma}s_into'
        
        # Control and regulation relationships
        elif verb_lemma in ['control', 'regulate', 'manage', 'govern', 'moderate']:
            return 'regulates'
        
        # Support and enable relationships
        elif verb_lemma in ['support', 'enable', 'allow', 'permit', 'facilitate', 'help']:
            return 'enables'
        
        # Influence relationships
        elif verb_lemma in ['influence', 'affect', 'impact', 'modify', 'alter']:
            return 'influences'
        
        # Containment relationships
        elif verb_lemma in ['contain', 'hold', 'store', 'keep', 'house']:
            return 'contains'
        
        # Transportation relationships
        elif verb_lemma in ['transport', 'carry', 'convey', 'deliver', 'transfer']:
            return 'transports'
        
        # Dependency relationships
        elif verb_lemma in ['depend', 'rely', 'require', 'need']:
            return 'depends_on'
        
        # Being/existence relationships
        elif verb_lemma in ['be', 'is', 'are', 'was', 'were', 'exist', 'represent']:
            # Check for different types of "being" relationships
            if any(token.text.lower() in ['type', 'kind', 'form', 'example'] for token in path):
                return 'is_type_of'
            elif any(token.text.lower() in ['part', 'component', 'element'] for token in path):
                return 'part_of'
            else:
                return 'is'
        
        # Possession relationships
        elif verb_lemma in ['have', 'has', 'had', 'own', 'possess', 'include']:
            return 'has'
        
        # Work/employment relationships
        elif verb_lemma in ['work', 'employ', 'serve']:
            return 'works_for'
        
        # Use context to create more specific relationship types
        # Look for domain-specific verbs and create appropriate relationships
        else:
            # For unknown verbs, create a relationship type based on the verb itself
            # This allows the system to adapt to any domain dynamically
            return f'{verb_lemma}s'
    
    def _normalize_concept_name(self, name: str) -> str:
        """Normalize concept name for consistent IDs"""
        return re.sub(r'[^\w\s]', '', name).lower().replace(' ', '_')
    
    def link_observations_to_concepts(self, observations: List[str], 
                                    concepts: List[ConceptNode]) -> Dict[str, List[str]]:
        """
        Link observations to relevant concepts using semantic similarity
        
        Args:
            observations: List of observation texts
            concepts: List of concept nodes
            
        Returns:
            Dict mapping concept IDs to lists of relevant observations
        """
        if not observations or not concepts:
            return {}
        
        # Generate embeddings for observations and concepts
        obs_embeddings = self.embedding_model.encode(observations)
        concept_embeddings = self.embedding_model.encode([c.name for c in concepts])
        
        # Calculate similarity matrix
        similarity_matrix = np.dot(obs_embeddings, concept_embeddings.T)
        
        # Link observations to concepts based on similarity threshold
        concept_observations = defaultdict(list)
        threshold = 0.3  # Minimum similarity threshold
        
        for obs_idx, observation in enumerate(observations):
            similarities = similarity_matrix[obs_idx]
            for concept_idx, similarity in enumerate(similarities):
                if similarity > threshold:
                    concept_id = concepts[concept_idx].id
                    concept_observations[concept_id].append(observation)
        
        return dict(concept_observations)
    
    def construct_knowledge_graph(self, text: str, source: str = "unknown") -> Dict:
        """
        Construct a complete knowledge graph from text
        
        Args:
            text: Input text to process
            source: Source identifier for the text
            
        Returns:
            Dictionary containing entities, relationships, and observations
        """
        # Extract concepts
        concepts = self.extract_concepts(text)
        
        # Extract relationships
        relationships = self.extract_semantic_relationships(text, concepts)
        
        # Extract observations (sentences as atomic facts)
        observations = [sent.strip() for sent in text.split('.') if len(sent.strip()) > 10]
        
        # Link observations to concepts
        concept_observations = self.link_observations_to_concepts(observations, concepts)
        
        # Update concept observations
        for concept in concepts:
            if concept.id in concept_observations:
                concept.observations.extend(concept_observations[concept.id])
        
        # Format output
        entities = []
        for concept in concepts:
            entities.append({
                "name": concept.name,
                "entityType": concept.type.lower(),
                "observations": concept.observations
            })
        
        relations = []
        for rel in relationships:
            relations.append({
                "from": next((c.name for c in concepts if c.id == rel.source), rel.source),
                "to": next((c.name for c in concepts if c.id == rel.target), rel.target),
                "relationType": rel.relation_type
            })
        
        return {
            "entities": entities,
            "relationships": relations,
            "observations": observations,
            "source": source,
            "stats": {
                "entity_count": len(entities),
                "relationship_count": len(relations),
                "observation_count": len(observations),
                "relationship_types": list(set(rel.relation_type for rel in relationships))
            }
        }
