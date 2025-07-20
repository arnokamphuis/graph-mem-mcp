"""
Enhanced Knowledge Graph Construction Module

This module provides advanced text-to-knowledge-graph construction using:
- spaCy for Named Entity Recognition and dependency parsing
- sentence-transformers for semantic similarity and concept clustering
- Advanced relationship extraction with semantic understanding
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
    """Advanced Knowledge Graph Constructor using modern NLP techniques"""
    
    def __init__(self, spacy_model: str = "en_core_web_sm", 
                 sentence_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the enhanced knowledge graph constructor
        
        Args:
            spacy_model: spaCy model to use for NLP processing
            sentence_model: Sentence transformer model for embeddings
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"spaCy model {spacy_model} not found. Please install with: python -m spacy download {spacy_model}")
            # Fallback to basic model
            self.nlp = spacy.load("en_core_web_sm")
        
        self.sentence_model = SentenceTransformer(sentence_model)
        
        # Semantic relationship patterns
        self.relation_patterns = {
            'is_a': ['is', 'are', 'was', 'were', 'being', 'been'],
            'has': ['has', 'have', 'had', 'owns', 'contains', 'includes'],
            'located_in': ['in', 'at', 'located', 'based', 'situated'],
            'works_for': ['works for', 'employed by', 'works at', 'employee of'],
            'part_of': ['part of', 'member of', 'belongs to', 'component of'],
            'created_by': ['created by', 'made by', 'developed by', 'authored by'],
            'causes': ['causes', 'leads to', 'results in', 'triggers'],
            'used_for': ['used for', 'utilized for', 'applied to', 'employed for']
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
        
        # Add semantic embeddings
        concept_list = list(concepts.values())
        if concept_list:
            concept_texts = [concept.name for concept in concept_list]
            embeddings = self.sentence_model.encode(concept_texts)
            for concept, embedding in zip(concept_list, embeddings):
                concept.embedding = embedding
        
        return concept_list
    
    def cluster_similar_concepts(self, concepts: List[ConceptNode], 
                                similarity_threshold: float = 0.8) -> List[ConceptNode]:
        """
        Cluster similar concepts and merge them
        
        Args:
            concepts: List of ConceptNode objects
            similarity_threshold: Threshold for considering concepts similar
            
        Returns:
            List of merged ConceptNode objects
        """
        if len(concepts) < 2:
            return concepts
        
        # Create embedding matrix
        embeddings = np.array([concept.embedding for concept in concepts if concept.embedding is not None])
        valid_concepts = [concept for concept in concepts if concept.embedding is not None]
        
        if len(embeddings) < 2:
            return concepts
        
        # Use DBSCAN clustering for automatic cluster detection
        clustering = DBSCAN(eps=1-similarity_threshold, min_samples=1, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Group concepts by cluster
        clusters = defaultdict(list)
        for concept, label in zip(valid_concepts, cluster_labels):
            clusters[label].append(concept)
        
        # Merge concepts in each cluster
        merged_concepts = []
        for cluster_concepts in clusters.values():
            if len(cluster_concepts) == 1:
                merged_concepts.append(cluster_concepts[0])
            else:
                # Merge multiple concepts
                main_concept = cluster_concepts[0]
                for other_concept in cluster_concepts[1:]:
                    main_concept.aliases.extend(other_concept.aliases)
                    main_concept.observations.extend(other_concept.observations)
                    # Use the concept with highest confidence as the main name
                    if other_concept.confidence > main_concept.confidence:
                        main_concept.name = other_concept.name
                        main_concept.confidence = other_concept.confidence
                
                # Remove duplicates from aliases
                main_concept.aliases = list(set(main_concept.aliases))
                merged_concepts.append(main_concept)
        
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
        """Classify relationship type based on dependency path"""
        if not path:
            return None
        
        # Simple classification based on dependency labels and POS tags
        verbs_in_path = [token for token in path if token.pos_ == 'VERB']
        if verbs_in_path:
            verb = verbs_in_path[0].lemma_.lower()
            if verb in ['be', 'is', 'are', 'was', 'were']:
                return 'is_a'
            elif verb in ['have', 'has', 'had', 'own']:
                return 'has'
            elif verb in ['work', 'employ']:
                return 'works_for'
            elif verb in ['create', 'make', 'develop']:
                return 'created_by'
            elif verb in ['cause', 'lead', 'result']:
                return 'causes'
            else:
                return 'related_to'
        
        return 'related_to'
    
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
            Dictionary mapping concept IDs to relevant observations
        """
        if not observations or not concepts:
            return {}
        
        concept_links = defaultdict(list)
        
        # Get embeddings for observations
        obs_embeddings = self.sentence_model.encode(observations)
        
        for i, observation in enumerate(observations):
            obs_embedding = obs_embeddings[i]
            best_concept = None
            best_similarity = 0.0
            
            # Find the most similar concept
            for concept in concepts:
                if concept.embedding is not None:
                    similarity = np.dot(obs_embedding, concept.embedding) / (
                        np.linalg.norm(obs_embedding) * np.linalg.norm(concept.embedding)
                    )
                    if similarity > best_similarity and similarity > 0.4:  # Threshold for relevance
                        best_similarity = similarity
                        best_concept = concept
            
            # Also check for explicit mentions
            obs_lower = observation.lower()
            for concept in concepts:
                for alias in concept.aliases:
                    if alias.lower() in obs_lower:
                        concept_links[concept.id].append(observation)
                        break
            
            # Link to best semantic match if found
            if best_concept and observation not in concept_links[best_concept.id]:
                concept_links[best_concept.id].append(observation)
        
        return dict(concept_links)
    
    def construct_knowledge_graph(self, text: str, existing_observations: List[str] = None) -> Dict:
        """
        Construct a complete knowledge graph from text
        
        Args:
            text: Input text to process
            existing_observations: Existing observations to link
            
        Returns:
            Dictionary containing nodes, edges, and observation links
        """
        logger.info("Starting enhanced knowledge graph construction")
        
        # Extract concepts
        concepts = self.extract_concepts(text)
        logger.info(f"Extracted {len(concepts)} initial concepts")
        
        # Cluster similar concepts
        concepts = self.cluster_similar_concepts(concepts)
        logger.info(f"After clustering: {len(concepts)} concepts")
        
        # Extract relationships
        relationships = self.extract_semantic_relationships(text, concepts)
        logger.info(f"Extracted {len(relationships)} relationships")
        
        # Link observations
        observation_links = {}
        if existing_observations:
            observation_links = self.link_observations_to_concepts(existing_observations, concepts)
            logger.info(f"Linked observations to {len(observation_links)} concepts")
        
        # Add text-based observations to concepts
        sentences = [sent.strip() for sent in text.split('.') if sent.strip()]
        text_observation_links = self.link_observations_to_concepts(sentences, concepts)
        
        # Merge observation links
        for concept_id, observations in text_observation_links.items():
            if concept_id in observation_links:
                observation_links[concept_id].extend(observations)
            else:
                observation_links[concept_id] = observations
        
        return {
            'concepts': concepts,
            'relationships': relationships,
            'observation_links': observation_links,
            'stats': {
                'total_concepts': len(concepts),
                'total_relationships': len(relationships),
                'linked_observations': sum(len(obs) for obs in observation_links.values())
            }
        }
