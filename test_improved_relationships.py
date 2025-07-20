#!/usr/bin/env python3
"""
Test script to demonstrate improved knowledge graph construction 
with domain-specific water cycle relationships
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mcp_server'))

from enhanced_kg_construction_improved import EnhancedKGConstructor

def test_water_cycle_relationships():
    """Test the improved knowledge graph constructor with water cycle content"""
    
    # Initialize with water_cycle domain
    constructor = EnhancedKGConstructor(domain="water_cycle")
    
    # Water cycle text for testing
    water_cycle_text = """
    Water evaporates from oceans due to solar energy. The sun drives the evaporation process.
    Water vapor condenses into clouds in the atmosphere. Clouds precipitate as rain.
    Rain flows into rivers. Rivers transport water to the ocean.
    Some water infiltrates into groundwater. Groundwater stores water underground.
    Plants absorb water and transpire water vapor. Evapotranspiration contributes to atmospheric moisture.
    The water cycle regulates Earth's temperature. Oceans store most of Earth's water.
    """
    
    # Construct knowledge graph
    print("Constructing knowledge graph with domain-specific relationships...")
    kg = constructor.construct_knowledge_graph(water_cycle_text, "water_cycle_test")
    
    print(f"\n=== KNOWLEDGE GRAPH STATS ===")
    print(f"Entities: {kg['stats']['entity_count']}")
    print(f"Relationships: {kg['stats']['relationship_count']}")
    print(f"Observations: {kg['stats']['observation_count']}")
    print(f"Relationship Types: {len(kg['stats']['relationship_types'])}")
    
    print(f"\n=== RELATIONSHIP TYPES FOUND ===")
    for rel_type in sorted(kg['stats']['relationship_types']):
        print(f"- {rel_type}")
    
    print(f"\n=== SAMPLE RELATIONSHIPS ===")
    for i, rel in enumerate(kg['relationships'][:15]):  # Show first 15
        print(f"{i+1:2d}. {rel['from']} --[{rel['relationType']}]--> {rel['to']}")
    
    print(f"\n=== ENTITIES ===")
    for entity in kg['entities'][:10]:  # Show first 10
        print(f"- {entity['name']} ({entity['entityType']})")
    
    return kg

if __name__ == "__main__":
    test_water_cycle_relationships()
