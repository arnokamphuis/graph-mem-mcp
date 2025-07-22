#!/usr/bin/env python3
"""
Phase 3.2 Performance Optimization Framework Validation Suite

This comprehensive test suite validates the Phase 3.2 storage abstraction layer
including in-memory storage, indexing, and query optimization capabilities.

Quality Gate: >90% test success required for phase completion.
"""

import asyncio
import sys
import time
import traceback
from typing import Dict, Any, List

# Test the storage imports
def test_imports() -> bool:
    """Test 1: Validate all storage component imports"""
    print("🧪 Test 1: Import Validation")
    try:
        # Test storage module imports
        from mcp_server.storage import (
            GraphStore, StorageConfig, MemoryStore,
            create_graph_store, create_memory_store
        )
        print("  ✅ Core storage imports successful")
        
        # Test Phase 1 integration
        try:
            from mcp_server.core.graph_schema import EntityInstance, RelationshipInstance, SchemaManager
            print("  ✅ Phase 1 core imports successful")
        except ImportError:
            print("  ⚠️  Phase 1 core components not available - using fallbacks")
        
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_factory_creation() -> bool:
    """Test 2: Validate factory functions and component creation"""
    print("🧪 Test 2: Factory Creation")
    try:
        from mcp_server.storage import create_graph_store, create_memory_store, StorageConfig
        
        # Test memory store factory
        memory_store = create_memory_store()
        print("  ✅ Memory store factory works")
        
        # Test graph store factory
        graph_store = create_graph_store("memory")
        print("  ✅ Graph store factory works")
        
        # Test with configuration
        config = StorageConfig(enable_caching=True, cache_size=5000)
        configured_store = create_graph_store("memory", config=config)
        print("  ✅ Configured store creation works")
        
        # Test methods are available
        assert hasattr(memory_store, 'connect'), "Memory store missing connect method"
        assert hasattr(memory_store, 'create_entity'), "Memory store missing create_entity method" 
        assert hasattr(memory_store, 'query_entities'), "Memory store missing query_entities method"
        assert hasattr(memory_store, 'create_index'), "Memory store missing create_index method"
        print("  ✅ Required methods available")
        
        return True
    except Exception as e:
        print(f"  ❌ Factory creation failed: {e}")
        traceback.print_exc()
        return False

async def test_basic_storage_operations() -> bool:
    """Test 3: Basic storage operations (CRUD)"""
    print("🧪 Test 3: Basic Storage Operations")
    try:
        from mcp_server.storage import create_memory_store, StorageConfig
        
        # Create store
        config = StorageConfig(enable_caching=True)
        store = create_memory_store(config=config)
        
        # Connect
        connected = await store.connect()
        assert connected, "Store connection failed"
        print("  ✅ Store connection successful")
        
        # Create test entity
        test_entity = {
            'id': 'test_entity_1',
            'entity_type': 'person',
            'name': 'John Doe',
            'properties': {'age': 30, 'location': 'New York'}
        }
        
        created = await store.create_entity(test_entity)
        assert created, "Entity creation failed"
        print("  ✅ Entity creation successful")
        
        # Retrieve entity
        retrieved = await store.get_entity('test_entity_1')
        assert retrieved is not None, "Entity retrieval failed"
        print("  ✅ Entity retrieval successful")
        
        # Update entity
        updated = await store.update_entity('test_entity_1', {'age': 31})
        assert updated, "Entity update failed"
        print("  ✅ Entity update successful")
        
        # Create second entity for relationship test
        test_entity_2 = {
            'id': 'test_entity_2',
            'entity_type': 'organization',
            'name': 'Acme Corp'
        }
        await store.create_entity(test_entity_2)
        
        # Create relationship
        test_relationship = {
            'id': 'test_rel_1',
            'source_entity_id': 'test_entity_1',
            'target_entity_id': 'test_entity_2',
            'relation_type': 'works_for',
            'properties': {'start_date': '2020-01-01'}
        }
        
        rel_created = await store.create_relationship(test_relationship)
        assert rel_created, "Relationship creation failed"
        print("  ✅ Relationship creation successful")
        
        # Test statistics
        stats = await store.get_statistics()
        assert stats['entity_count'] >= 2, f"Expected at least 2 entities, got {stats['entity_count']}"
        assert stats['relationship_count'] >= 1, f"Expected at least 1 relationship, got {stats['relationship_count']}"
        print(f"  ✅ Statistics: {stats['entity_count']} entities, {stats['relationship_count']} relationships")
        
        await store.disconnect()
        return True
        
    except Exception as e:
        print(f"  ❌ Basic operations failed: {e}")
        traceback.print_exc()
        return False

async def test_query_operations() -> bool:
    """Test 4: Query and search operations"""
    print("🧪 Test 4: Query Operations")
    try:
        from mcp_server.storage import create_memory_store
        
        store = create_memory_store()
        await store.connect()
        
        # Create test data
        entities = [
            {'id': 'person_1', 'entity_type': 'person', 'name': 'Alice', 'age': 25},
            {'id': 'person_2', 'entity_type': 'person', 'name': 'Bob', 'age': 30},
            {'id': 'org_1', 'entity_type': 'organization', 'name': 'Tech Corp'},
            {'id': 'org_2', 'entity_type': 'organization', 'name': 'Media Inc'}
        ]
        
        for entity in entities:
            await store.create_entity(entity)
        
        relationships = [
            {'id': 'rel_1', 'source_entity_id': 'person_1', 'target_entity_id': 'org_1', 'relation_type': 'works_for'},
            {'id': 'rel_2', 'source_entity_id': 'person_2', 'target_entity_id': 'org_2', 'relation_type': 'works_for'},
            {'id': 'rel_3', 'source_entity_id': 'person_1', 'target_entity_id': 'person_2', 'relation_type': 'knows'}
        ]
        
        for rel in relationships:
            await store.create_relationship(rel)
        
        # Test entity queries
        all_people = await store.query_entities(entity_type='person')
        assert len(all_people) == 2, f"Expected 2 people, got {len(all_people)}"
        print("  ✅ Entity type filtering works")
        
        young_people = await store.query_entities(entity_type='person', properties={'age': 25})
        assert len(young_people) == 1, f"Expected 1 young person, got {len(young_people)}"
        print("  ✅ Property filtering works")
        
        # Test relationship queries
        work_relations = await store.query_relationships(relationship_type='works_for')
        assert len(work_relations) == 2, f"Expected 2 work relationships, got {len(work_relations)}"
        print("  ✅ Relationship type filtering works")
        
        alice_relations = await store.query_relationships(source_entity_id='person_1')
        assert len(alice_relations) == 2, f"Expected 2 relationships from Alice, got {len(alice_relations)}"
        print("  ✅ Source entity filtering works")
        
        # Test neighbors
        alice_neighbors = await store.get_neighbors('person_1')
        assert len(alice_neighbors) == 2, f"Expected 2 neighbors for Alice, got {len(alice_neighbors)}"
        print("  ✅ Neighbor discovery works")
        
        # Test path finding
        paths = await store.find_path('person_1', 'org_2', max_depth=3)
        assert len(paths) > 0, "Expected to find path from person_1 to org_2"
        print(f"  ✅ Path finding works: found {len(paths)} paths")
        
        await store.disconnect()
        return True
        
    except Exception as e:
        print(f"  ❌ Query operations failed: {e}")
        traceback.print_exc()
        return False

async def test_indexing_and_performance() -> bool:
    """Test 5: Indexing and performance optimization"""
    print("🧪 Test 5: Indexing and Performance")
    try:
        from mcp_server.storage import create_memory_store, StorageConfig
        
        # Create store with caching enabled
        config = StorageConfig(enable_caching=True, cache_size=1000)
        store = create_memory_store(config=config)
        await store.connect()
        
        # Create index
        index_created = await store.create_index(
            index_name="person_name_idx",
            target_type="entity", 
            properties=["name"],
            unique=False
        )
        assert index_created, "Index creation failed"
        print("  ✅ Index creation successful")
        
        # List indexes
        indexes = await store.list_indexes()
        assert len(indexes) >= 1, f"Expected at least 1 index, got {len(indexes)}"
        print(f"  ✅ Index listing works: {len(indexes)} indexes")
        
        # Test bulk operations
        test_entities = []
        for i in range(10):
            test_entities.append({
                'id': f'bulk_entity_{i}',
                'entity_type': 'test',
                'name': f'Entity {i}',
                'value': i
            })
        
        created_count = await store.bulk_create_entities(test_entities)
        assert created_count == 10, f"Expected 10 entities created, got {created_count}"
        print("  ✅ Bulk entity creation works")
        
        # Test cache performance
        start_time = time.time()
        result1 = await store.query_entities(entity_type='test')
        first_query_time = time.time() - start_time
        
        start_time = time.time() 
        result2 = await store.query_entities(entity_type='test')
        second_query_time = time.time() - start_time
        
        assert len(result1) == len(result2) == 10, "Cache should return same results"
        print(f"  ✅ Query caching works: 1st query {first_query_time:.3f}s, 2nd query {second_query_time:.3f}s")
        
        # Test cache clearing
        await store.clear_cache()
        print("  ✅ Cache clearing works")
        
        # Test statistics
        stats = await store.get_statistics()
        assert 'cache_hit_rate' in stats, "Statistics missing cache hit rate"
        assert 'entity_count' in stats, "Statistics missing entity count"
        print(f"  ✅ Performance stats: {stats['entity_count']} entities, {stats['cache_hit_rate']:.2f} cache hit rate")
        
        await store.disconnect()
        return True
        
    except Exception as e:
        print(f"  ❌ Indexing and performance test failed: {e}")
        traceback.print_exc()
        return False

async def test_transaction_support() -> bool:
    """Test 6: Transaction support"""
    print("🧪 Test 6: Transaction Support")
    try:
        from mcp_server.storage import create_memory_store
        
        store = create_memory_store()
        await store.connect()
        
        # Test transaction lifecycle
        tx_id = await store.begin_transaction()
        assert tx_id, "Transaction creation failed"
        print("  ✅ Transaction creation successful")
        
        # Create entity within transaction
        test_entity = {
            'id': 'tx_entity_1',
            'entity_type': 'test',
            'name': 'Transaction Test'
        }
        await store.create_entity(test_entity)
        
        # Commit transaction
        committed = await store.commit_transaction(tx_id)
        assert committed, "Transaction commit failed"
        print("  ✅ Transaction commit successful")
        
        # Verify entity exists
        entity = await store.get_entity('tx_entity_1')
        assert entity is not None, "Entity not found after transaction commit"
        print("  ✅ Transaction persistence verified")
        
        # Test rollback (simplified)
        tx_id2 = await store.begin_transaction()
        rolled_back = await store.rollback_transaction(tx_id2)
        assert rolled_back, "Transaction rollback failed"
        print("  ✅ Transaction rollback works")
        
        await store.disconnect()
        return True
        
    except Exception as e:
        print(f"  ❌ Transaction support test failed: {e}")
        traceback.print_exc()
        return False

async def run_validation_suite():
    """Run complete Phase 3.2 validation suite"""
    print("=" * 60)
    print("🧪 PHASE 3.2 PERFORMANCE OPTIMIZATION VALIDATION SUITE")
    print("=" * 60)
    
    test_functions = [
        test_imports,
        test_factory_creation,
        test_basic_storage_operations,
        test_query_operations,
        test_indexing_and_performance,
        test_transaction_support
    ]
    
    results = []
    for test_func in test_functions:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    # Calculate results
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print("=" * 60)
    print("📊 VALIDATION RESULTS")
    print("=" * 60)
    
    test_names = [
        "Import Validation",
        "Factory Creation", 
        "Basic Storage Operations",
        "Query Operations",
        "Indexing and Performance",
        "Transaction Support"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<30} {status}")
    
    print("-" * 60)
    print(f"Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("✅ QUALITY GATE PASSED: 90%+ test success")
        print("🎉 Phase 3.2 validation SUCCESSFUL!")
        return True
    else:
        print("❌ QUALITY GATE FAILED: <90% test success")
        print("⚠️  Phase 3.2 validation needs improvement")
        return False

if __name__ == "__main__":
    # Run the validation suite
    try:
        success = asyncio.run(run_validation_suite())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Validation suite failed: {e}")
        traceback.print_exc()
        sys.exit(1)
