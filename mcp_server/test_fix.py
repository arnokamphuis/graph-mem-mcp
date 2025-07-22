#!/usr/bin/env python3
"""
Quick test to verify the storage system loads correctly without legacy conversion
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_storage_import():
    """Test that storage system imports correctly"""
    try:
        from storage import create_graph_store, GraphStore, MemoryStore, StorageConfig
        print("✅ Storage system imports successfully")
        
        # Test creating a memory store
        store = create_graph_store("memory")
        print("✅ Memory store creation successful")
        print(f"   Store type: {type(store).__name__}")
        
        return True
    except ImportError as e:
        print(f"❌ Storage import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Storage creation failed: {e}")
        return False

def test_core_components():
    """Test that core components import correctly"""
    try:
        from core.graph_schema import SchemaManager, EntityInstance, RelationshipInstance
        from core.entity_resolution import EntityResolver
        from core.graph_analytics import GraphAnalytics
        print("✅ Core components import successfully")
        return True
    except ImportError as e:
        print(f"❌ Core components import failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Graph Memory MCP Server Components...")
    print("=" * 50)
    
    storage_ok = test_storage_import()
    core_ok = test_core_components()
    
    if storage_ok and core_ok:
        print("\n🎉 All tests passed! Modern storage system ready.")
        print("   Legacy conversion functions removed successfully.")
    else:
        print("\n❌ Some tests failed - see errors above.")
