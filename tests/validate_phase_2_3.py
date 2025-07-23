#!/usr/bin/env python3
"""
Phase 2.3 Coreference Resolution Validation Suite
Direct validation testing following prompt quality gate requirements
"""

import sys
sys.path.append('.')

def test_import_validation():
    """Test 1: Import Validation"""
    print("\n🧪 Test 1: Import Validation")
    try:
        # Test core imports
        from mcp_server.extraction.coreference_resolver import (
            CoreferenceResolver, ResolutionContext, CoreferenceCluster,
            ResolutionCandidate, ReferenceType, ResolutionMethod,
            resolve_coreferences_quick
        )
        print("  ✅ Core coreference resolution imports successful")
        
        # Test Phase 1 core integration
        from mcp_server.core.graph_schema import EntityInstance, SchemaManager
        print("  ✅ Phase 1 core imports successful")
        
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_factory_creation():
    """Test 2: Factory Creation"""
    print("\n🧪 Test 2: Factory Creation")
    try:
        from mcp_server.extraction.coreference_resolver import create_coreference_resolver
        
        resolver = create_coreference_resolver()
        print("  ✅ Factory created resolver successfully")
        
        # Check that it's the right type
        if hasattr(resolver, 'resolve_coreferences'):
            print("  ✅ Resolver has resolve_coreferences method")
        else:
            print("  ❌ Missing resolve_coreferences method")
            return False
            
        # Check stats initialization
        if hasattr(resolver, 'stats'):
            stats = resolver.stats
            print(f"  ✅ Statistics initialized: {len(stats)} metrics")
        else:
            print("  ❌ Missing stats")
            return False
        
        return True
    except Exception as e:
        print(f"  ❌ Factory creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_resolver_initialization():
    """Test 3: CoreferenceResolver Initialization"""
    print("\n🧪 Test 3: CoreferenceResolver Initialization")
    try:
        from mcp_server.extraction.coreference_resolver import CoreferenceResolver
        
        resolver = CoreferenceResolver()
        print("  ✅ CoreferenceResolver created successfully")
        
        # Check stats initialization
        if hasattr(resolver, 'stats'):
            stats = resolver.stats
            print(f"  ✅ Statistics initialized: {len(stats)} metrics")
            
            # Check key resolution methods
            methods = stats.get('resolution_methods_enabled', {})
            print(f"  ✅ Resolution methods: {len(methods)} strategies enabled")
            for method, enabled in methods.items():
                status = "✅" if enabled else "⚠️"
                print(f"    {status} {method}: {enabled}")
        
        return True
    except Exception as e:
        print(f"  ❌ Resolver initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_resolution_context():
    """Test 4: ResolutionContext Creation"""
    print("\n🧪 Test 4: ResolutionContext Creation")
    try:
        from mcp_server.extraction.coreference_resolver import ResolutionContext
        
        context = ResolutionContext(
            text="John Smith is a software engineer. He works at Google. The company is very successful.",
            confidence_threshold=0.5,
            enable_pronoun_resolution=True,
            enable_nominal_resolution=True,
            enable_proper_noun_resolution=True
        )
        print("  ✅ ResolutionContext created successfully")
        print(f"  ✅ Text: {context.text[:50]}...")
        print(f"  ✅ Confidence threshold: {context.confidence_threshold}")
        print(f"  ✅ Pronoun resolution: {context.enable_pronoun_resolution}")
        print(f"  ✅ Nominal resolution: {context.enable_nominal_resolution}")
        print(f"  ✅ Proper noun resolution: {context.enable_proper_noun_resolution}")
        
        return True
    except Exception as e:
        print(f"  ❌ ResolutionContext creation failed: {e}")
        return False

def test_quick_resolution():
    """Test 5: Quick Resolution Function"""
    print("\n🧪 Test 5: Quick Resolution Function")
    try:
        from mcp_server.extraction.coreference_resolver import resolve_coreferences_quick
        
        test_text = "John Smith is a software engineer. He works at Google Inc. The company hired him last year."
        
        print("  🔍 Attempting coreference resolution...")
        clusters, candidates = resolve_coreferences_quick(
            text=test_text,
            confidence_threshold=0.3
        )
        
        print(f"  ✅ Resolution completed: {len(clusters)} clusters, {len(candidates)} candidates")
        
        # Show first few results
        if clusters:
            for i, cluster in enumerate(clusters[:2]):
                print(f"    Cluster {i+1}: {cluster.get_canonical_text()} ({len(cluster.mentions)} mentions)")
        
        if candidates:
            for i, candidate in enumerate(candidates[:2]):
                print(f"    Candidate {i+1}: '{candidate.mention.text}' -> '{candidate.antecedent.text}' (conf: {candidate.confidence:.2f})")
        
        return True
    except Exception as e:
        print(f"  ❌ Quick resolution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_resolution_pipeline():
    """Test 6: Full Resolution Pipeline"""
    print("\n🧪 Test 6: Full Resolution Pipeline")
    try:
        from mcp_server.extraction.coreference_resolver import CoreferenceResolver, ResolutionContext
        
        resolver = CoreferenceResolver()
        context = ResolutionContext(
            text="Mary Johnson founded TechCorp. She is the CEO. The organization has grown rapidly under her leadership.",
            confidence_threshold=0.4,
            enable_pronoun_resolution=True,
            enable_nominal_resolution=True,
            enable_proper_noun_resolution=True
        )
        
        print("  🔍 Attempting full pipeline resolution...")
        clusters, candidates = resolver.resolve_coreferences(context)
        
        print(f"  ✅ Full pipeline completed: {len(clusters)} clusters, {len(candidates)} candidates")
        
        # Validate results structure
        for cluster in clusters:
            if hasattr(cluster, 'cluster_id') and hasattr(cluster, 'mentions'):
                print(f"    ✅ Cluster {cluster.cluster_id}: {len(cluster.mentions)} mentions")
            else:
                print(f"    ⚠️ Cluster missing required attributes")
                
        for candidate in candidates[:2]:
            if hasattr(candidate, 'confidence') and hasattr(candidate, 'resolution_method'):
                print(f"    ✅ Candidate: {candidate.resolution_method.value} (conf: {candidate.confidence:.2f})")
            else:
                print(f"    ⚠️ Candidate missing required attributes")
        
        return True
    except Exception as e:
        print(f"  ❌ Full pipeline resolution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_validation_suite():
    """Run complete Phase 2.3 validation suite"""
    print("="*60)
    print("🧪 PHASE 2.3 COREFERENCE RESOLUTION VALIDATION SUITE")
    print("="*60)
    
    tests = [
        ("Import Validation", test_import_validation),
        ("Factory Creation", test_factory_creation),
        ("Resolver Initialization", test_resolver_initialization),
        ("Resolution Context", test_resolution_context),
        ("Quick Resolution", test_quick_resolution),
        ("Full Pipeline", test_full_resolution_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATION RESULTS")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print("-"*60)
    coverage = (passed / total * 100) if total > 0 else 0
    print(f"Tests Passed: {passed}/{total} ({coverage:.1f}%)")
    
    if coverage >= 90:
        print("✅ QUALITY GATE PASSED: 90%+ test success")
        return True
    else:
        print("❌ QUALITY GATE FAILED: <90% test success")
        return False

if __name__ == "__main__":
    success = run_validation_suite()
    if success:
        print("\n🎉 Phase 2.3 validation SUCCESSFUL!")
        exit(0)
    else:
        print("\n⚠️ Phase 2.3 validation INCOMPLETE")
        exit(1)
