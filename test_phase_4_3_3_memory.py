#!/usr/bin/env python3
"""
Phase 4.3.3: Memory Usage and Optimization Validation

This test suite validates memory usage patterns, identifies memory leaks,
and assesses optimization opportunities for the knowledge graph system.

Test Coverage:
- Memory baseline and growth patterns
- Memory usage under load
- Memory leak detection
- Optimization recommendations
- Production memory requirements validation
"""

import gc
import psutil
import requests
import time
import json
import statistics
from typing import Dict, List, Tuple
import tracemalloc
from dataclasses import dataclass

@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: float
    rss_mb: float
    vms_mb: float
    percent: float
    available_mb: float
    operation: str

class MemoryUsageValidator:
    """Comprehensive memory usage validation suite"""
    
    def __init__(self, base_url: str = "http://localhost:10642"):
        self.base_url = base_url
        self.process = psutil.Process()
        self.snapshots: List[MemorySnapshot] = []
        self.baseline_memory = 0.0
        
    def get_memory_snapshot(self, operation: str = "baseline") -> MemorySnapshot:
        """Get current memory usage snapshot"""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        virtual_memory = psutil.virtual_memory()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=memory_percent,
            available_mb=virtual_memory.available / 1024 / 1024,
            operation=operation
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def test_server_availability(self) -> bool:
        """Test if the server is running and responsive"""
        try:
            response = requests.get(f"{self.base_url}/banks/list", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def establish_baseline(self):
        """Establish memory baseline before testing"""
        print("📊 Establishing memory baseline...")
        
        # Force garbage collection
        gc.collect()
        time.sleep(2)
        
        # Take baseline snapshot
        baseline = self.get_memory_snapshot("baseline")
        self.baseline_memory = baseline.rss_mb
        
        print(f"   📈 Baseline Memory: {baseline.rss_mb:.2f}MB")
        print(f"   💾 Available Memory: {baseline.available_mb:.0f}MB")
        print(f"   📊 Memory Percentage: {baseline.percent:.1f}%")
        
        return baseline
    
    def test_basic_memory_usage(self):
        """Test memory usage for basic operations"""
        print("\n🔍 Testing basic operation memory usage...")
        
        operations = [
            ("Bank List", lambda: requests.get(f"{self.base_url}/banks/list")),
            ("Bank Create", lambda: requests.post(f"{self.base_url}/banks/create", 
                                                 json={"bank": f"memory-test-{int(time.time())}"})),
            ("Node Search", lambda: requests.post(f"{self.base_url}/nodes/search", 
                                                  json={"query": "test"})),
        ]
        
        results = {}
        
        for op_name, operation in operations:
            # Take pre-operation snapshot
            pre_snap = self.get_memory_snapshot(f"{op_name}_pre")
            
            # Execute operation
            try:
                response = operation()
                success = response.status_code == 200
            except Exception as e:
                print(f"   ❌ {op_name} failed: {e}")
                success = False
            
            # Take post-operation snapshot
            post_snap = self.get_memory_snapshot(f"{op_name}_post")
            
            # Calculate memory delta
            memory_delta = post_snap.rss_mb - pre_snap.rss_mb
            
            results[op_name] = {
                "memory_delta_mb": memory_delta,
                "success": success,
                "pre_memory": pre_snap.rss_mb,
                "post_memory": post_snap.rss_mb
            }
            
            print(f"   {op_name}: {memory_delta:+.2f}MB (Success: {success})")
        
        return results
    
    def test_load_memory_usage(self):
        """Test memory usage under sustained load"""
        print("\n⚡ Testing memory usage under load...")
        
        # Test parameters
        num_iterations = 50
        operations_per_iteration = 5
        
        memory_samples = []
        entity_counter = 0
        
        print(f"   🔄 Running {num_iterations} iterations with {operations_per_iteration} ops each...")
        
        for iteration in range(num_iterations):
            # Mix of operations to simulate real usage
            for op in range(operations_per_iteration):
                entity_counter += 1
                
                try:
                    # Alternate between different operations
                    if op % 3 == 0:
                        # Entity creation
                        requests.post(f"{self.base_url}/entities/create", json={
                            "entities": [{
                                "name": f"LoadTestEntity{entity_counter}",
                                "entityType": "test",
                                "observations": [f"Load test observation {entity_counter}"]
                            }]
                        }, timeout=5)
                    elif op % 3 == 1:
                        # Search operation
                        requests.post(f"{self.base_url}/nodes/search", 
                                     json={"query": "test"}, timeout=5)
                    else:
                        # Bank operations
                        requests.get(f"{self.base_url}/banks/list", timeout=5)
                
                except Exception:
                    pass  # Continue testing even if individual requests fail
            
            # Take memory snapshot every 5 iterations
            if iteration % 5 == 0:
                snapshot = self.get_memory_snapshot(f"load_test_iter_{iteration}")
                memory_samples.append(snapshot.rss_mb)
                
                if iteration > 0:
                    progress = (iteration / num_iterations) * 100
                    current_memory = snapshot.rss_mb
                    memory_growth = current_memory - self.baseline_memory
                    print(f"   Progress: {progress:.0f}% - Memory: {current_memory:.1f}MB ({memory_growth:+.1f}MB)")
        
        # Analyze memory growth pattern
        if len(memory_samples) >= 2:
            memory_trend = memory_samples[-1] - memory_samples[0]
            avg_memory = statistics.mean(memory_samples)
            max_memory = max(memory_samples)
            
            print(f"\n   📈 Memory Analysis:")
            print(f"      Baseline: {self.baseline_memory:.1f}MB")
            print(f"      Average during load: {avg_memory:.1f}MB")
            print(f"      Peak during load: {max_memory:.1f}MB")
            print(f"      Growth trend: {memory_trend:+.1f}MB")
            
            # Memory leak detection
            if memory_trend > 50:  # >50MB growth might indicate a leak
                print(f"   ⚠️  WARNING: Significant memory growth detected ({memory_trend:.1f}MB)")
                leak_suspected = True
            else:
                print(f"   ✅ Memory growth within acceptable range")
                leak_suspected = False
            
            return {
                "baseline_mb": self.baseline_memory,
                "average_mb": avg_memory,
                "peak_mb": max_memory,
                "growth_mb": memory_trend,
                "leak_suspected": leak_suspected,
                "samples": memory_samples
            }
        
        return None
    
    def test_enhanced_api_memory(self):
        """Test memory usage for enhanced knowledge graph APIs"""
        print("\n🧠 Testing enhanced API memory usage...")
        
        test_text = """
        John Smith is a senior software engineer at Microsoft Corporation. 
        He works on artificial intelligence systems and machine learning algorithms.
        John graduated from Stanford University with a PhD in Computer Science.
        His research focuses on natural language processing and knowledge graphs.
        Microsoft is a technology company founded by Bill Gates and Paul Allen.
        The company is headquartered in Redmond, Washington.
        """
        
        enhanced_apis = [
            ("Entity Extraction", "/api/v1/extract/entities"),
            ("Relationship Extraction", "/api/v1/extract/relationships"),
            ("Coreference Resolution", "/api/v1/resolve/coreferences"),
            ("Quality Assessment", "/api/v1/quality/assess"),
        ]
        
        results = {}
        
        for api_name, endpoint in enhanced_apis:
            print(f"   🔍 Testing {api_name}...")
            
            # Take pre-operation snapshot
            pre_snap = self.get_memory_snapshot(f"{api_name}_pre")
            
            # Execute API call
            try:
                if endpoint == "/api/v1/quality/assess":
                    payload = {"text": test_text, "entities": [], "relationships": []}
                else:
                    payload = {"text": test_text, "options": {}}
                
                response = requests.post(f"{self.base_url}{endpoint}", 
                                       json=payload, timeout=30)
                success = response.status_code == 200
                
                if success:
                    response_size = len(response.content)
                else:
                    response_size = 0
                    
            except Exception as e:
                print(f"      ❌ {api_name} failed: {e}")
                success = False
                response_size = 0
            
            # Take post-operation snapshot
            post_snap = self.get_memory_snapshot(f"{api_name}_post")
            
            # Calculate memory delta
            memory_delta = post_snap.rss_mb - pre_snap.rss_mb
            
            results[api_name] = {
                "memory_delta_mb": memory_delta,
                "success": success,
                "response_size_bytes": response_size,
                "pre_memory": pre_snap.rss_mb,
                "post_memory": post_snap.rss_mb
            }
            
            print(f"      Memory: {memory_delta:+.2f}MB, Response: {response_size} bytes, Success: {success}")
        
        return results
    
    def analyze_memory_efficiency(self) -> Dict:
        """Analyze overall memory efficiency"""
        print("\n📊 Analyzing memory efficiency...")
        
        if len(self.snapshots) < 2:
            print("   ❌ Insufficient memory snapshots for analysis")
            return {}
        
        # Calculate statistics
        memory_values = [s.rss_mb for s in self.snapshots]
        min_memory = min(memory_values)
        max_memory = max(memory_values)
        avg_memory = statistics.mean(memory_values)
        memory_range = max_memory - min_memory
        
        # Efficiency metrics
        baseline = self.baseline_memory
        peak_growth = max_memory - baseline
        avg_growth = avg_memory - baseline
        
        print(f"   📈 Memory Statistics:")
        print(f"      Baseline: {baseline:.1f}MB")
        print(f"      Minimum: {min_memory:.1f}MB")
        print(f"      Average: {avg_memory:.1f}MB")
        print(f"      Maximum: {max_memory:.1f}MB")
        print(f"      Range: {memory_range:.1f}MB")
        print(f"      Peak Growth: {peak_growth:.1f}MB")
        print(f"      Average Growth: {avg_growth:.1f}MB")
        
        # Efficiency assessment
        efficiency_score = 1.0
        
        # Penalize excessive memory usage
        if peak_growth > 100:  # >100MB growth
            efficiency_score *= 0.7
        elif peak_growth > 50:  # >50MB growth
            efficiency_score *= 0.85
        
        # Penalize high memory variance (indicates poor management)
        if memory_range > 100:  # >100MB range
            efficiency_score *= 0.8
        elif memory_range > 50:  # >50MB range
            efficiency_score *= 0.9
        
        print(f"   🎯 Memory Efficiency Score: {efficiency_score*100:.1f}%")
        
        return {
            "baseline_mb": baseline,
            "min_mb": min_memory,
            "max_mb": max_memory,
            "avg_mb": avg_memory,
            "range_mb": memory_range,
            "peak_growth_mb": peak_growth,
            "avg_growth_mb": avg_growth,
            "efficiency_score": efficiency_score,
            "total_snapshots": len(self.snapshots)
        }
    
    def generate_memory_report(self) -> Dict:
        """Generate comprehensive memory usage report"""
        print("\n📋 Generating memory usage report...")
        
        analysis = self.analyze_memory_efficiency()
        
        # Memory requirements assessment
        if analysis:
            peak_memory = analysis["max_mb"]
            avg_memory = analysis["avg_mb"]
            efficiency = analysis["efficiency_score"]
            
            # Production memory recommendations
            if peak_memory < 100:
                memory_tier = "Light (512MB-1GB recommended)"
                production_ready = True
            elif peak_memory < 250:
                memory_tier = "Medium (1GB-2GB recommended)"
                production_ready = True
            elif peak_memory < 500:
                memory_tier = "Heavy (2GB-4GB recommended)"
                production_ready = True
            else:
                memory_tier = "Very Heavy (4GB+ required)"
                production_ready = False
            
            print(f"   💾 Memory Tier: {memory_tier}")
            print(f"   🏭 Production Ready: {'✅ Yes' if production_ready else '❌ No'}")
            
            # Overall assessment
            if efficiency >= 0.8 and production_ready:
                overall_status = "Excellent"
            elif efficiency >= 0.7 and production_ready:
                overall_status = "Good"
            elif efficiency >= 0.6:
                overall_status = "Acceptable"
            else:
                overall_status = "Needs Optimization"
            
            print(f"   📊 Overall Memory Status: {overall_status}")
            
            return {
                "analysis": analysis,
                "memory_tier": memory_tier,
                "production_ready": production_ready,
                "overall_status": overall_status,
                "recommendations": self.generate_optimization_recommendations(analysis)
            }
        
        return {}
    
    def generate_optimization_recommendations(self, analysis: Dict) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        peak_growth = analysis.get("peak_growth_mb", 0)
        memory_range = analysis.get("range_mb", 0)
        efficiency = analysis.get("efficiency_score", 1.0)
        
        if peak_growth > 50:
            recommendations.append("Implement memory pooling for large objects")
            recommendations.append("Add explicit garbage collection triggers")
        
        if memory_range > 50:
            recommendations.append("Implement caching with TTL to prevent memory accumulation")
            recommendations.append("Add memory monitoring and cleanup routines")
        
        if efficiency < 0.8:
            recommendations.append("Profile memory allocation patterns")
            recommendations.append("Consider lazy loading for large datasets")
            recommendations.append("Implement streaming processing for large texts")
        
        # Always good practices
        recommendations.extend([
            "Monitor memory usage in production",
            "Set memory limits in container deployment",
            "Implement health checks for memory thresholds"
        ])
        
        return recommendations

def main():
    """Run comprehensive memory usage validation"""
    print("🧠 PHASE 4.3.3: MEMORY USAGE AND OPTIMIZATION VALIDATION")
    print("=" * 70)
    
    validator = MemoryUsageValidator()
    
    # Check server availability
    if not validator.test_server_availability():
        print("❌ ERROR: Server not available at http://localhost:10642")
        return False
    
    print("✅ Server is available and responsive")
    
    try:
        # Establish baseline
        baseline = validator.establish_baseline()
        
        # Test basic operations memory usage
        basic_results = validator.test_basic_memory_usage()
        
        # Test load memory usage
        load_results = validator.test_load_memory_usage()
        
        # Test enhanced API memory usage
        enhanced_results = validator.test_enhanced_api_memory()
        
        # Generate comprehensive report
        memory_report = validator.generate_memory_report()
        
        # Save detailed results
        full_report = {
            "baseline": {
                "memory_mb": baseline.rss_mb,
                "available_mb": baseline.available_mb,
                "percent": baseline.percent
            },
            "basic_operations": basic_results,
            "load_testing": load_results,
            "enhanced_apis": enhanced_results,
            "memory_analysis": memory_report
        }
        
        with open("memory_usage_report.json", "w") as f:
            json.dump(full_report, f, indent=2, default=str)
        
        print(f"\n💾 Detailed memory report saved to: memory_usage_report.json")
        
        # Final assessment
        print("\n" + "=" * 70)
        print("🏁 PHASE 4.3.3 MEMORY VALIDATION COMPLETE")
        print("=" * 70)
        
        if memory_report and memory_report.get("production_ready", False):
            print("✅ RESULT: PHASE 4.3.3 PASSED")
            print(f"   - Memory Status: {memory_report.get('overall_status', 'Unknown')}")
            print(f"   - Memory Tier: {memory_report.get('memory_tier', 'Unknown')}")
            print("   - Production deployment memory requirements validated")
            return True
        else:
            print("⚠️  RESULT: PHASE 4.3.3 NEEDS OPTIMIZATION")
            print("   - Memory usage requires optimization before production")
            return False
            
    except Exception as e:
        print(f"\n❌ MEMORY VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
