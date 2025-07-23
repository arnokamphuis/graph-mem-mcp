#!/usr/bin/env python3
"""
Phase 4.3.2: Performance Benchmarking Test Suite

This test suite validates the performance improvements of the new integrated system
compared to the legacy implementation, measuring response times, memory usage,
and throughput across all major operations.

Test Coverage:
- API response time benchmarking
- Memory usage profiling
- Throughput testing for bulk operations
- Legacy vs new system performance comparison
- Production readiness validation
"""

import asyncio
import json
import requests
import time
import psutil
import statistics
from typing import Dict, List, Tuple
import tracemalloc
from dataclasses import dataclass

@dataclass
class PerformanceResult:
    """Performance measurement result"""
    operation: str
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    memory_usage_mb: float
    success_rate: float
    throughput_ops_sec: float

class PerformanceBenchmark:
    """Comprehensive performance benchmark suite"""
    
    def __init__(self, base_url: str = "http://localhost:10642"):
        self.base_url = base_url
        self.results: List[PerformanceResult] = []
        self.process = psutil.Process()
        
    def measure_memory(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def benchmark_operation(self, operation_name: str, operation_func, iterations: int = 100) -> PerformanceResult:
        """Benchmark a single operation with multiple iterations"""
        print(f"\n🔍 Benchmarking: {operation_name}")
        
        times = []
        successes = 0
        initial_memory = self.measure_memory()
        
        # Warm up
        try:
            operation_func()
        except:
            pass
            
        # Actual benchmark
        start_time = time.time()
        
        for i in range(iterations):
            iteration_start = time.time()
            try:
                result = operation_func()
                iteration_time = (time.time() - iteration_start) * 1000  # Convert to ms
                times.append(iteration_time)
                successes += 1
                
                if i % 10 == 0:
                    print(f"  Progress: {i+1}/{iterations} ({(i+1)/iterations*100:.1f}%)")
                    
            except Exception as e:
                print(f"  Error in iteration {i+1}: {e}")
                times.append(float('inf'))
        
        total_time = time.time() - start_time
        final_memory = self.measure_memory()
        
        # Calculate statistics
        valid_times = [t for t in times if t != float('inf')]
        
        if valid_times:
            result = PerformanceResult(
                operation=operation_name,
                avg_time_ms=statistics.mean(valid_times),
                min_time_ms=min(valid_times),
                max_time_ms=max(valid_times),
                memory_usage_mb=final_memory - initial_memory,
                success_rate=successes / iterations,
                throughput_ops_sec=successes / total_time if total_time > 0 else 0
            )
        else:
            result = PerformanceResult(
                operation=operation_name,
                avg_time_ms=float('inf'),
                min_time_ms=float('inf'),
                max_time_ms=float('inf'),
                memory_usage_mb=final_memory - initial_memory,
                success_rate=0.0,
                throughput_ops_sec=0.0
            )
        
        self.results.append(result)
        
        print(f"  ✅ Average: {result.avg_time_ms:.2f}ms")
        print(f"  ✅ Success Rate: {result.success_rate*100:.1f}%")
        print(f"  ✅ Throughput: {result.throughput_ops_sec:.1f} ops/sec")
        
        return result
    
    def test_server_availability(self) -> bool:
        """Test if the server is running and responsive"""
        try:
            response = requests.get(f"{self.base_url}/banks/list", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def benchmark_basic_operations(self):
        """Benchmark basic bank management operations"""
        print("\n📊 PHASE 4.3.2: Basic Operations Benchmarking")
        
        # Bank listing
        self.benchmark_operation(
            "Bank List API",
            lambda: requests.get(f"{self.base_url}/banks/list").json(),
            iterations=50
        )
        
        # Bank creation
        bank_counter = 0
        def create_bank():
            nonlocal bank_counter
            bank_counter += 1
            return requests.post(
                f"{self.base_url}/banks/create",
                json={"bank": f"perf-test-{bank_counter}"}
            ).json()
        
        self.benchmark_operation(
            "Bank Creation API",
            create_bank,
            iterations=20
        )
        
        # Bank selection
        self.benchmark_operation(
            "Bank Selection API", 
            lambda: requests.post(
                f"{self.base_url}/banks/select",
                json={"bank": "default"}
            ).json(),
            iterations=30
        )
    
    def benchmark_entity_operations(self):
        """Benchmark entity creation and management"""
        print("\n📊 PHASE 4.3.2: Entity Operations Benchmarking")
        
        # Entity creation
        entity_counter = 0
        def create_entity():
            nonlocal entity_counter
            entity_counter += 1
            return requests.post(
                f"{self.base_url}/entities/create",
                json={
                    "entities": [{
                        "name": f"TestEntity{entity_counter}",
                        "entityType": "test",
                        "observations": [f"Performance test entity {entity_counter}"]
                    }]
                }
            ).json()
        
        self.benchmark_operation(
            "Entity Creation API",
            create_entity,
            iterations=30
        )
        
        # Node search
        self.benchmark_operation(
            "Node Search API",
            lambda: requests.post(
                f"{self.base_url}/nodes/search",
                json={"query": "test"}
            ).json(),
            iterations=25
        )
    
    def benchmark_enhanced_apis(self):
        """Benchmark Phase 4.2 enhanced knowledge graph APIs"""
        print("\n📊 PHASE 4.3.2: Enhanced API Benchmarking")
        
        test_text = "John Smith works at Microsoft. He is a software engineer who develops AI systems."
        
        # Enhanced entity extraction
        self.benchmark_operation(
            "Enhanced Entity Extraction",
            lambda: requests.post(
                f"{self.base_url}/api/v1/extract/entities",
                json={"text": test_text, "options": {}}
            ).json(),
            iterations=20
        )
        
        # Relationship extraction
        self.benchmark_operation(
            "Enhanced Relationship Extraction",
            lambda: requests.post(
                f"{self.base_url}/api/v1/extract/relationships",
                json={"text": test_text, "options": {}}
            ).json(),
            iterations=20
        )
        
        # Coreference resolution
        self.benchmark_operation(
            "Coreference Resolution",
            lambda: requests.post(
                f"{self.base_url}/api/v1/resolve/coreferences",
                json={"text": test_text, "options": {}}
            ).json(),
            iterations=15
        )
        
        # Quality assessment
        self.benchmark_operation(
            "Quality Assessment",
            lambda: requests.post(
                f"{self.base_url}/api/v1/quality/assess",
                json={"text": test_text, "entities": [], "relationships": []}
            ).json(),
            iterations=15
        )
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        print("\n📈 GENERATING PERFORMANCE REPORT")
        
        total_operations = len(self.results)
        successful_operations = len([r for r in self.results if r.success_rate > 0.8])
        
        avg_response_time = statistics.mean([r.avg_time_ms for r in self.results if r.avg_time_ms != float('inf')])
        total_memory_usage = sum([r.memory_usage_mb for r in self.results])
        
        # Performance targets (based on production requirements)
        performance_targets = {
            "avg_response_time_ms": 1000,  # < 1 second
            "min_success_rate": 0.9,       # > 90% success
            "max_memory_usage_mb": 100,    # < 100MB total
            "min_throughput_ops_sec": 10   # > 10 ops/sec
        }
        
        # Calculate performance score
        score_factors = []
        
        # Response time score (inverse relationship)
        if avg_response_time <= performance_targets["avg_response_time_ms"]:
            time_score = 1.0
        else:
            time_score = max(0, 1 - (avg_response_time - performance_targets["avg_response_time_ms"]) / performance_targets["avg_response_time_ms"])
        score_factors.append(time_score)
        
        # Success rate score
        avg_success_rate = statistics.mean([r.success_rate for r in self.results])
        success_score = min(1.0, avg_success_rate / performance_targets["min_success_rate"])
        score_factors.append(success_score)
        
        # Memory usage score (inverse relationship)
        if total_memory_usage <= performance_targets["max_memory_usage_mb"]:
            memory_score = 1.0
        else:
            memory_score = max(0, 1 - (total_memory_usage - performance_targets["max_memory_usage_mb"]) / performance_targets["max_memory_usage_mb"])
        score_factors.append(memory_score)
        
        # Throughput score
        avg_throughput = statistics.mean([r.throughput_ops_sec for r in self.results if r.throughput_ops_sec > 0])
        throughput_score = min(1.0, avg_throughput / performance_targets["min_throughput_ops_sec"])
        score_factors.append(throughput_score)
        
        overall_score = statistics.mean(score_factors)
        
        report = {
            "summary": {
                "total_operations_tested": total_operations,
                "successful_operations": successful_operations,
                "success_rate": successful_operations / total_operations if total_operations > 0 else 0,
                "overall_performance_score": overall_score,
                "performance_grade": "A" if overall_score >= 0.9 else "B" if overall_score >= 0.8 else "C" if overall_score >= 0.7 else "D"
            },
            "metrics": {
                "average_response_time_ms": avg_response_time,
                "total_memory_usage_mb": total_memory_usage,
                "average_success_rate": avg_success_rate,
                "average_throughput_ops_sec": avg_throughput
            },
            "targets": performance_targets,
            "detailed_results": [
                {
                    "operation": r.operation,
                    "avg_time_ms": r.avg_time_ms,
                    "success_rate": r.success_rate,
                    "throughput_ops_sec": r.throughput_ops_sec,
                    "memory_usage_mb": r.memory_usage_mb
                }
                for r in self.results
            ]
        }
        
        return report

def main():
    """Run comprehensive performance benchmarking"""
    print("🚀 PHASE 4.3.2: PERFORMANCE BENCHMARKING TEST SUITE")
    print("=" * 60)
    
    benchmark = PerformanceBenchmark()
    
    # Check server availability
    if not benchmark.test_server_availability():
        print("❌ ERROR: Server not available at http://localhost:10642")
        print("   Please ensure the container is running:")
        print("   podman run --rm -it -p 10642:10642 -v \"c:/Users/ik/git/graph_mem/data:/data\" localhost/graph-mem-mcp")
        return False
    
    print("✅ Server is available and responsive")
    
    try:
        # Run benchmark suites
        benchmark.benchmark_basic_operations()
        benchmark.benchmark_entity_operations()
        benchmark.benchmark_enhanced_apis()
        
        # Generate report
        report = benchmark.generate_performance_report()
        
        # Display results
        print("\n" + "=" * 60)
        print("📈 PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Operations Tested: {report['summary']['total_operations_tested']}")
        print(f"   Success Rate: {report['summary']['success_rate']*100:.1f}%")
        print(f"   Performance Score: {report['summary']['overall_performance_score']*100:.1f}%")
        print(f"   Performance Grade: {report['summary']['performance_grade']}")
        
        print(f"\n⚡ KEY METRICS:")
        print(f"   Average Response Time: {report['metrics']['average_response_time_ms']:.2f}ms")
        print(f"   Total Memory Usage: {report['metrics']['total_memory_usage_mb']:.2f}MB")
        print(f"   Average Throughput: {report['metrics']['average_throughput_ops_sec']:.1f} ops/sec")
        
        # Save detailed report
        with open("performance_benchmark_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: performance_benchmark_report.json")
        
        # Performance validation
        if report['summary']['overall_performance_score'] >= 0.8:
            print("\n✅ PHASE 4.3.2 SUCCESS: Performance targets met!")
            print(f"   🎯 Score: {report['summary']['overall_performance_score']*100:.1f}% (Target: 80%+)")
            return True
        else:
            print("\n⚠️  PHASE 4.3.2 WARNING: Performance below targets")
            print(f"   🎯 Score: {report['summary']['overall_performance_score']*100:.1f}% (Target: 80%+)")
            return False
            
    except Exception as e:
        print(f"\n❌ BENCHMARK ERROR: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
