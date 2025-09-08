#!/usr/bin/env node

// Performance validation test for OpenBLAS WASM SIMD implementation
// Tests both SIMD and fallback versions to validate performance improvements

import { OpenBLASLoader } from '../src/openblas_loader.js';

class OpenBLASPerformanceTest {
    constructor() {
        this.results = {
            simd: null,
            fallback: null,
            comparison: null
        };
    }

    // Test specific operation performance
    async testOperation(openblas, operationName, testFunc, iterations = 1000) {
        console.log(`\n🧪 Testing ${operationName} (${iterations} iterations)...`);
        
        // Warmup
        for (let i = 0; i < 10; i++) {
            await testFunc();
        }
        
        // Actual benchmark
        const start = performance.now();
        for (let i = 0; i < iterations; i++) {
            await testFunc();
        }
        const end = performance.now();
        
        const avgTime = (end - start) / iterations;
        const opsPerSec = 1000 / avgTime;
        
        console.log(`   Average time: ${avgTime.toFixed(3)}ms per operation`);
        console.log(`   Operations/sec: ${opsPerSec.toFixed(0)}`);
        
        return { avgTime, opsPerSec, iterations };
    }

    // Test vector dot product performance
    async testDotProduct(openblas, size = 10000) {
        const x = openblas.mallocFloat32(size);
        const y = openblas.mallocFloat32(size);
        
        // Initialize with test data
        for (let i = 0; i < size; i++) {
            x.view[i] = Math.random();
            y.view[i] = Math.random();
        }
        
        const testFunc = () => {
            return openblas.sdot(size, x.ptr, 1, y.ptr, 1);
        };
        
        const result = await this.testOperation(openblas, `SDOT (size: ${size})`, testFunc, 1000);
        
        // Verify result correctness
        const dotResult = testFunc();
        let expectedResult = 0;
        for (let i = 0; i < Math.min(100, size); i++) {
            expectedResult += x.view[i] * y.view[i];
        }
        
        console.log(`   Sample result verification: ${dotResult.toFixed(6)} (first 100 elements expected: ${expectedResult.toFixed(6)})`);
        
        openblas.free(x.ptr);
        openblas.free(y.ptr);
        
        return result;
    }

    // Test matrix multiplication performance
    async testMatrixMultiply(openblas, size = 256) {
        const A = openblas.mallocFloat32(size * size);
        const B = openblas.mallocFloat32(size * size);
        const C = openblas.mallocFloat32(size * size);
        
        // Initialize with test data
        for (let i = 0; i < size * size; i++) {
            A.view[i] = Math.random();
            B.view[i] = Math.random();
            C.view[i] = 0.0;
        }
        
        const testFunc = () => {
            openblas.sgemm('N', 'N', size, size, size, 1.0, A.ptr, size, B.ptr, size, 0.0, C.ptr, size);
        };
        
        const result = await this.testOperation(openblas, `SGEMM (${size}x${size})`, testFunc, 10);
        
        // Calculate GFLOPS
        const operations = 2.0 * size * size * size; // 2n^3 operations for matrix multiply
        const gflops = (operations * result.opsPerSec) / 1e9;
        result.gflops = gflops;
        
        console.log(`   Performance: ${gflops.toFixed(2)} GFLOPS`);
        
        // Verify result correctness (check first element)
        const expectedFirstElement = A.view.slice(0, size).reduce((sum, a, i) => sum + a * B.view[i * size], 0);
        console.log(`   Result verification: C[0,0] = ${C.view[0].toFixed(6)}, expected ≈ ${expectedFirstElement.toFixed(6)}`);
        
        openblas.free(A.ptr);
        openblas.free(B.ptr);
        openblas.free(C.ptr);
        
        return result;
    }

    // Force load specific version for testing
    async loadSpecificVersion(useSIMD) {
        const openblas = new OpenBLASLoader();
        
        // Override SIMD detection for controlled testing
        openblas.hasSIMD = useSIMD;
        openblas.detectSIMDSupport = async () => useSIMD;
        
        console.log(`\n🔧 Force loading ${useSIMD ? 'SIMD' : 'fallback'} version...`);
        await openblas.load();
        
        return openblas;
    }

    // Run comprehensive performance comparison
    async runFullComparison() {
        console.log('🚀 OpenBLAS WASM SIMD Performance Comparison');
        console.log('=' .repeat(60));
        
        try {
            // Test SIMD version
            console.log('\n📊 Testing SIMD Version');
            console.log('-'.repeat(40));
            const simdOpenBLAS = await this.loadSpecificVersion(true);
            
            this.results.simd = {
                dotProduct: await this.testDotProduct(simdOpenBLAS),
                matrixMultiply: await this.testMatrixMultiply(simdOpenBLAS),
                benchmark: await simdOpenBLAS.benchmark(256, 5)
            };
            
            // Test fallback version
            console.log('\n📊 Testing Fallback Version');
            console.log('-'.repeat(40));
            const fallbackOpenBLAS = await this.loadSpecificVersion(false);
            
            this.results.fallback = {
                dotProduct: await this.testDotProduct(fallbackOpenBLAS),
                matrixMultiply: await this.testMatrixMultiply(fallbackOpenBLAS),
                benchmark: await fallbackOpenBLAS.benchmark(256, 5)
            };
            
            // Calculate performance improvements
            this.calculatePerformanceGains();
            
            // Display comparison results
            this.displayResults();
            
            return this.results;
            
        } catch (error) {
            console.error('❌ Performance test failed:', error);
            throw error;
        }
    }

    calculatePerformanceGains() {
        const simd = this.results.simd;
        const fallback = this.results.fallback;
        
        this.results.comparison = {
            dotProduct: {
                speedup: fallback.dotProduct.avgTime / simd.dotProduct.avgTime,
                simdOpsPerSec: simd.dotProduct.opsPerSec,
                fallbackOpsPerSec: fallback.dotProduct.opsPerSec
            },
            matrixMultiply: {
                speedup: fallback.matrixMultiply.avgTime / simd.matrixMultiply.avgTime,
                simdGFLOPS: simd.matrixMultiply.gflops,
                fallbackGFLOPS: fallback.matrixMultiply.gflops
            },
            benchmark: {
                speedup: fallback.benchmark.gflops / simd.benchmark.gflops,
                simdGFLOPS: simd.benchmark.gflops,
                fallbackGFLOPS: fallback.benchmark.gflops
            }
        };
    }

    displayResults() {
        console.log('\n🏆 Performance Comparison Results');
        console.log('=' .repeat(60));
        
        const comp = this.results.comparison;
        
        console.log('\n🔢 Vector Dot Product (SDOT):');
        console.log(`   SIMD:     ${comp.dotProduct.simdOpsPerSec.toFixed(0)} ops/sec`);
        console.log(`   Fallback: ${comp.dotProduct.fallbackOpsPerSec.toFixed(0)} ops/sec`);
        console.log(`   Speedup:  ${comp.dotProduct.speedup.toFixed(2)}x ${this.getSpeedupEmoji(comp.dotProduct.speedup)}`);
        
        console.log('\n🧮 Matrix Multiplication (SGEMM):');
        console.log(`   SIMD:     ${comp.matrixMultiply.simdGFLOPS.toFixed(2)} GFLOPS`);
        console.log(`   Fallback: ${comp.matrixMultiply.fallbackGFLOPS.toFixed(2)} GFLOPS`);
        console.log(`   Speedup:  ${comp.matrixMultiply.speedup.toFixed(2)}x ${this.getSpeedupEmoji(comp.matrixMultiply.speedup)}`);
        
        console.log('\n📈 Built-in Benchmark:');
        console.log(`   SIMD:     ${comp.benchmark.simdGFLOPS.toFixed(2)} GFLOPS`);
        console.log(`   Fallback: ${comp.benchmark.fallbackGFLOPS.toFixed(2)} GFLOPS`);
        console.log(`   Speedup:  ${comp.benchmark.speedup.toFixed(2)}x ${this.getSpeedupEmoji(comp.benchmark.speedup)}`);
        
        // Overall assessment
        const avgSpeedup = (comp.dotProduct.speedup + comp.matrixMultiply.speedup + comp.benchmark.speedup) / 3;
        console.log('\n🎯 Overall Assessment:');
        console.log(`   Average SIMD speedup: ${avgSpeedup.toFixed(2)}x`);
        
        if (avgSpeedup >= 2.0) {
            console.log('   ✅ EXCELLENT: Significant SIMD performance improvements achieved!');
        } else if (avgSpeedup >= 1.5) {
            console.log('   ✅ GOOD: Solid SIMD performance improvements');
        } else if (avgSpeedup >= 1.1) {
            console.log('   ⚠️ MODEST: Some SIMD improvements, consider further optimization');
        } else {
            console.log('   ❌ POOR: Limited SIMD benefits, investigate implementation');
        }
    }

    getSpeedupEmoji(speedup) {
        if (speedup >= 3.0) return '🚀';
        if (speedup >= 2.0) return '⚡';
        if (speedup >= 1.5) return '📈';
        if (speedup >= 1.1) return '👍';
        return '😐';
    }
}

// Run the performance test if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    const test = new OpenBLASPerformanceTest();
    test.runFullComparison()
        .then(results => {
            console.log('\n✅ Performance test completed successfully');
            process.exit(0);
        })
        .catch(error => {
            console.error('❌ Performance test failed:', error);
            process.exit(1);
        });
}

export { OpenBLASPerformanceTest };