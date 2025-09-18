/**
 * @fileoverview OpenBLAS.wasm Demo - Showcasing SIMD-optimized linear algebra
 */

import OpenBLAS, { BlasLayout, BlasTranspose, BlasPrecision } from "./src/lib/index.ts";

async function demonstrateOpenBLAS() {
  console.log("🧮 OpenBLAS.wasm Demo");
  console.log("====================");

  // Initialize OpenBLAS
  const blas = new OpenBLAS();
  await blas.initialize();

  // Show configuration
  const config = blas.getConfig();
  console.log("📊 Configuration:", config);
  console.log();

  // Test basic vector operations
  console.log("🔢 Vector Operations");
  console.log("-------------------");

  // Test dot product with SIMD optimization
  const n = 1000;
  const x = new Float32Array(n);
  const y = new Float32Array(n);

  // Fill with test data
  for (let i = 0; i < n; i++) {
    x[i] = i + 1;
    y[i] = (i + 1) * 2;
  }

  console.time("Dot product (SIMD-optimized)");
  const dotResult = blas.dot(n, x, 1, y, 1);
  console.timeEnd("Dot product (SIMD-optimized)");
  console.log(`Result: ${dotResult.value} (SIMD: ${dotResult.simdUsed})`);

  // Test Euclidean norm
  const normResult = blas.nrm2(n, x, 1);
  console.log(`Euclidean norm: ${normResult.value} (SIMD: ${normResult.simdUsed})`);

  // Test vector scaling
  const scaleCopy = new Float32Array(x);
  blas.scal(10, 2.5, scaleCopy, 1); // Scale first 10 elements by 2.5
  console.log(`Scaled vector [0:10]: [${Array.from(scaleCopy.slice(0, 10)).join(', ')}]`);

  console.log();

  // Test matrix operations
  console.log("🔢 Matrix Operations (GEMM)");
  console.log("---------------------------");

  // Small matrix multiplication for demonstration
  const size = 4;
  const a = new Float32Array(size * size);
  const b = new Float32Array(size * size);
  const c = new Float32Array(size * size);

  // Fill matrices with test data
  for (let i = 0; i < size * size; i++) {
    a[i] = i + 1;
    b[i] = (i + 1) * 0.5;
    c[i] = 0; // Initialize result matrix
  }

  const gemmParams = {
    layout: BlasLayout.RowMajor,
    transA: BlasTranspose.NoTrans,
    transB: BlasTranspose.NoTrans,
    m: size,
    n: size,
    k: size,
    alpha: 1.0,
    beta: 0.0
  };

  console.time("GEMM (Matrix multiplication)");
  const gemmResult = blas.gemm(gemmParams, a, size, b, size, c, size);
  console.timeEnd("GEMM (Matrix multiplication)");
  console.log(`GEMM Success: ${gemmResult.success} (SIMD: ${gemmResult.simdUsed})`);

  // Display result matrix (first row)
  console.log(`Result matrix [row 0]: [${Array.from(c.slice(0, size)).join(', ')}]`);

  console.log();

  // Performance benchmarking
  console.log("⚡ Performance Benchmarks");
  console.log("------------------------");

  const benchmarkSizes = [100, 500, 1000];

  for (const benchSize of benchmarkSizes) {
    const bx = new Float32Array(benchSize);
    const by = new Float32Array(benchSize);

    // Fill with random data
    for (let i = 0; i < benchSize; i++) {
      bx[i] = Math.random();
      by[i] = Math.random();
    }

    // Benchmark dot product
    const iterations = 1000;
    console.time(`Dot product ${benchSize}x${iterations}`);

    let result = 0;
    for (let i = 0; i < iterations; i++) {
      result += blas.dot(benchSize, bx, 1, by, 1).value;
    }

    console.timeEnd(`Dot product ${benchSize}x${iterations}`);
    console.log(`  Average result: ${(result / iterations).toFixed(6)}`);
  }

  // Show performance statistics
  console.log();
  console.log("📈 Performance Statistics");
  console.log("------------------------");
  const stats = blas.getStats();
  console.log("Total operations:", stats.totalOperations);
  console.log("SIMD operations:", stats.simdOperations);
  console.log("SIMD usage:", `${((stats.simdOperations / stats.totalOperations) * 100).toFixed(1)}%`);
  console.log("Average time per operation:", `${stats.averageTimeMs.toFixed(4)}ms`);
  console.log("Peak throughput:", `${stats.peakThroughputGFlops.toFixed(2)} GFlops`);

  // Test memory allocation utilities
  console.log();
  console.log("💾 Memory Allocation");
  console.log("-------------------");

  const matrix32 = blas.allocateMatrix(3, 4, BlasPrecision.Single);
  const vector64 = blas.allocateVector(10, BlasPrecision.Double);

  console.log(`Allocated ${matrix32.constructor.name}[${matrix32.length}] for 3x4 matrix`);
  console.log(`Allocated ${vector64.constructor.name}[${vector64.length}] for 10-element vector`);

  // Clean up
  blas.cleanup();
  console.log();
  console.log("✅ Demo completed successfully!");
}

// Handle both Deno and browser execution
if (typeof Deno !== "undefined") {
  // Deno execution
  demonstrateOpenBLAS().catch(error => {
    console.error("❌ Demo failed:", error);
    Deno.exit(1);
  });
} else {
  // Browser execution
  demonstrateOpenBLAS().catch(error => {
    console.error("❌ Demo failed:", error);
  });
}