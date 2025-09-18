/**
 * @fileoverview Basic functionality tests for OpenBLAS.wasm
 */

import { assert, assertEquals, assertExists } from "@std/assert";
import OpenBLAS, { BlasLayout, BlasTranspose, BlasPrecision } from "../../src/lib/index.ts";

Deno.test("OpenBLAS initialization", async () => {
  const blas = new OpenBLAS();

  // Should not be initialized initially
  assert(!blas.isInitialized(), "Should not be initialized initially");

  // Initialize the module
  await blas.initialize();

  // Should be initialized now
  assert(blas.isInitialized(), "Should be initialized after initialize()");

  // Check configuration
  const config = blas.getConfig();
  assertExists(config.version, "Version should exist");
  assertExists(config.coreName, "Core name should exist");
  assert(typeof config.simdSupport === "boolean", "SIMD support should be boolean");
  assert(typeof config.maxThreads === "number", "Max threads should be number");
  assert(Array.isArray(config.features), "Features should be array");

  console.log("✓ Configuration:", config);
});

Deno.test("SIMD support detection", async () => {
  const blas = new OpenBLAS();
  await blas.initialize();

  const hasSIMD = blas.hasSIMD();
  assert(typeof hasSIMD === "boolean", "SIMD support should be boolean");

  const features = blas.getSIMDFeatures();
  assert(Array.isArray(features), "SIMD features should be array");

  console.log("✓ SIMD support:", hasSIMD);
  console.log("✓ SIMD features:", features);
});

Deno.test("Vector operations - dot product", async () => {
  const blas = new OpenBLAS();
  await blas.initialize();

  // Test single precision
  const x32 = new Float32Array([1, 2, 3, 4, 5]);
  const y32 = new Float32Array([2, 3, 4, 5, 6]);

  const result32 = blas.dot(5, x32, 1, y32, 1);

  // Expected: 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 2 + 6 + 12 + 20 + 30 = 70
  assertEquals(result32.value, 70, "Single precision dot product should be 70");
  assert(typeof result32.simdUsed === "boolean", "SIMD usage should be reported");
  assertEquals(result32.operationsCount, 10, "Operation count should be 10 (2*n)");

  // Test double precision
  const x64 = new Float64Array([1, 2, 3, 4, 5]);
  const y64 = new Float64Array([2, 3, 4, 5, 6]);

  const result64 = blas.dot(5, x64, 1, y64, 1);

  assertEquals(result64.value, 70, "Double precision dot product should be 70");

  console.log("✓ Dot product results:", { result32, result64 });
});

Deno.test("Vector operations - nrm2 (Euclidean norm)", async () => {
  const blas = new OpenBLAS();
  await blas.initialize();

  // Test with [3, 4] -> should give 5
  const x = new Float32Array([3, 4]);
  const result = blas.nrm2(2, x, 1);

  assertEquals(result.value, 5, "Euclidean norm of [3,4] should be 5");

  console.log("✓ Norm result:", result);
});

Deno.test("Vector operations - scal (scaling)", async () => {
  const blas = new OpenBLAS();
  await blas.initialize();

  const x = new Float32Array([1, 2, 3, 4]);
  const original = new Float32Array(x); // Copy for comparison

  const result = blas.scal(4, 2.5, x, 1);

  assert(result.success, "Scaling operation should succeed");

  // Check that vector was scaled
  for (let i = 0; i < 4; i++) {
    assertEquals(x[i], original[i] * 2.5, `Element ${i} should be scaled by 2.5`);
  }

  console.log("✓ Scaled vector:", Array.from(x));
});

Deno.test("Vector operations - axpy (y = alpha*x + y)", async () => {
  const blas = new OpenBLAS();
  await blas.initialize();

  const x = new Float32Array([1, 2, 3, 4]);
  const y = new Float32Array([2, 4, 6, 8]);
  const originalY = new Float32Array(y); // Copy for comparison

  const result = blas.axpy(4, 3.0, x, 1, y, 1);

  assert(result.success, "AXPY operation should succeed");

  // Check that y = 3*x + y
  for (let i = 0; i < 4; i++) {
    assertEquals(y[i], 3.0 * x[i] + originalY[i], `Element ${i} should be 3*x[i] + originalY[i]`);
  }

  console.log("✓ AXPY result:", Array.from(y));
});

Deno.test("Matrix operations - GEMM (matrix multiplication)", async () => {
  const blas = new OpenBLAS();
  await blas.initialize();

  // Test 2x2 matrix multiplication
  // A = [[1, 2], [3, 4]]
  // B = [[5, 6], [7, 8]]
  // C = A * B = [[19, 22], [43, 50]]

  const a = new Float32Array([1, 2, 3, 4]); // Row-major
  const b = new Float32Array([5, 6, 7, 8]); // Row-major
  const c = new Float32Array([0, 0, 0, 0]); // Result matrix

  const params = {
    layout: BlasLayout.RowMajor,
    transA: BlasTranspose.NoTrans,
    transB: BlasTranspose.NoTrans,
    m: 2,
    n: 2,
    k: 2,
    alpha: 1.0,
    beta: 0.0
  };

  const result = blas.gemm(params, a, 2, b, 2, c, 2);

  assert(result.success, "GEMM operation should succeed");

  // Check results
  const expected = [19, 22, 43, 50];
  for (let i = 0; i < 4; i++) {
    assertEquals(c[i], expected[i], `Element ${i} should be ${expected[i]}`);
  }

  console.log("✓ GEMM result:", Array.from(c));
});

Deno.test("Memory allocation utilities", async () => {
  const blas = new OpenBLAS();
  await blas.initialize();

  // Test matrix allocation
  const matrix32 = blas.allocateMatrix(3, 4, BlasPrecision.Single);
  assert(matrix32 instanceof Float32Array, "Should allocate Float32Array for single precision");
  assertEquals(matrix32.length, 12, "Matrix should have 12 elements (3x4)");

  const matrix64 = blas.allocateMatrix(2, 3, BlasPrecision.Double);
  assert(matrix64 instanceof Float64Array, "Should allocate Float64Array for double precision");
  assertEquals(matrix64.length, 6, "Matrix should have 6 elements (2x3)");

  // Test vector allocation
  const vector32 = blas.allocateVector(5, BlasPrecision.Single);
  assert(vector32 instanceof Float32Array, "Should allocate Float32Array for single precision");
  assertEquals(vector32.length, 5, "Vector should have 5 elements");

  const vector64 = blas.allocateVector(7, BlasPrecision.Double);
  assert(vector64 instanceof Float64Array, "Should allocate Float64Array for double precision");
  assertEquals(vector64.length, 7, "Vector should have 7 elements");

  console.log("✓ Memory allocation successful");
});

Deno.test("Performance statistics", async () => {
  const blas = new OpenBLAS();
  await blas.initialize();

  // Reset stats
  blas.resetStats();
  let stats = blas.getStats();
  assertEquals(stats.totalOperations, 0, "Should start with 0 operations");

  // Perform some operations
  const x = new Float32Array([1, 2, 3, 4, 5]);
  const y = new Float32Array([2, 3, 4, 5, 6]);

  blas.dot(5, x, 1, y, 1);
  blas.nrm2(5, x, 1);
  blas.scal(5, 2.0, x, 1);

  // Check stats
  stats = blas.getStats();
  assertEquals(stats.totalOperations, 3, "Should have 3 total operations");
  assert(stats.totalTimeMs >= 0, "Total time should be non-negative");
  assert(stats.averageTimeMs >= 0, "Average time should be non-negative");

  console.log("✓ Performance stats:", stats);
});

Deno.test("Cleanup resources", async () => {
  const blas = new OpenBLAS();
  await blas.initialize();

  assert(blas.isInitialized(), "Should be initialized");

  blas.cleanup();

  assert(!blas.isInitialized(), "Should not be initialized after cleanup");

  console.log("✓ Cleanup successful");
});