/**
 * @fileoverview TypeScript type definitions for OpenBLAS.wasm
 * High-performance linear algebra library with SIMD optimizations
 */

/**
 * Matrix layout enumeration
 */
export enum BlasLayout {
  RowMajor = 101,
  ColMajor = 102,
}

/**
 * Matrix transpose enumeration
 */
export enum BlasTranspose {
  NoTrans = 111,
  Trans = 112,
  ConjTrans = 113,
}

/**
 * Vector increment type
 */
export type BlasIncrement = number;

/**
 * Matrix dimensions
 */
export interface MatrixDimensions {
  readonly rows: number;
  readonly cols: number;
}

/**
 * BLAS Level 1 operation result
 */
export interface BlasLevel1Result {
  readonly value: number;
  readonly simdUsed: boolean;
  readonly operationsCount: number;
}

/**
 * BLAS Level 2/3 operation result
 */
export interface BlasOperationResult {
  readonly success: boolean;
  readonly simdUsed: boolean;
  readonly operationsCount: number;
  readonly timeMs?: number;
}

/**
 * GEMM operation parameters
 */
export interface GemmParams {
  readonly layout: BlasLayout;
  readonly transA: BlasTranspose;
  readonly transB: BlasTranspose;
  readonly m: number;
  readonly n: number;
  readonly k: number;
  readonly alpha: number;
  readonly beta: number;
}

/**
 * GEMV operation parameters
 */
export interface GemvParams {
  readonly layout: BlasLayout;
  readonly trans: BlasTranspose;
  readonly m: number;
  readonly n: number;
  readonly alpha: number;
  readonly beta: number;
}

/**
 * Vector operation parameters
 */
export interface VectorParams {
  readonly n: number;
  readonly incx?: BlasIncrement;
  readonly incy?: BlasIncrement;
}

/**
 * Performance metrics for benchmarking
 */
export interface PerformanceMetrics {
  readonly operation: string;
  readonly dataSize: number;
  readonly simdEnabled: boolean;
  readonly timeMs: number;
  readonly throughputGFlops: number;
  readonly throughputMBps: number;
}

/**
 * OpenBLAS configuration information
 */
export interface OpenBlasConfig {
  readonly version: string;
  readonly coreName: string;
  readonly simdSupport: boolean;
  readonly maxThreads: number;
  readonly features: string[];
}

/**
 * Memory layout for matrix data
 */
export type MatrixData = Float32Array | Float64Array;

/**
 * Vector data type
 */
export type VectorData = Float32Array | Float64Array;

/**
 * Precision type for operations
 */
export enum BlasPrecision {
  Single = 'single',
  Double = 'double',
}

/**
 * Error types that can occur during BLAS operations
 */
export enum BlasErrorType {
  InvalidDimensions = 'invalid_dimensions',
  InvalidLayout = 'invalid_layout',
  InvalidTranspose = 'invalid_transpose',
  InvalidIncrement = 'invalid_increment',
  MemoryAllocation = 'memory_allocation',
  NotInitialized = 'not_initialized',
  SIMDNotSupported = 'simd_not_supported',
}

/**
 * BLAS operation error
 */
export class BlasError extends Error {
  constructor(
    public readonly type: BlasErrorType,
    message: string,
    public readonly details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'BlasError';
  }
}

/**
 * Options for OpenBLAS initialization
 */
export interface OpenBlasOptions {
  readonly enableSIMD?: boolean;
  readonly maxMemoryMB?: number;
  readonly logLevel?: 'none' | 'error' | 'warn' | 'info' | 'debug';
}

/**
 * BLAS operation statistics
 */
export interface BlasStats {
  readonly totalOperations: number;
  readonly simdOperations: number;
  readonly totalTimeMs: number;
  readonly averageTimeMs: number;
  readonly peakThroughputGFlops: number;
}

/**
 * Matrix multiplication algorithms
 */
export enum GemmAlgorithm {
  /** Standard algorithm */
  Standard = 'standard',
  /** SIMD-optimized algorithm */
  SIMD = 'simd',
  /** Cache-optimized blocking */
  Blocked = 'blocked',
}

/**
 * Advanced GEMM parameters
 */
export interface AdvancedGemmParams extends GemmParams {
  readonly algorithm?: GemmAlgorithm;
  readonly blockSize?: number;
  readonly enableProfiling?: boolean;
}

/**
 * Memory pool configuration
 */
export interface MemoryPoolConfig {
  readonly initialSizeMB: number;
  readonly maxSizeMB: number;
  readonly growthFactor: number;
  readonly enableReuse: boolean;
}

/**
 * Batch operation parameters
 */
export interface BatchParams {
  readonly batchSize: number;
  readonly stride: number;
  readonly enableParallel: boolean;
}

/**
 * Level 1 BLAS operations interface
 */
export interface BlasLevel1 {
  // Dot product
  dot(n: number, x: VectorData, incx: BlasIncrement, y: VectorData, incy: BlasIncrement): BlasLevel1Result;

  // Vector norm
  nrm2(n: number, x: VectorData, incx: BlasIncrement): BlasLevel1Result;

  // Sum of absolute values
  asum(n: number, x: VectorData, incx: BlasIncrement): BlasLevel1Result;

  // Scale vector
  scal(n: number, alpha: number, x: VectorData, incx: BlasIncrement): BlasOperationResult;

  // Vector addition: y = alpha*x + y
  axpy(n: number, alpha: number, x: VectorData, incx: BlasIncrement, y: VectorData, incy: BlasIncrement): BlasOperationResult;

  // Copy vector
  copy(n: number, x: VectorData, incx: BlasIncrement, y: VectorData, incy: BlasIncrement): BlasOperationResult;
}

/**
 * Level 2 BLAS operations interface
 */
export interface BlasLevel2 {
  // Matrix-vector multiplication
  gemv(params: GemvParams, a: MatrixData, lda: number, x: VectorData, incx: BlasIncrement, y: VectorData, incy: BlasIncrement): BlasOperationResult;
}

/**
 * Level 3 BLAS operations interface
 */
export interface BlasLevel3 {
  // General matrix multiplication
  gemm(params: GemmParams, a: MatrixData, lda: number, b: MatrixData, ldb: number, c: MatrixData, ldc: number): BlasOperationResult;

  // Advanced GEMM with algorithm selection
  gemmAdvanced(params: AdvancedGemmParams, a: MatrixData, lda: number, b: MatrixData, ldb: number, c: MatrixData, ldc: number): BlasOperationResult;
}

/**
 * Main OpenBLAS interface
 */
export interface OpenBLAS extends BlasLevel1, BlasLevel2, BlasLevel3 {
  // Initialization and configuration
  initialize(options?: OpenBlasOptions): Promise<void>;
  isInitialized(): boolean;
  getConfig(): OpenBlasConfig;

  // SIMD support
  hasSIMD(): boolean;
  getSIMDFeatures(): string[];

  // Memory management
  allocateMatrix(rows: number, cols: number, precision: BlasPrecision): MatrixData;
  allocateVector(size: number, precision: BlasPrecision): VectorData;
  deallocate(data: MatrixData | VectorData): void;

  // Performance monitoring
  getStats(): BlasStats;
  resetStats(): void;
  enableProfiling(enable: boolean): void;

  // Utility functions
  validateDimensions(m: number, n: number, k?: number): boolean;
  computeGFlops(operation: string, dimensions: MatrixDimensions, timeMs: number): number;

  // Cleanup
  cleanup(): void;
}