// OpenBLAS WASM Loader with SIMD Feature Detection
// Automatically loads SIMD or fallback version based on browser support

class OpenBLASLoader {
    constructor() {
        this.module = null;
        this.hasSIMD = false;
        this.version = null;
    }

    // Feature detection for WASM SIMD support
    async detectSIMDSupport() {
        try {
            // Test for WebAssembly SIMD support
            const simdTestWasm = new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01,
                0x60, 0x00, 0x01, 0x7b, 0x03, 0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01,
                0x08, 0x00, 0x41, 0x00, 0xfd, 0x0f, 0x0b
            ]);
            
            await WebAssembly.instantiate(simdTestWasm);
            this.hasSIMD = true;
            console.log('🚀 WASM SIMD support detected - loading optimized version');
            return true;
        } catch (e) {
            this.hasSIMD = false;
            console.log('⚠️ WASM SIMD not supported - loading fallback version');
            return false;
        }
    }

    // Load appropriate WASM module
    async load() {
        await this.detectSIMDSupport();
        
        try {
            let moduleConstructor;
            
            if (this.hasSIMD) {
                const { default: OpenBLASSIMD } = await import('./dist/browser-simd/openblas-simd.js');
                moduleConstructor = OpenBLASSIMD;
                this.version = 'simd';
            } else {
                const { default: OpenBLASFallback } = await import('./dist/browser-fallback/openblas-fallback.js');
                moduleConstructor = OpenBLASFallback;
                this.version = 'fallback';
            }
            
            this.module = await moduleConstructor({
                onRuntimeInitialized: () => {
                    console.log(`✅ OpenBLAS WASM loaded (${this.version} version)`);
                    
                    // Verify SIMD functionality
                    const reportedSIMD = this.module.ccall('blas_has_simd', 'number', [], []);
                    const versionString = this.module.ccall('blas_get_version', 'string', [], []);
                    
                    console.log(`📊 SIMD active: ${reportedSIMD}, Version: ${versionString}`);
                    
                    // Get performance thresholds
                    const has_simd_ptr = this.module._malloc(4);
                    const vector_threshold_ptr = this.module._malloc(4);
                    const matrix_threshold_ptr = this.module._malloc(4);
                    
                    this.module.ccall('blas_get_performance_info', null, ['number', 'number', 'number'],
                        [has_simd_ptr, vector_threshold_ptr, matrix_threshold_ptr]);
                    
                    const perfInfo = {
                        hasSIMD: this.module.getValue(has_simd_ptr, 'i32'),
                        vectorThreshold: this.module.getValue(vector_threshold_ptr, 'i32'),
                        matrixThreshold: this.module.getValue(matrix_threshold_ptr, 'i32')
                    };
                    
                    this.module._free(has_simd_ptr);
                    this.module._free(vector_threshold_ptr);
                    this.module._free(matrix_threshold_ptr);
                    
                    console.log('🔧 Performance configuration:', perfInfo);
                }
            });
            
            return this.module;
            
        } catch (error) {
            console.error('❌ Failed to load OpenBLAS WASM:', error);
            throw error;
        }
    }

    // High-level BLAS operations
    dgemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) {
        if (!this.module) throw new Error('OpenBLAS not loaded');
        
        const order = 101; // Row major
        const transA_val = transA === 'T' ? 112 : 111;
        const transB_val = transB === 'T' ? 112 : 111;
        
        return this.module.ccall('blas_dgemm', null, 
            ['number', 'number', 'number', 'number', 'number', 'number', 
             'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number'],
            [order, transA_val, transB_val, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc]);
    }
    
    sgemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) {
        if (!this.module) throw new Error('OpenBLAS not loaded');
        
        const order = 101; // Row major
        const transA_val = transA === 'T' ? 112 : 111;
        const transB_val = transB === 'T' ? 112 : 111;
        
        return this.module.ccall('blas_sgemm', null, 
            ['number', 'number', 'number', 'number', 'number', 'number', 
             'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number'],
            [order, transA_val, transB_val, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc]);
    }
    
    ddot(n, x, incx, y, incy) {
        if (!this.module) throw new Error('OpenBLAS not loaded');
        return this.module.ccall('blas_ddot', 'number', ['number', 'number', 'number', 'number', 'number'],
            [n, x, incx, y, incy]);
    }
    
    sdot(n, x, incx, y, incy) {
        if (!this.module) throw new Error('OpenBLAS not loaded');
        return this.module.ccall('blas_sdot', 'number', ['number', 'number', 'number', 'number', 'number'],
            [n, x, incx, y, incy]);
    }

    // Memory management helpers
    mallocFloat32(count) {
        const ptr = this.module._malloc(count * 4);
        return { ptr, view: new Float32Array(this.module.HEAPF32.buffer, ptr, count) };
    }
    
    mallocFloat64(count) {
        const ptr = this.module._malloc(count * 8);
        return { ptr, view: new Float64Array(this.module.HEAPF64.buffer, ptr, count) };
    }
    
    free(ptr) {
        this.module._free(ptr);
    }

    // Performance benchmarking
    async benchmark(size = 512, iterations = 10) {
        if (!this.module) throw new Error('OpenBLAS not loaded');
        
        console.log(`🏃 Running SGEMM benchmark: ${size}x${size} matrices, ${iterations} iterations`);
        
        const start = performance.now();
        const gflops = this.module.ccall('blas_benchmark_sgemm', 'number', ['number', 'number'],
            [size, iterations]);
        const end = performance.now();
        
        const results = {
            size,
            iterations,
            gflops,
            executionTime: (end - start) / 1000,
            hasSIMD: this.hasSIMD,
            version: this.version
        };
        
        console.log('📈 Benchmark results:', results);
        return results;
    }
}

// Example usage
async function demonstrateOpenBLAS() {
    const openblas = new OpenBLASLoader();
    
    try {
        await openblas.load();
        
        // Run performance benchmark
        const benchResults = await openblas.benchmark(256, 5);
        
        // Example matrix multiplication
        const size = 4;
        const A = openblas.mallocFloat32(size * size);
        const B = openblas.mallocFloat32(size * size);
        const C = openblas.mallocFloat32(size * size);
        
        // Initialize test data
        for (let i = 0; i < size * size; i++) {
            A.view[i] = Math.random();
            B.view[i] = Math.random();
            C.view[i] = 0.0;
        }
        
        // Perform matrix multiplication: C = A * B
        openblas.sgemm('N', 'N', size, size, size, 1.0, A.ptr, size, B.ptr, size, 0.0, C.ptr, size);
        
        console.log('🧮 Matrix multiplication result (first 4 elements):', 
            Array.from(C.view.slice(0, 4)));
        
        // Cleanup
        openblas.free(A.ptr);
        openblas.free(B.ptr);
        openblas.free(C.ptr);
        
        return openblas;
        
    } catch (error) {
        console.error('Demo failed:', error);
        throw error;
    }
}

export { OpenBLASLoader, demonstrateOpenBLAS };