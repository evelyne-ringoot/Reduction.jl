#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <curand.h>
#include <cstdlib>
#include <cub/cub.cuh>

// Error checking macro for CUDA calls
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " \
                      << __FILE__ << ":" << __LINE__ << std::endl;             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)


// Error checking macro for cuRAND calls
#define CURAND_CHECK(call)                                                     \
    do {                                                                       \
        curandStatus_t status = call;                                          \
        if (status != CURAND_STATUS_SUCCESS) {                                 \
            std::cerr << "cuRAND error: " << status << " at "                  \
                      << __FILE__ << ":" << __LINE__ << std::endl;             \
            exit(EXIT_FAILURE);                                               \
        }                                                                      \
    } while (0)

//references
//https://github.com/accelerated-computing-class/lab6

constexpr int32_t __host__ __device__ ceil_div_static(int32_t a, int32_t b) { return (a + b - 1) / b; }


template <typename Reset, typename F>
double
benchmark_ms(double target_time_ms, int32_t num_iters_inner, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    int k=0;
    while (elapsed_ms < target_time_ms || k<2) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
        k++;
    }
    return best_time_ms;
}

struct BenchmarkConfig {
    int32_t size_in1;
    int32_t size_in2;
};

enum class Phase {
    TEST,
    WARMUP,
    BENCHMARK,
};

void run_config( Phase phase,
    BenchmarkConfig const &config) {
    auto size_in1 = config.size_in1;
    auto size_in2 = config.size_in2;

    if (phase==Phase::BENCHMARK){
        printf("  %6d x %6d ", size_in1, size_in2);
    }else{
        printf("  %6d x %6d \n", size_in1, size_in2);
    }
 
    curandGenerator_t curandGen;
    CURAND_CHECK(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curandGen, 12345ULL));

    float *a_gpu;
    float *d_sums;
    CUDA_CHECK(cudaMalloc(&a_gpu, size_in1 * size_in2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums, size_in1 * sizeof(float)));

    // Create segment offsets for DeviceSegmentedReduce
    // Each row is a segment: [0, size_in2, 2*size_in2, ..., size_in1*size_in2]
    int *d_offsets;
    CUDA_CHECK(cudaMalloc(&d_offsets, (size_in1 + 1) * sizeof(int)));
    std::vector<int> offsets(size_in1 + 1);
    for (int i = 0; i <= size_in1; ++i) {
        offsets[i] = i * size_in2;
    }
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), (size_in1 + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Determine temporary storage size for segmented reduction
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, 
                                     a_gpu, d_sums, size_in1, 
                                     d_offsets, d_offsets + 1);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    double elapsed_ms = benchmark_ms(
        200.0,
        2,
        [&]() {
            CURAND_CHECK(curandGenerateUniform(curandGen, a_gpu, size_in1 * size_in2)); 
        },
        [&]() {
            cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                            a_gpu, d_sums, size_in1,
                                            d_offsets, d_offsets + 1);
        });

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(d_sums));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_temp_storage));
    CURAND_CHECK(curandDestroyGenerator(curandGen));

    if (phase==Phase::BENCHMARK){
        printf("  %8.03f \n", elapsed_ms);
    }
}

void run_all_configs(
    Phase phase,
    std::vector<BenchmarkConfig> const &configs) {
    if (phase == Phase::WARMUP) {
        printf("warmup\n\n");
    }else {
        printf("\n\n");
        printf(
            "  %-13s  %-9s \n",
            "size_in1 x size_in2",
            "time (ms)");
        printf(
            "  %-13s  %-9s  \n",
            "-----------------",
            "---------");
    }
    for (auto const &config : configs) {
        run_config( phase, config);
    }
    printf("\n");
}



int main(int argc, char **argv) {
    std::vector<BenchmarkConfig> configs_test;
    if (argc==1){
        configs_test = std::vector<BenchmarkConfig>{
            {{32*1024, 1024}, {1024, 1024*32}, {32, 1024*1024}, {1024*1024, 32}, 
            {1024*1024, 1024}, {1024, 1024*1024}, {32, 1024*1024*32}, {1024*1024*32, 32}},
        };
    }else if (argc==3){
        int n1 = std::stoi(argv[1]);
        int n2 = std::stoi(argv[2]);
        configs_test = std::vector<BenchmarkConfig>{
            {n1, n2},
        };
    }else{
        std::cerr << "Usage: " << argv[0] << " [size_in1 size_in2]" << std::endl;
        return 1;
    }

    run_all_configs(Phase::WARMUP,  configs_test);
    run_all_configs(Phase::BENCHMARK, configs_test);

    return 0;
}

