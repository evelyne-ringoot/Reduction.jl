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
            exit(EXIT_FAILURE);                                                \
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
    int32_t size_in;
};

struct TestData {
    std::map<int32_t, float*> input;
    std::map<int32_t, float*> sum_result;
};



enum class Phase {
    TEST,
    WARMUP,
    BENCHMARK,
};

void run_config( Phase phase,
    BenchmarkConfig const &config) {
    auto size_in = config.size_in;

    if (phase==Phase::BENCHMARK){
        printf("  %6d ", size_in);
    }else{
        printf("  %6d \n", size_in);
    }
 
    curandGenerator_t curandGen;
    CURAND_CHECK(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curandGen, 12345ULL));

    float *a_gpu;
    float *d_sum;
    CUDA_CHECK(cudaMalloc(&a_gpu, size_in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));

    // Determine temporary storage size
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, a_gpu, d_sum, size_in);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    double elapsed_ms = benchmark_ms(
        200.0,
        2,
        [&]() {
            CURAND_CHECK(curandGenerateUniform(curandGen, a_gpu, size_in)); 
        },
        [&]() {
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, a_gpu, d_sum, size_in);
        });

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(d_sum));
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
            "  %-6s  %-9s \n",
            "size_i",
            "time (ms)");
        printf(
            "  %-6s  %-9s  \n",
            "------",
            "---------");
    }
    for (auto const &config : configs) {
        run_config( phase, config);
    }
    printf("\n");
}



int main(int argc, char **argv) {
    std::string test_data_dir = ".";
    std::vector<BenchmarkConfig> configs_test;
    if (argc==1){
        configs_test = std::vector<BenchmarkConfig>{
            {{1024},{1024*32},{1024*1024},{1024*1024*32},{1024*1024*1024}},
        };
    }else{
        int n = std::stoi(argv[1]);
        configs_test = std::vector<BenchmarkConfig>{
            {n},
        };
    }

    run_all_configs(Phase::WARMUP,  configs_test);
    run_all_configs(Phase::BENCHMARK, configs_test);

    return 0;
}