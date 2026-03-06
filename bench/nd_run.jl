# change if you have a different GPU
using CUDA, KernelAbstractions, Printf, Random
using AMDGPU
using KernelAbstractions: @context
using ArgCheck: @argcheck
using GPUArraysCore: @allowscalar
const backend = CUDABackend()
include("nd_benchmark.jl")
include("../src/utils.jl")
include("../src/reduce/utilities.jl")
include("../src/reduce/mapreduce_1d_gpu.jl")
include("../src/reduce/mapreduce_nd_v2.jl")

# Function to sum array using nd mapreduce with pre-allocated temp
function gpu_reduce_nd(a, temp)
    mapreducedim!(identity, +, temp, a; init=0.0f0)
    return temp
end

# sizes=[1024, 1024*32, 1024*1024, 1024*1024*32, 1024*1024*1024]
# reducing over the second dimension 
sizes_2d = [
        (32*1024, 1024),
        (1024, 1024*32),
        (32, 1024*1024),
        (1024*1024, 32)
    ]

sizes_3d = [
    (32*1024, 32, 32), # different locations for large dimension
    (32, 32*1024, 32),
    (32, 32, 32*1024),
    # (1024*32, 1024*32, 32), # sizes are too large
    # (1024*32, 32, 1024*32),
    # (32, 1024*32, 1024*32),
    (256, 256, 256), # cube 
]

# Correctness Check
function mapreduce_2d_serial(
    f, op,
    src::AbstractMatrix;
    init,
)
    size_in1, size_in2 = size(src)
    # output has shape (size_in1, 1)
    result = similar(src, size_in1, 1)

    for i in 1:size_in1
        acc = init
        for j in 1:size_in2
            acc = op(acc, f(src[i, j]))
        end
        result[i, 1] = acc
    end

    return result
end

# Correctness Check
function mapreduce_3d_serial( 
    f, op, src::AbstractArray;
    init,
    dim,
)
    result = init
    for i in eachindex(src)
        result = op(result, f(src[i]))
    end
    return result
end

#########
# 2D tests
#########


# println("testing correctness")
# global is_correct = true
# for (i, (size_in1, size_in2)) in enumerate(sizes_2d)
#     host_a = randn(Float32, size_in1, size_in2)
#     expected = mapreduce_2d_serial(identity, +, host_a; init=0)
#     # run on gpu
#     a = KernelAbstractions.zeros(backend, Float32, size_in1, size_in2)
#     copyto!(a, host_a)

#     temp = KernelAbstractions.zeros(backend, Float32, size_in1, 1)

#     actual = gpu_reduce_nd(a, temp)
#     actual_host = Array(actual)
    
#     if !all(isapprox.(actual_host, expected; rtol=1e-2, atol=1e-2))
#         println("results do not match for size ($size_in1, $size_in2)")
#         global is_correct = false
#     end

#     KernelAbstractions.unsafe_free!(a)
#     KernelAbstractions.unsafe_free!(temp)
# end

# if is_correct
#     # 2d benchmarks
#     timings_2d = ones(length(sizes_2d)) * 10000000
#     println("warmup");
#     for (i, (size_in1, size_in2)) in enumerate(sizes_2d)
#         timings_2d[i] = min(benchmark_ms_2d(size_in1, size_in2, gpu_reduce_nd, Float32), timings_2d[i])
#     end
#     println("run");
#     for (i, (size_in1, size_in2)) in enumerate(sizes_2d)
#         timings_2d[i] = min(benchmark_ms_2d(size_in1, size_in2, gpu_reduce_nd, Float32), timings_2d[i])
#     end
#     println(" size_in1 x size_in2    time (ms)");
#     println(" -------------------   --------- ");
#     for (i, (size_in1, size_in2)) in enumerate(sizes_2d)
#         @printf " %6d x %6d    %8.03f\n" size_in1 size_in2 timings_2d[i]
#     end
#     flush(stdout)
# end

# 3D 

for reduce_dim in 1:3

    timings_3d = ones(length(sizes_3d)) * 1e7
    println("warmup");
    for (i, sizes) in enumerate(sizes_3d)
        timings_3d[i] = min(
            benchmark_ms_3d(sizes, reduce_dim, gpu_reduce_nd, Float32),
            timings_3d[i]
        )
    end

    println("run");
    for (i, sizes) in enumerate(sizes_3d)
        timings_3d[i] = min(
            benchmark_ms_3d(sizes, reduce_dim, gpu_reduce_nd, Float32),
            timings_3d[i]
        )
    end

    println("     size1 x     size2 x     size3   |   reduce_dim=$reduce_dim   |   time (ms)");
    println(" ------------------------------------   --------------------------   ---------");

    for (i, (s1, s2, s3)) in enumerate(sizes_3d)
        @printf(" %8d x %8d x %8d      |        %2d          |  %8.03f\n",
            s1, s2, s3, reduce_dim, timings_3d[i])
    end

    flush(stdout)
end





