# change if you have a different GPU
using CUDA, KernelAbstractions, Printf, Random
using AMDGPU
using KernelAbstractions: @context
using ArgCheck: @argcheck
using GPUArraysCore: @allowscalar
const backend = CUDABackend()
include("benchmark.jl")
include("../src/utils.jl")
include("../src/reduce/utilities.jl")
include("../src/reduce/mapreduce_1d_gpu.jl")
include("../src/reduce/mapreduce_nd_v2.jl")

# Function to sum array using mapreduce_1d_gpu with pre-allocated temp
function gpu_sum(a, temp)
    return mapreduce_1d_gpu(
        identity, +, a, backend;
        init=zero(eltype(a)),
        neutral=zero(eltype(a)),
        max_tasks=1, #not used on GPU
        min_elems=1, #not used on GPU
        block_size=512,
        temp=temp,
        switch_below=0 #switch to cpu below this size
    )
end

function gpu_sum_2d(a, temp)
    mapreducedim!(identity, +, temp, a; init=0.0f0)
    return temp
end

sizes=[1024, 1024*32, 1024*1024, 1024*1024*32, 1024*1024*1024]

# Correctness Check
function mapreduce_1d_serial( 
    f, op, src::AbstractArray;
    init,
    neutral,
)
    result = init
    for i in eachindex(src)
        result = op(result, f(src[i]))
    end
    return result
end

println("testing correctness")
global is_correct = true
for (i,size_i) in enumerate(sizes)
    host_a = randn(Float32, size_i)
    expected = sum(Float32, host_a)
    # run on gpu
    a = KernelAbstractions.zeros(backend, Float32, size_i)
    copyto!(a, host_a)

    temp_size = calc_temp_size(size_i)
    temp = KernelAbstractions.zeros(backend, Float32, temp_size)
    
    actual = gpu_sum(a, temp)
    
    if !isapprox(actual, expected; rtol=1e-5, atol=1e-3)
        println("results do not match for size $size_i: gpu $actual vs cpu $expected")
        global is_correct = false
    end

    KernelAbstractions.unsafe_free!(a)
    KernelAbstractions.unsafe_free!(temp)
end

if is_correct
    println("all tests passed")
    timings=ones(length(sizes))*10000000
    println( "warmup ");

    for (i,size_i) in enumerate(sizes)
        timings[i] = min( benchmark_ms(size_i, gpu_sum, Float32), timings[i])
    end

    println( "run ");
    for (i,size_i) in enumerate(sizes)
        timings[i] = min( benchmark_ms(size_i, gpu_sum, Float32), timings[i])
    end

    println( " size_i    time (ms)");
    println(" ------   --------- ");
    for (i,size_i) in enumerate(sizes)
        @printf " %6d    %8.03f\n" size_i timings[i]
    end  
    flush(stdout)

    # 2d benchmarks
    # sizes_2d = [
    #     (32*1024, 1024),
    #     (1024, 1024*32),
    #     (32, 1024*1024),
    #     (1024*1024, 32)
    # ]
    # timings_2d = ones(length(sizes_2d)) * 10000000
    # println("warmup");
    # for (i, (size_in1, size_in2)) in enumerate(sizes_2d)
    #     timings_2d[i] = min(benchmark_ms_2d(size_in1, size_in2, gpu_sum_2d, Float32), timings_2d[i])
    # end
    # println("run");
    # for (i, (size_in1, size_in2)) in enumerate(sizes_2d)
    #     timings_2d[i] = min(benchmark_ms_2d(size_in1, size_in2, gpu_sum_2d, Float32), timings_2d[i])
    # end
    # println(" size_in1 x size_in2    time (ms)");
    # println(" -------------------   --------- ");
    # for (i, (size_in1, size_in2)) in enumerate(sizes_2d)
    #     @printf " %6d x %6d    %8.03f\n" size_in1 size_in2 timings_2d[i]
    # end
    # flush(stdout)
end
println("#########################")
println("running adversarial tests")
println("#########################")
println("Case 1: Many small values + one large value")
for (i,size_i) in enumerate(sizes)
    a = KernelAbstractions.zeros(backend, Float32, size_i)
    temp_size = calc_temp_size(size_i)
    temp = KernelAbstractions.zeros(backend, Float32, temp_size)

    small = 1f-12
    large = 1f0
    host_a = fill(small, size_i)
    host_a[end] = large
    copyto!(a, host_a)

    actual = gpu_sum(a, temp)
    expected = sum(Float32, host_a)
    println("expected: $expected")
    println("actual $actual")
    
    KernelAbstractions.unsafe_free!(a)
    KernelAbstractions.unsafe_free!(temp)  
end

println("#########################")
println("Case 2: int overflow case")
for (i,size_i) in enumerate(sizes)
    a = KernelAbstractions.zeros(backend, Float32, size_i)
    temp_size = calc_temp_size(size_i)
    temp = KernelAbstractions.zeros(backend, Float32, temp_size)

    host_a = fill(1f0, size_i)
    host_a[end-5:end] .= 1f38

    copyto!(a, host_a)

    actual = gpu_sum(a, temp)
    expected = sum(Float32, host_a)

    println("expected: $expected")
    println("actual $actual")
    
    KernelAbstractions.unsafe_free!(a)
    KernelAbstractions.unsafe_free!(temp)
end

println("#########################")
println("Case 2: int overflow case")
for (i,size_i) in enumerate(sizes)
    a = KernelAbstractions.zeros(backend, Float32, size_i)
    temp_size = calc_temp_size(size_i)
    temp = KernelAbstractions.zeros(backend, Float32, temp_size)

    host_a = fill(1f0, size_i)
    host_a[end-5:end] .= 1f38

    copyto!(a, host_a)

    actual = gpu_sum(a, temp)
    expected = sum(Float32, host_a)

    println("expected: $expected")
    println("actual $actual")
    
    KernelAbstractions.unsafe_free!(a)
    KernelAbstractions.unsafe_free!(temp)
end



