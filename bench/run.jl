# change if you have a different GPU
using CUDA, KernelAbstractions, Printf, Random
using AMDGPU
using KernelAbstractions: @context
using ArgCheck: @argcheck
using GPUArraysCore: @allowscalar
# const backend = CUDABackend()
const backend = ROCBackend()
include("benchmark.jl")
include("../src/utils.jl")
include("../src/reduce/utilities.jl")
include("../src/reduce/mapreduce_1d_gpu.jl")
include("../src/reduce/mapreduce_nd_v2.jl")

function format_size_factor(n)
    power = 0
    while n % 1024 == 0
        n ÷= 1024
        power += 1
    end
    if power == 0
        return string(n)
    elseif n == 1
        return @sprintf("1024^%d", power)
    else
        return @sprintf("%d * 1024^%d", n, power)
    end
end

# Function to sum array using mapreduce_1d_gpu with pre-allocated temp
function gpu_sum(a, temp, dst_type::Type{U}) where {U}
    return mapreduce_1d_gpu(
        identity, +, a, backend, dst_type;
        init=zero(eltype(a)),
        neutral=zero(eltype(a)),
        max_tasks=1, #not used on GPU
        min_elems=1, #not used on GPU
        block_size=512,
        temp=temp,
        switch_below=0 #switch to cpu below this size
    )
end

sizes = [2^i for i in range(10, 31)]

# Correctness Check
# TODO: implement mixed precision here? 
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
    
    actual = gpu_sum(a, temp, Float32)
    
    if !isapprox(actual, expected; rtol=1e-5, atol=1e-3)
        formatted_size = format_size_factor(size_i)
        println("results do not match for size $formatted_size: gpu $actual vs cpu $expected")
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
        timings[i] = min( benchmark_ms(size_i, gpu_sum, Float32, Float32), timings[i])
    end

    println( "run ");
    for (i,size_i) in enumerate(sizes)
        timings[i] = min( benchmark_ms(size_i, gpu_sum, Float32, Float32), timings[i])
    end

    println( " size_i    time (ms)");
    println(" ------   --------- ");
    for (i,size_i) in enumerate(sizes)
        formatted_size = format_size_factor(size_i)
        @printf " %6s    %8.03f\n" formatted_size timings[i]
    end  
    flush(stdout)
end

println("#########################")
println("running adversarial tests")
println("#########################")

# entire reduction will overflow, but each single block 
println("#########################")
println("int overflow case 2")

for (i,size_i) in enumerate(sizes)
    a = KernelAbstractions.zeros(backend, Float32, size_i)
    promoted_a = KernelAbstractions.zeros(backend, Float64, size_i)

    temp_size = calc_temp_size(size_i)
    temp = KernelAbstractions.zeros(backend, Float32, temp_size)

    host_a = fill(1f0, size_i)

    if size_i > 1024
        host_a[1:2048] .= 2f35 # Maximum julia value is ~3.48f38
    else 
        host_a[1:1024] .= 2f35 # Maximum julia value is ~3.48f38
    end

    copyto!(a, host_a)

    actual = gpu_sum(a, temp, Float32)
    expected = sum(Float32, host_a)
    expected_promoted = sum(Float64, host_a)

    println("cpu result (without cast up): $expected")
    println("cpu result (with cast up): $expected_promoted")
    println("actual $actual")
    
    KernelAbstractions.unsafe_free!(a)
    KernelAbstractions.unsafe_free!(temp)
end


