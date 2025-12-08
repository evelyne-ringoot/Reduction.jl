

# Function to calculate temp size needed for reduction
calc_temp_size(len::Int, block_size::Int=256) = 2 * cld(len, 2 * block_size)

function benchmark_ms( size_i::Int, myfunc, eltype::Type)
    a=randn!(KernelAbstractions.zeros(backend,eltype,size_i))
    temp_size = calc_temp_size(size_i)
    temp = KernelAbstractions.zeros(backend, Float32, temp_size)
    elapsed=0.0
    best=10000000000
    i=0
    myfunc(a, temp)
    while(elapsed<1000.0 || (i<2 &&elapsed<5000.0))
        KernelAbstractions.synchronize(backend)
        start = time_ns()
        for i=1:20
            myfunc(a, temp)
        end
        KernelAbstractions.synchronize(backend)
        endtime = time_ns()
        thisduration=(endtime-start)/1e6
        elapsed+=thisduration
        best = min(thisduration/20,best)
        i+=1
    end
    KernelAbstractions.unsafe_free!(a)
    KernelAbstractions.unsafe_free!(temp)
    return best
end

function benchmark_ms_2d(size_in1::Int, size_in2::Int, myfunc, eltype::Type)
    a = randn!(KernelAbstractions.zeros(backend, eltype, size_in1, size_in2))
    temp = KernelAbstractions.zeros(backend, Float32, size_in1, 1)
    elapsed = 0.0
    best = 10000000000
    i = 0
    myfunc(a, temp)
    while(elapsed < 1000.0 || (i < 2 && elapsed < 5000.0))
        KernelAbstractions.synchronize(backend)
        start = time_ns()
        for j = 1:20
            myfunc(a, temp)
        end
        KernelAbstractions.synchronize(backend)
        endtime = time_ns()
        thisduration = (endtime - start) / 1e6
        elapsed += thisduration
        best = min(thisduration / 20, best)
        i += 1
    end
    KernelAbstractions.unsafe_free!(a)
    KernelAbstractions.unsafe_free!(temp)
    return best
end
