# Code from GPUArrays.jl PR #561: https://github.com/JuliaGPU/GPUArrays.jl/pull/561/files
# KernelAbstractions-based mapreduce implementation for dimensional reductions

import KernelAbstractions: @context
using GPUArraysCore: AnyGPUArray
using ArgCheck: @argcheck

const AbstractArrayOrBroadcasted = Union{AbstractArray, Broadcast.Broadcasted}

@inline function reduce_group(@context, op, val::T, neutral, ::Val{maxitems}) where {T, maxitems}
    items = @groupsize()[1]
    item = @index(Local, Linear)

    # local mem for a complete reduction
    shared = @localmem T (maxitems,)
    @inbounds shared[item] = val

    # perform a reduction
    d = 1
    while d < items
        @synchronize() # legal since cpu=false
        index = 2 * d * (item-1) + 1
        @inbounds if index <= items
            other_val = if index + d <= items
                shared[index+d]
            else
                neutral
            end
            shared[index] = op(shared[index], other_val)
        end
        d *= 2
    end

    # load the final value on the first item
    if item == 1
        val = @inbounds shared[item]
    end

    return val
end

Base.@propagate_inbounds _map_getindex(args::Tuple, I) = ((args[1][I]), _map_getindex(Base.tail(args), I)...)
Base.@propagate_inbounds _map_getindex(args::Tuple{Any}, I) = ((args[1][I]),)
Base.@propagate_inbounds _map_getindex(args::Tuple{}, I) = ()

@kernel cpu=false unsafe_indices=true function partial_mapreduce_device(f, op, neutral, maxitems, Rreduce, Rother, R, As...)
    # decompose the 1D hardware indices into separate ones for reduction (across items
    # and possibly groups if it doesn't fit) and other elements (remaining groups)
    localIdx_reduce = @index(Local, Linear)
    localDim_reduce = @groupsize()[1]
    groupIdx_reduce, groupIdx_other = fldmod1(@index(Group, Linear), length(Rother))
    numGroups = length(KernelAbstractions.blocks(KernelAbstractions.__iterspace(@context())))
    groupDim_reduce = numGroups ÷ length(Rother)

    # group-based indexing into the values outside of the reduction dimension
    # (that means we can safely synchronize items within this group)
    iother = groupIdx_other
    @inbounds if iother <= length(Rother)
        Iother = Rother[iother]

        # load the neutral value
        Iout = CartesianIndex(Tuple(Iother)..., groupIdx_reduce)
        neutral = if neutral === nothing
            R[Iout]
        else
            neutral
        end

        val = op(neutral, neutral)

        # reduce serially across chunks of input vector that don't fit in a group
        ireduce = localIdx_reduce + (groupIdx_reduce - 1) * localDim_reduce
        while ireduce <= length(Rreduce)
            Ireduce = Rreduce[ireduce]
            J = max(Iother, Ireduce)
            val = op(val, f(_map_getindex(As, J)...))
            ireduce += localDim_reduce * groupDim_reduce
        end

        val = reduce_group(@context(), op, val, neutral, maxitems)

        # write back to memory
        if localIdx_reduce == 1
            R[Iout] = val
        end
    end
end

## COV_EXCL_STOP
"""
A: input array to be reduced
R: preallocated output array
"""
function mapreducedim!(f::F, op::OP, R::AnyGPUArray{T}, A::AbstractArrayOrBroadcasted;
                                 init=nothing) where {F, OP, T}
    Base.check_reducedims(R, A)
    length(A) == 0 && return R # isempty(::Broadcasted) iterates

    # add singleton dimensions to the output container, if needed
    if ndims(R) < ndims(A)
        dims = Base.fill_to_length(size(R), 1, Val(ndims(A)))
        R = reshape(R, dims)
    end

    # iteration domain, split in two: one part covers the dimensions that should
    # be reduced, and the other covers the rest. combining both covers all values.
    Rall = CartesianIndices(axes(A))
    Rother = CartesianIndices(axes(R))
    Rreduce = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))
    # NOTE: we hard-code `OneTo` (`first.(axes(A))` would work too) or we get a
    #       CartesianIndices object with UnitRanges that behave badly on the GPU.
    @assert length(Rall) == length(Rother) * length(Rreduce)

    # how many items do we want?
    #
    # items in a group work together to reduce values across the reduction dimensions;
    # we want as many as possible to improve algorithm efficiency and execution occupancy.
    wanted_items = length(Rreduce)
    function compute_items(max_items)
        if wanted_items > max_items
            max_items
        else
            wanted_items
        end
    end

    # how many items can we launch?
    #
    # we might not be able to launch all those items to reduce each slice in one go.
    # that's why each items also loops across their inputs, processing multiple values
    # so that we can span the entire reduction dimension using a single item group.

    # group size is restricted by local memory
    # max_lmem_elements = compute_properties(device()).maxSharedLocalMemory ÷ sizeof(T)
    # max_items = min(compute_properties(device()).maxTotalGroupSize,
    #                 compute_items(max_lmem_elements ÷ 2))
    # TODO: dynamic local memory to avoid two compilations

    # let the driver suggest a group size
    # args = (f, op, init, Val(max_items), Rreduce, Rother, R′, A)
    # kernel_args = kernel_convert.(args)
    # kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
    # kernel = zefunction(partial_mapreduce_device, kernel_tt)
    # reduce_items = launch_configuration(kernel)
    reduce_items = compute_items(512)
    reduce_kernel = partial_mapreduce_device(get_backend(R), (reduce_items,))

    # how many groups should we launch?
    #
    # even though we can always reduce each slice in a single item group, that may not be
    # optimal as it might not saturate the GPU. we already launch some groups to process
    # independent dimensions in parallel; pad that number to ensure full occupancy.
    other_groups = length(Rother)
    reduce_groups = cld(length(Rreduce), reduce_items)

    # determine the launch configuration
    items = reduce_items
    groups = reduce_groups*other_groups

    ndrange = groups*items

    # perform the actual reduction
    if reduce_groups == 1
        # we can cover the dimensions to reduce using a single group
        reduce_kernel(f, op, init, Val(items), Rreduce, Rother, R, A; ndrange)
    else
        # we need multiple steps to cover all values to reduce
        partial = similar(R, (size(R)..., reduce_groups))
        if init === nothing
            # without an explicit initializer we need to copy from the output container
            partial .= R
        end
        reduce_kernel(f, op, init, Val(items), Rreduce, Rother, partial, A; ndrange)

        mapreducedim!(identity, op, R, partial; init=init)  # Recursive call to self
    end

    return R
end
