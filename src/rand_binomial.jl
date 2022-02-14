## extend the CUDA.jl functionality (rand, randn, rand_poisson, etc.) to include binomial distributions

const BinomialType = Union{Type{<:Integer}}
const BinomialArray = AnyCuArray{<:Integer}


# RNG interface
rand_binomial(rng, T::Type, dims::Dims; kwargs...) =
    rand_binomial!(rng, CuArray{T}(undef, dims); kwargs...)
    
## specify default type
rand_binomial(rng, dims::Dims; kwargs...) =
    rand_binomial(rng, Int, dims; kwargs...)

## support all dimension specifications
rand_binomial(rng, dim1::Integer, dims::Integer...; kwargs...) =
    rand_binomial(rng, Dims((dim1, dims...)); kwargs...)

## ... and with a type
rand_binomial(rng, T::Type, dim1::Integer, dims::Integer...; kwargs...) =
    rand_binomial(rng, T, Dims((dim1, dims...)); kwargs...)


# RNG-less API

## in-place
rand_binomial!(A::BinomialArray; kwargs...) = rand_binomial!(cuda_rng(), A; kwargs...)

rand_binomial!(A::AnyCuArray; kwargs...) =
    error("BinomialGPU.jl does not support generating binomially-distributed random numbers of type $(eltype(A))")

## out-of-place
rand_binomial(T::Type, dims::Dims; kwargs...) = rand_binomial!(CuArray{T}(undef, dims...); kwargs...)

## support all dimension specifications
rand_binomial(T::Type, dim1::Integer, dims::Integer...; kwargs...) =
    rand_binomial!(CuArray{T}(undef, dim1, dims...); kwargs...)

## untyped out-of-place
rand_binomial(dim1::Integer, dims::Integer...; kwargs...) =
    rand_binomial(cuda_rng(), Dims((dim1, dims...)); kwargs...)


# explicit keyword arguments

## without dimension specifications
function rand_binomial(; count, prob)
    countsize = size(count)
    probsize = size(prob)
    if isempty(countsize) && isempty(probsize)
        return rand_binomial(1; count=count, prob=prob)
    elseif length(countsize) > length(probsize)
        return rand_binomial(countsize...; count=count, prob=prob)
    else
        return rand_binomial(probsize...; count=count, prob=prob)
    end
end

function rand_binomial!(rng, A::BinomialArray; count, prob)
    return rand_binom!(rng, A, count, prob)
end


# dispatching on parameter types

## constant (scalar) parameters
function rand_binom!(rng, A::BinomialArray, count::Integer, prob::AbstractFloat)
    n = count

    # edge cases
    if prob <= 0 || n <= 0
        A .= 0
        return A
    elseif prob >= 1
        A .= n
        return A
    end

    invert = prob > 0.5f0
    if invert
        p = 1 - prob
    else
        p = prob
    end

    # Use naive algorithm for n <= 17
    if n <= 17
        kernel = @cuda launch=false kernel_naive_scalar!(
            A, n, Float32(p), rng.seed, rng.counter
        )
    # Use inversion algorithm for n*p < 10
    elseif n * p < 10f0
        kernel = @cuda launch=false kernel_inversion_scalar!(
            A, n, Float32(p), rng.seed, rng.counter
        )
    # BTRS algorithm
    else
        kernel = @cuda launch=false kernel_BTRS_scalar!(
            A, n, Float32(p), rng.seed, rng.counter
        )
    end

    config  = launch_configuration(kernel.fun)
    threads = max(32, min(config.threads, length(A)))
    blocks  = min(config.blocks, cld(length(A), threads))
    kernel(A, n, Float32(p), rng.seed, rng.counter; threads=threads, blocks=blocks)

    new_counter = Int64(rng.counter) + length(A)
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow     # XXX: is this OK?
    rng.counter = remainder

    if invert
        return n .- A
    else
        return A
    end
end

## arrays of parameters
function rand_binom!(rng, A::BinomialArray, count::AbstractArray{<:Integer}, prob::AbstractFloat)
    # revert to full parameter case (this could be suboptimal, as a table-based method should in principle be faster)
    cucount = cu(count)
    ps = CUDA.fill(Float32(prob), size(A))
    return rand_binom!(rng, A, cucount, ps)
end

function rand_binom!(
    rng, A::BinomialArray, count::Integer, prob::AbstractArray{<:AbstractFloat})
    # revert to full parameter case (this could be suboptimal, as a table-based method should in principle be faster)
    ns = CUDA.fill(Int(count), size(A))
    cuprob  = cu(prob)
    return rand_binom!(rng, A, ns, cuprob)
end

function rand_binom!(
    rng, 
    A::BinomialArray, 
    count::AbstractArray{<:Integer}, 
    prob::AbstractArray{<:AbstractFloat}
)
    cucount = cu(count)
    cuprob  = cu(prob)
    return rand_binom!(rng, A, cucount, cuprob)
end

function rand_binom!(rng, A::BinomialArray, count::BinomialArray, prob::DenseCuArray{Float32})
    if ndims(count) > ndims(A) || ndims(prob) > ndims(A)
        throw(DimensionMismatch("`count` and `prob` need to be scalar or have less or equal dimensions than A"))
        return A
    end
    if size(A)[1:ndims(count)] == size(count) && size(A)[1:ndims(prob)] == size(prob)
        count_dim_larger_than_prob_dim = ndims(count) > ndims(prob)
        if count_dim_larger_than_prob_dim
            # indices for prob
            R1 = CartesianIndices(prob)
            # indices for count that are not included in R1
            R2 = CartesianIndices(size(count)[ndims(prob)+1:end])
            # remaining indices in A
            Rr = CartesianIndices(size(A)[ndims(count)+1:end])
        else
            # indices for count
            R1 = CartesianIndices(count)
            # indices for count that are not included in R1
            R2 = CartesianIndices(size(prob)[ndims(count)+1:end])
            # remaining indices in A
            Rr = CartesianIndices(size(A)[ndims(prob)+1:end])
        end
        Rp = CartesianIndices((length(R1), length(R2))) # indices for parameters
        Ra = CartesianIndices((length(Rp), length(Rr))) # indices for parameters and A

        kernel = @cuda launch=false kernel_BTRS!(
            A, 
            count, prob, 
            rng.seed, rng.counter, 
            R1, R2, Rp, Ra, 
            count_dim_larger_than_prob_dim
        )
        config  = launch_configuration(kernel.fun)
        threads = max(32, min(config.threads, length(A)))
        blocks  = min(config.blocks, cld(length(A), threads))
        kernel(
            A, 
            count, prob, 
            rng.seed, rng.counter,
            R1, R2, Rp, Ra, 
            count_dim_larger_than_prob_dim; 
            threads=threads, blocks=blocks
        )

        new_counter = Int64(rng.counter) + length(A)
        overflow, remainder = fldmod(new_counter, typemax(UInt32))
        rng.seed += overflow     # XXX: is this OK?
        rng.counter = remainder
    else
        throw(DimensionMismatch("`count` and `prob` need have size compatible with A"))
    end
    return A
end
