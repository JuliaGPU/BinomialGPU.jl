## extend the CUDA.jl functionality (rand, randn, rand_poisson, etc.) to include binomial distributions

gpuarrays_rng() = GPUArrays.default_rng(CuArray)

const BinomialType = Union{Type{Int}}
const BinomialArray = DenseCuArray{Int}

## exported functions: in-place
rand_binomial!(A::BinomialArray; kwargs...) = rand_binomial!(gpuarrays_rng(), A; kwargs...)

rand_binomial!(A::AnyCuArray; kwargs...) =
    error("BinomialGPU.jl does not support generating binomially-distributed random numbers of type $(eltype(A))")

## unexported functions: out of place
rand_binomial(T::BinomialType, dims::Dims; kwargs...) = rand_binomial(gpuarrays_rng(), T, dims; kwargs...)

rand_binomial(T::BinomialType, dim1::Integer, dims::Integer...; kwargs...) =
    rand_binomial(gpuarrays_rng(), T, Dims((dim1, dims...)); kwargs...)

rand_binomial(T::Type, dims::Dims; kwargs...) = rand_binomial!(CuArray{T}(undef, dims...); kwargs...)

rand_binomial(T::Type, dim1::Integer, dims::Integer...; kwargs...) =
    rand_binomial!(CuArray{T}(undef, dim1, dims...); kwargs...)

rand_binomial(dim1::Integer, dims::Integer...; kwargs...) =
    rand_binomial(gpuarrays_rng(), Dims((dim1, dims...)); kwargs...)

## main internal function
function rand_binomial!(rng, A::DenseCuArray{Int}; count, prob)
    return rand_binom!(rng, A, count, prob)
end

## dispatching on parameter types

# constant parameters
function rand_binom!(rng, A::DenseCuArray{Int}, count::Integer, prob::Number)
    # revert to full parameter case (this could be suboptimal, as a table-based method should in principle be faster)
    ns = CUDA.fill(Int(count), size(A))
    ps = CUDA.fill(Float32(prob), size(A))
    return rand_binom!(rng, A, ns, ps)
end

# arrays of parameters
function rand_binom!(rng, A::DenseCuArray{Int}, count::AbstractArray{<:Integer}, prob::AbstractArray{<:Number})
    cucount = cu(count)
    cuprob  = cu(prob)
    return rand_binom!(rng, A, cucount, cuprob)
end

function rand_binom!(rng, A::DenseCuArray{Int}, count::DenseCuArray{Int}, prob::DenseCuArray{Float32})
    if ndims(count) > ndims(A) || ndims(prob) > ndims(A)
        throw(DimensionMismatch("`count` and `prob` need to be scalar or have less or equal dimensions than A"))
        return A
    end
    if size(A)[1:ndims(count)] == size(count) && size(A)[1:ndims(prob)] == size(prob)
        kernel  = @cuda name="BTRS_full" launch=false kernel_BTRS!(A, count, prob, rng.state)
        config  = launch_configuration(kernel.fun)
        threads = Base.min(length(A), config.threads, 256) # strangely seems to be faster when defaulting to 256 threads
        blocks  = cld(length(A), threads)
        kernel(A, count, prob, rng.state; threads=threads, blocks=blocks)
    else
        throw(DimensionMismatch("`count` and `prob` need have size compatible with A"))
    end
    return A
end
