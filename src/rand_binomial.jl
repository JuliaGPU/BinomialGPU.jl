# extend the CUDA.jl functionality (rand, randn, rand_poisson, etc.) to include binomial distributions

gpuarrays_rng() = GPUArrays.default_rng(CuArray)

const BinomialType = Union{Type{Int}}
const BinomialArray = DenseCuArray{Int}

rand_binomial!(A::BinomialArray; kwargs...) = CURAND.rand_binomial!(gpuarrays_rng(), A; kwargs...)

rand_binomial(T::BinomialType, dims::Dims; kwargs...) = CURAND.rand_binomial(gpuarrays_rng(), T, dims; kwargs...)

rand_binomial(T::BinomialType, dim1::Integer, dims::Integer...; kwargs...) =
    CURAND.rand_binomial(gpuarrays_rng(), T, Dims((dim1, dims...)); kwargs...)

rand_binomial!(A::AnyCuArray; kwargs...) =
    error("CUDA.jl does not support generating binomially-distributed random numbers of type $(eltype(A))")

rand_binomial(T::Type, dims::Dims; kwargs...) = rand_binomial!(CuArray{T}(undef, dims...); kwargs...)

rand_binomial(T::Type, dim1::Integer, dims::Integer...; kwargs...) =
    rand_binomial!(CuArray{T}(undef, dim1, dims...); kwargs...)

rand_binomial(dim1::Integer, dims::Integer...; kwargs...) =
    CURAND.rand_binomial(gpuarrays_rng(), Dims((dim1, dims...)); kwargs...)
