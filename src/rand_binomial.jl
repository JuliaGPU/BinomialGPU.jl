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


# constant parameters
function rand_binomial!(rng, A::DenseCuArray{Int}; count::Integer, prob::Number)
    begin
        @cuda threads=256 blocks=ceil(Int, length(A)/256) kernel_binomial_const!(A, Int(count), Float32(prob), rng.state)
    end
    return A
end

# arrays of parameters
function rand_binomial!(rng, A::DenseCuArray{Int}; count::AbstractArray{<:Integer}, prob::AbstractArray{<:Number})
    cucount = cu(counts)
    cuprob  = cu(probs)
    return rand_binomial!(rng, A, cucount, cuprob)
end

function rand_binomial!(rng, A::DenseCuArray{Int}; count::DenseCuArray{Int}, prob::DenseCuArray{Float32})
    if size(A) == size(count) == size(prob)
        begin
            @cuda threads=256 blocks=ceil(Int, length(A)/256) kernel_binomial_full!(A, count, prob, rng.state)
        end
    elseif ndims(count) > ndims(A) || ndims(prob) > ndims(A)
        throw(DimensionMismatch("`count` and `prob` need to have smaller number of dimensions than `A`"))
    elseif ndims(A) == 2
        nothing
    elseif ndims(A) == 3
        nothing
    elseif ndims(A) > 3
        nothing
    end
    return A
end
