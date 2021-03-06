# extend the CUDA.jl functionality (rand, randn, rand_poisson, etc.) 
# to include binomial distributions

const BinomialType = Union{Type{<:Integer}}
const BinomialArray = DenseCuArray{<:Integer}

## exported functions: in-place
rand_binomial!(A::AnyCuArray; kwargs...) =
    error("BinomialGPU.jl does not support generating 
    binomially-distributed random numbers of type $(eltype(A))")

## unexported functions: out of place
rand_binomial(T::BinomialType, dims::Dims; kwargs...) = rand_binomial(T, dims; kwargs...)

rand_binomial(T::BinomialType, dim1::Integer, dims::Integer...; kwargs...) =
    rand_binomial(T, Dims((dim1, dims...)); kwargs...)

rand_binomial(T::Type, dims::Dims; kwargs...) = 
    rand_binomial!(CuArray{T}(undef, dims...); kwargs...)

rand_binomial(T::Type, dim1::Integer, dims::Integer...; kwargs...) =
    rand_binomial!(CuArray{T}(undef, dim1, dims...); kwargs...)

rand_binomial(dim1::Integer, dims::Integer...; kwargs...) =
    rand_binomial(Int, Dims((dim1, dims...)); kwargs...)

## main internal function
function rand_binomial!(A::BinomialArray; count, prob)
    return rand_binom!(A, count, prob)
end

## dispatching on parameter types

# constant parameters
function rand_binom!(A::BinomialArray, count::Integer, prob::Number)
    # revert to full parameter case (this could be suboptimal, 
    # as a table-based method should in principle be faster)
    ns = CUDA.fill(Int(count), size(A))
    ps = CUDA.fill(Float32(prob), size(A))
    return rand_binom!(A, ns, ps)
end

# arrays of parameters
function rand_binom!(A::BinomialArray, count::BinomialArray, prob::Number)
    # revert to full parameter case (this could be suboptimal, 
    # as a table-based method should in principle be faster)
    cucount = cu(count)
    ps = CUDA.fill(Float32(prob), size(A))
    return rand_binom!(A, cucount, ps)
end

function rand_binom!(A::BinomialArray, count::Integer, prob::AbstractArray{<:Number})
    # revert to full parameter case (this could be suboptimal, 
    # as a table-based method should in principle be faster)
    ns = CUDA.fill(Int(count), size(A))
    cuprob  = cu(prob)
    return rand_binom!(A, ns, cuprob)
end

function rand_binom!(A::BinomialArray, count::BinomialArray, prob::AbstractArray{<:Number})
    cucount = cu(count)
    cuprob  = cu(prob)
    return rand_binom!(A, cucount, cuprob)
end

function rand_binom!(A::BinomialArray, count::BinomialArray, prob::DenseCuArray{Float32})
    if ndims(count) > ndims(A) || ndims(prob) > ndims(A)
        throw(DimensionMismatch("`count` and `prob` need to be scalar 
                or have less or equal dimensions than A"))
        return A
    end
    if size(A)[1:ndims(count)] == size(count) && size(A)[1:ndims(prob)] == size(prob)
        count_dim_larger_than_prob_dim = ndims(count) > ndims(prob)
        if count_dim_larger_than_prob_dim
            # indices for count
            R1 = CartesianIndices(prob) 
            # indices for prob that are not included in R1
            R2 = CartesianIndices(size(count)[ndims(prob)+1:end]) 
            # remaining indices in A
            Rr = CartesianIndices(size(A)[ndims(count)+1:end]) 
        else
            # indices for count
            R1 = CartesianIndices(count)
            # indices for prob that are not included in R1
            R2 = CartesianIndices(size(prob)[ndims(count)+1:end])
            # remaining indices in A
            Rr = CartesianIndices(size(A)[ndims(prob)+1:end])#
        end
        Rp = CartesianIndices((length(R1), length(R2))) # indices for parameters
        Ra = CartesianIndices((length(Rp), length(Rr))) # indices for parameters and A

        seed = rand(UInt32)

        kernel  = @cuda launch=false kernel_BTRS!(
                                        A, count, prob, 
                                        R1, R2, Rp, Ra, 
                                        count_dim_larger_than_prob_dim,
                                        seed
                                     )
        config  = launch_configuration(kernel.fun)
        threads = Base.min(length(A), config.threads, 256)
        blocks  = cld(length(A), threads)

        kernel(
            A, count, prob, 
            R1, R2, Rp, Ra, 
            count_dim_larger_than_prob_dim,
            seed;
            threads = threads, blocks=blocks
         )
    else
        throw(DimensionMismatch("`count` and `prob` need have size compatible with A"))
    end
    return A
end
