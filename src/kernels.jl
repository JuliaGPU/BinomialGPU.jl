## custom samplers for generating binomially distributed CuArrays

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


## kernels

# constant parameters
function kernel_binomial_const!(A, count, prob, randstates)
    index1  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride1 = blockDim().x * gridDim().x

    for i in index1:stride1:length(A)
        A[i] = 0
        for m in 1:count
            @inbounds A[i] += GPUArrays.gpu_rand(Float32, CUDA.CuKernelContext(), randstates) < prob
        end
    end
    return
end

# full parameter arrays (same dimension as A)
function kernel_binomial_full!(A, count, prob, randstates)
    index1  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride1 = blockDim().x * gridDim().x

    for i in index1:stride1:length(A)
        A[i] = 0
        for m in 1:count[i]
            @inbounds A[i] += GPUArrays.gpu_rand(Float32, CUDA.CuKernelContext(), randstates) < prob[i]
        end
    end
    return
end
