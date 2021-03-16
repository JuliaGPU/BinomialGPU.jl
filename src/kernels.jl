## custom samplers for generating binomially distributed CuArrays

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
