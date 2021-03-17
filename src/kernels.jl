## custom samplers for generating binomially distributed CuArrays


function stirling_approx_tail(k)::Float32
    if k == 0
        return 0.0810614667953272f0
    elseif k == 1
        return 0.0413406959554092f0
    elseif k == 2
        return 0.0276779256849983f0
    elseif k == 3
        return 0.0207906721037650f0
    elseif k == 4
        return 0.0166446911898211f0
    elseif k == 5
        return 0.0138761288230707f0
    elseif k == 6
        return 0.0118967099458917f0
    elseif k == 7
        return 0.0104112652619720f0
    elseif k == 8
        return 0.00925546218271273f0
    elseif k == 9
        return 0.00833056343336287f0
    end
    kp1sq = (k + 1f0)^2;
    return (1.0f0 / 12 - (1.0f0 / 360 - 1.0f0 / 1260 / kp1sq) / kp1sq) / (k + 1)
end


# BTRD algorithm, adapted from the tensorflow library (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/random_binomial_op.cc)
function kernel_BTRD!(A, count, prob, randstates)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    indices = CartesianIndices(A)

    @inbounds if i <= length(A)
        I = indices[i].I
        n = count[CartesianIndex(I[1:ndims(count)])]
        p = prob[CartesianIndex(I[1:ndims(prob)])]

        # edge cases
        if p == 0 || n == 0
            A[i] = 0
            return
        elseif p == 1
            A[i] = n
            return
        end

        # Use naive algorithm for n <= 17
        if n <= 17
            A[i] = 0
            for m in 1:n
                A[i] += GPUArrays.gpu_rand(Float32, CUDA.CuKernelContext(), randstates) < p
            end
            return
        end

        # Use inversion algorithm for n*p < 10
        if n * p < 10f0
            logp = CUDA.log(1-p)
            geom_sum = 0f0
            num_geom = 0
            while true
                geom      = ceil(CUDA.log(GPUArrays.gpu_rand(Float32, CUDA.CuKernelContext(), randstates)) / logp)
                geom_sum += geom
                geom_sum > n && break
                num_geom += 1
            end
            A[i] = num_geom
            return
        end

        # BTRD algorithm
        # BTRD approximations work well for p <= 0.5
        invert     = p > 0.5f0
        pp         = invert ? 1-p : p

        r          = pp/(1-pp)
        s          = pp*(1-pp)

        stddev     = sqrt(n * s)
        b          = 1.15f0 + 2.53f0 * stddev
        a          = -0.0873f0 + 0.0248f0 * b + 0.01f0 * pp
        c          = n * pp + 0.5f0
        v_r        = 0.92f0 - 4.2f0 / b

        alpha      = (2.83f0 + 5.1f0 / b) * stddev;
        m          = floor((n + 1) * pp)

        while true

            usample = GPUArrays.gpu_rand(Float32, CUDA.CuKernelContext(), randstates) - 0.5f0
            vsample = GPUArrays.gpu_rand(Float32, CUDA.CuKernelContext(), randstates)

            us = 0.5f0 - abs(usample)
            ks = floor((2 * a / us + b) * usample + c)

            if us >= 0.07f0 && vsample <= v_r
                A[i] = ks
                return
            end

            if ks < 0 || ks > n
                continue
            end

            v2 = CUDA.log(vsample * alpha / (a / (us * us) + b))
            ub = (m + 0.5f0) * CUDA.log((m + 1) / (r * (n - m + 1))) +
                 (n + 1) * CUDA.log((n - m + 1) / (n - ks + 1)) +
                 (ks + 0.5f0) * CUDA.log(r * (n - ks + 1) / (ks + 1)) +
                 stirling_approx_tail(m) + stirling_approx_tail(n - m) - stirling_approx_tail(ks) - stirling_approx_tail(n - ks)
            if v2 <= ub
                A[i] = ks
                return
            end
        end

        # if pp = 1 - p[i] was used, undo inversion
        invert && (A[i] = n - A[i])
    end
    return
end


## old, unused kernels (for reference)

#naive algorithm, full
function kernel_naive_full!(A, count, prob, randstates)
    index1  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride1 = blockDim().x * gridDim().x

    @inbounds for i in index1:stride1:length(A)
        A[i] = 0
        for m in 1:count[i]
            @inbounds A[i] += GPUArrays.gpu_rand(Float32, CUDA.CuKernelContext(), randstates) < prob[i]
        end
    end
    return
end
