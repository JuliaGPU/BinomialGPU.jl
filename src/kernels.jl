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
function kernel_BTRD_full!(A, count, prob, randstates)
    index1  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride1 = blockDim().x * gridDim().x

    @inbounds for i in index1:stride1:length(A)
        # edge cases
        if prob[i] == 0 || count[i] == 0
            A[i] = 0
            continue
        elseif prob[i] == 1
            A[i] = count[i]
            continue
        end

        # Use naive algorithm for n <= 17
        if count[i] <= 17
            A[i] = 0
            for m in 1:count[i]
                A[i] += GPUArrays.gpu_rand(Float32, CUDA.CuKernelContext(), randstates) < prob[i]
            end
            continue
        end

        # Use inversion algorithm for n*p < 10
        if count[i] * prob[i] < 10f0
            logp = CUDA.log(1-prob[i])
            geom_sum = 0f0
            num_geom = 0
            while true
                geom      = ceil(CUDA.log(GPUArrays.gpu_rand(Float32, CUDA.CuKernelContext(), randstates)) / logp)
                geom_sum += geom
                geom_sum > count[i] && break
                num_geom += 1
            end
            A[i] = num_geom
            continue
        end

        # BTRD algorithm
        # BTRD approximations work well for p <= 0.5
        invert     = prob[i] > 0.5f0
        pp         = invert ? 1-prob[i] : prob[i]

        r          = pp/(1-pp)
        s          = pp*(1-pp)

        A[i]       = -1
        stddev     = sqrt(count[i] * s)
        b          = 1.15f0 + 2.53f0 * stddev
        a          = -0.0873f0 + 0.0248f0 * b + 0.01f0 * pp
        c          = count[i] * pp + 0.5f0
        v_r        = 0.92f0 - 4.2f0 / b

        alpha      = (2.83f0 + 5.1f0 / b) * stddev;
        m          = floor((count[i] + 1) * pp)

        while true
            A[i] >= 0 && break

            usample = GPUArrays.gpu_rand(Float32, CUDA.CuKernelContext(), randstates) - 0.5f0
            vsample = GPUArrays.gpu_rand(Float32, CUDA.CuKernelContext(), randstates)

            us = 0.5f0 - abs(usample)
            ks = floor((2 * a / us + b) * usample + c)

            if us >= 0.07f0 && vsample <= v_r
                A[i] = ks
                continue
            end

            if ks < 0 || ks > count[i]
                continue
            end

            v2 = CUDA.log(vsample * alpha / (a / (us * us) + b))
            ub = (m + 0.5f0) * CUDA.log((m + 1) / (r * (count[i] - m + 1))) +
                 (count[i] + 1) * CUDA.log((count[i] - m + 1) / (count[i] - ks + 1)) +
                 (ks + 0.5f0) * CUDA.log(r * (count[i] - ks + 1) / (ks + 1)) +
                 stirling_approx_tail(m) + stirling_approx_tail(count[i] - m) - stirling_approx_tail(ks) - stirling_approx_tail(count[i] - ks)
            if v2 <= ub
                A[i] = ks
            end
        end
        A[i] = max(0, A[i])

        # if pp = 1 - p[i] was used, undo inversion
        invert && (A[i] = count[i] - A[i])

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
