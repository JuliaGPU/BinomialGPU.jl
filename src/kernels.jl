## custom samplers for generating binomially distributed CuArrays



## COV_EXCL_START

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


# BTRS algorithm, adapted from the tensorflow library 
# (github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/random_binomial_op.cc)
function kernel_BTRS!(
    A, count, prob, 
    R1, R2, Rp, Ra, 
    count_dim_larger_than_prob_dim, 
    seed::UInt32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds Random.seed!(seed)

    @inbounds if i <= length(A)
        I  = Ra[i]
        Ip = Rp[I[1]]
        I1 = R1[Ip[1]]
        I2 = R2[Ip[2]]

        if count_dim_larger_than_prob_dim
            n = count[CartesianIndex(I1, I2)]
            p = prob[I1]
        else
            n = count[I1]
            p = prob[CartesianIndex(I1, I2)]
        end

        # edge cases
        if p <= 0 || n <= 0
            A[i] = 0
            return
        elseif p >= 1
            A[i] = n
            return
        end
        
        # Use naive algorithm for n <= 17
        if n <= 17
            k = 0
            ctr = 1
            while ctr <= n
                rand(Float32) < p && (k += 1)
                ctr += 1
            end
            A[i] = k
            return
        end

        # Use inversion algorithm for n*p < 10
        if n * p < 10f0
            logp = log(1f0-p)
            geom_sum = 0f0
            num_geom = 0
            while true
                geom      = ceil(log(rand(Float32)) / logp)
                geom_sum += geom
                geom_sum > n && break
                num_geom += 1
            end
            A[i] = num_geom
            return
        end

        # BTRS algorithm
        # BTRS approximations work well for p <= 0.5
        (invert = p > 0.5f0) && (p = 1f0 - p)

        r       = p/(1f0-p)
        s       = p*(1f0-p)

        stddev  = sqrt(n * s)
        b       = 1.15f0 + 2.53f0 * stddev
        a       = -0.0873f0 + 0.0248f0 * b + 0.01f0 * p
        c       = n * p + 0.5f0
        v_r     = 0.92f0 - 4.2f0 / b

        alpha   = (2.83f0 + 5.1f0 / b) * stddev;
        m       = floor((n + 1) * p)

        while true
            usample = rand(Float32) - 0.5f0
            vsample = rand(Float32)

            us = 0.5f0 - abs(usample)
            ks = floor((2 * a / us + b) * usample + c)

            if us >= 0.07f0 && vsample <= v_r
                break
            end

            if ks < 0 || ks > n
                continue
            end

            v2 = log(vsample * alpha / (a / (us * us) + b))
            ub = (m + 0.5f0) * log((m + 1) / (r * (n - m + 1))) +
                 (n + 1) * log((n - m + 1) / (n - ks + 1)) +
                 (ks + 0.5f0) * log(r * (n - ks + 1) / (ks + 1)) +
                 stirling_approx_tail(m) + stirling_approx_tail(n - m) - 
                 stirling_approx_tail(ks) - stirling_approx_tail(n - ks)
            if v2 <= ub
                break
            end
        end

        # if p = 1 - p[i] was used, undo inversion
        invert && (ks = n - ks)
        A[i] = Int(ks);
        nothing
    end#if
    return
end

## COV_EXCL_STOP
