using BinomialGPU
using Test
using CUDA
using BenchmarkTools

@testset "BinomialGPU.jl" begin
    n = 128
    p = 0.5

    # normal tests
    for Adims in [[2,], [2, 4], [2, 4, 8]]#, [2, 4, 8, 16]]
        A = CUDA.zeros(Int, Tuple(Adims))

        # tests with constant parameters
        @test rand_binomial!(A, count = n, prob = p) isa CuArray{Int}
        @test minimum(rand_binomial!(A, count = n, prob = p)) >= 0
        @test maximum(rand_binomial!(A, count = n, prob = p)) <= n

        # tests with parameter arrays of all possible sizes
        for i in 1:length(Adims), j in 1:length(Adims)
            ndims = Adims[1:i]
            pdims = Adims[1:j]
            ns = CUDA.fill(128, Tuple(ndims))
            ps = CUDA.rand(pdims...)
            @test rand_binomial!(A, count = ns, prob = ps) isa CuArray{Int}
            @test minimum(rand_binomial!(A, count = ns, prob = ps)) >= 0
            @test minimum(ns .- rand_binomial!(A, count = ns, prob = ps)) >= 0

            # wrong size in last dimension
            for k in 1:i, l in 1:j
                ndims[k] += 1
                ns = CUDA.fill(128, Tuple(ndims))
                ps = CUDA.rand(pdims...)
                @test_throws DimensionMismatch rand_binomial!(A, count = ns, prob = ps)
                ndims[k] -= 1
                pdims[l] += 1
                ns = CUDA.fill(128, Tuple(ndims))
                ps = CUDA.rand(pdims...)
                @test_throws DimensionMismatch rand_binomial!(A, count = ns, prob = ps)
                pdims[l] -= 1
            end

            # wrong number of dimensions
            if i == length(Adims)
                push!(ndims, 32)
                ns = CUDA.fill(128, Tuple(ndims))
                ps = CUDA.rand(pdims...)
                @test_throws DimensionMismatch rand_binomial!(A, count = ns, prob = ps)
                pop!(ndims)
            end
            if j == length(Adims)
                push!(pdims, 32)
                ns = CUDA.fill(128, Tuple(ndims))
                ps = CUDA.rand(pdims...)
                @test_throws DimensionMismatch rand_binomial!(A, count = ns, prob = ps)
                pop!(pdims)
            end
        end
    end


    # benchmarks
    A = CUDA.zeros(Int, 1024, 1024)
    n = 128
    p = 0.5
    ns = CUDA.fill(128, (1024, 1024))
    ps = CUDA.rand(1024, 1024)
    println("")
    println("Benchmarking constant parameter array: should run in less than 2ms on an RTX20xx card")
    display(@benchmark CUDA.@sync rand_binomial!($A, count = $n, prob = $p))
    println("")
    println("Benchmarking full parameter array: should run in less than 2ms on an RTX20xx card")
    display(@benchmark CUDA.@sync rand_binomial!($A, count = $ns, prob = $ps))
    println("")

end
