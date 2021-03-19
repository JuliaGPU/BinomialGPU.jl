using BinomialGPU
using Test
using CUDA
using BenchmarkTools

@testset "BinomialGPU.jl" begin
    @testset "constant parameters" begin
        n = 128
        p = 0.5

        @testset "A of dim $(length(Adims))" for Adims in [[2,], [2, 4], [2, 4, 8]]#, [2, 4, 8, 16]]
            A = CUDA.zeros(Int, Tuple(Adims))
            @test rand_binomial!(A, count = n, prob = p) isa CuArray{Int}
            @test minimum(rand_binomial!(A, count = n, prob = p)) >= 0
            @test maximum(rand_binomial!(A, count = n, prob = p)) <= n
        end
    end

    @testset "parameter arrays" begin
        @testset "A of dim $(length(Adims))" for Adims in [[2,], [2, 4], [2, 4, 8]]#, [2, 4, 8, 16]]
            A = CUDA.zeros(Int, Tuple(Adims))

            @testset "count of dim $i, prob of dim $j" for i in 1:length(Adims), j in 1:length(Adims)
                ndim = Adims[1:i]
                pdim = Adims[1:j]
                ns = CUDA.fill(128, Tuple(ndim))
                ps = CUDA.rand(pdim...)
                @test rand_binomial!(A, count = ns, prob = ps) isa CuArray{Int}
                @test minimum(rand_binomial!(A, count = ns, prob = ps)) >= 0
                @test minimum(ns .- rand_binomial!(A, count = ns, prob = ps)) >= 0

                # wrong size in the last dimension
                for k in 1:i, l in 1:j
                    ndim[k] += 1
                    ns = CUDA.fill(128, Tuple(ndim))
                    ps = CUDA.rand(pdim...)
                    @test_throws DimensionMismatch rand_binomial!(A, count = ns, prob = ps)
                    ndim[k] -= 1
                    pdim[l] += 1
                    ns = CUDA.fill(128, Tuple(ndim))
                    ps = CUDA.rand(pdim...)
                    @test_throws DimensionMismatch rand_binomial!(A, count = ns, prob = ps)
                    pdim[l] -= 1
                end

                # wrong number of dimensions
                if i == length(Adims)
                    push!(ndim, 32)
                    ns = CUDA.fill(128, Tuple(ndim))
                    ps = CUDA.rand(pdim...)
                    @test_throws DimensionMismatch rand_binomial!(A, count = ns, prob = ps)
                    pop!(ndim)
                end
                if j == length(Adims)
                    push!(pdim, 32)
                    ns = CUDA.fill(128, Tuple(ndim))
                    ps = CUDA.rand(pdim...)
                    @test_throws DimensionMismatch rand_binomial!(A, count = ns, prob = ps)
                    pop!(pdim)
                end
            end
        end
    end
    
    @testset "bad parameter values" begin
        # bad parameter values default
        A = CUDA.zeros(Int, 256)
        @test rand_binomial!(A, count = -1, prob = 0.5) == CUDA.zeros(256) # negative counts are equivalent to zero
        @test rand_binomial!(A, count = 2, prob = -0.1) == CUDA.zeros(256) # negative probabilities are equivalent to zero
        @test rand_binomial!(A, count = 2, prob = 1.5) == CUDA.fill(2, 256) # probabilities greater than 1 are equivalent to 1
        @test_throws MethodError rand_binomial!(A, count = 5., prob = 0.5) # non-integer counts throw an error
    end

    @testset "benchmarks" begin
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
end
