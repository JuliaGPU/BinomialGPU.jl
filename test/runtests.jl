using BinomialGPU
using CUDA
using Distributions
using Statistics

using BenchmarkTools
using Test

@testset "BinomialGPU.jl" begin
    @testset "in-place" begin
        @testset "scalar parameters" begin
            n = 128
            p = 0.5

            @testset "A of dim $(length(Adims))" for Adims in [[2,], [2, 4], [2, 4, 8], [2, 4, 8, 16]]
                A = CUDA.zeros(Int, Tuple(Adims))
                @test rand_binomial!(A, count = n, prob = p) isa CuArray{Int}
                @test minimum(rand_binomial!(A, count = n, prob = p)) >= 0
                @test maximum(rand_binomial!(A, count = n, prob = p)) <= n
            end
        end

        @testset "parameter arrays" begin
            @testset "A of dim $(length(Adims))" for Adims in [[2,], [2, 4], [2, 4, 8], [2, 4, 8, 16]]
                A = CUDA.zeros(Int, Tuple(Adims))
                @testset "count of dim 0, prob of dim $j" for j in eachindex(Adims)
                    pdim = Adims[1:j]
                    n = 128
                    ps = CUDA.rand(pdim...)
                    @test rand_binomial!(A, count = n, prob = ps) isa CuArray{Int}
                    @test minimum(rand_binomial!(A, count = n, prob = ps)) >= 0
                    @test minimum(n .- rand_binomial!(A, count = n, prob = ps)) >= 0
                end

                @testset "count of dim $i, prob of dim 0" for i in eachindex(Adims)
                    ndim = Adims[1:i]
                    ns = CUDA.fill(128, Tuple(ndim))
                    p = 0.5
                    @test rand_binomial!(A, count = ns, prob = p) isa CuArray{Int}
                    @test minimum(rand_binomial!(A, count = ns, prob = p)) >= 0
                    @test minimum(ns .- rand_binomial!(A, count = ns, prob = p)) >= 0
                end

                @testset "count of dim $i, prob of dim $j" for i in eachindex(Adims), j in eachindex(Adims)
                    ndim = Adims[1:i]
                    pdim = Adims[1:j]
                    ns = CUDA.fill(128, Tuple(ndim))
                    ps = CUDA.rand(pdim...)
                    n = 128
                    p = 0.5
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
    end # in-place

    @testset "out-of-place" begin
        @testset "scalar parameters" begin
            A = rand_binomial(count = 10, prob = 0.5)
            @test size(A) == (1,)
            @test A isa CuVector{Int}

            A = rand_binomial(10, count = 10, prob = 0.5)
            @test size(A) == (10,)
            @test A isa CuVector{Int}

            A = rand_binomial(16, 32, count = 10, prob = 0.5)
            @test size(A) == (16, 32)
            @test A isa CuMatrix{Int}

            A = rand_binomial(2, 16, 32, count = 10, prob = 0.5)
            @test size(A) == (2, 16, 32)
            @test A isa CuArray{Int, 3}
        end
        @testset "parameter arrays" begin
            @testset "array of dim $(length(Adims))" for Adims in [[2,], [2, 4], [2, 4, 8], [2, 4, 8, 16]]
                @testset "count of dim 0, prob of dim $j" for j in eachindex(Adims)
                    pdim = Adims[1:j]
                    n = 128
                    ps = CUDA.rand(pdim...)
                    A = rand_binomial(count = n, prob = ps)
                    @test size(A) == Tuple(pdim)
                end

                @testset "count of dim $i, prob of dim 0" for i in eachindex(Adims)
                    ndim = Adims[1:i]
                    ns = CUDA.fill(128, Tuple(ndim))
                    p = 0.5
                    A = rand_binomial(count = ns, prob = p)
                    @test size(A) == Tuple(ndim)
                end

                @testset "count of dim $i, prob of dim $j" for i in eachindex(Adims), j in eachindex(Adims)
                    ndim = Adims[1:i]
                    pdim = Adims[1:j]
                    ns = CUDA.fill(128, Tuple(ndim))
                    ps = CUDA.rand(pdim...)
                    A = rand_binomial(count = ns, prob = ps)
                    if length(ndim) > length(pdim)
                        @test size(A) == Tuple(ndim)
                    else
                        @test size(A) == Tuple(pdim)
                    end

                    # wrong size in the last dimension
                    for k in 1:i, l in 1:j
                        if i <= j
                            ndim[k] += 1
                            ns = CUDA.fill(128, Tuple(ndim))
                            ps = CUDA.rand(pdim...)
                            @test_throws DimensionMismatch rand_binomial(count = ns, prob = ps)
                            ndim[k] -= 1
                        end
                        if i >= j
                            pdim[l] += 1
                            ns = CUDA.fill(128, Tuple(ndim))
                            ps = CUDA.rand(pdim...)
                            @test_throws DimensionMismatch rand_binomial(count = ns, prob = ps)
                            pdim[l] -= 1
                        end
                    end
                end
            end
        end
    end # out-of-place

    @testset "Distributional tests" begin
        function mean_var_CI(m, S2, n, p, N, α)
            truemean = n*p
            truevar = n*p*(1-p)
            a = quantile(Normal(), α/2)
            b = quantile(Normal(), 1-α/2)
            c = quantile(Chisq(N-1), α/2)
            d = quantile(Chisq(N-1), 1-α/2)
            @test a <= (m - truemean)/sqrt(N*truevar) <= b
            @test c <= (N-1)*S2/truevar <= d
        end
        @testset "Scalar parameters" begin
            function test_mean_variance(N, n, p)
                CUDA.@sync A = rand_binomial(N, count = n, prob = p)
                mean_var_CI(mean(A), var(A), n, p, N, 1e-3)
            end
            N = 2^20
            @testset "n = $n, p = $p" for n in [1, 10, 20, 50, 100, 200, 500, 1000],
                p in 0.1:0.1:0.9
                test_mean_variance(N, n, p)
            end
        end
        @testset "Arrays of parameters" begin
            function test_mean_variance(N, n, p)
                CUDA.@sync A = rand_binomial(N, count = fill(n, N), prob = fill(p, N))
                mean_var_CI(mean(A), var(A), n, p, N, 1e-3)
            end
            N = 2^20
            @testset "n = $n, p = $p" for n in [1, 10, 20, 50, 100, 200, 500, 1000],
                p in 0.1:0.1:0.9
                test_mean_variance(N, n, p)
            end
        end
    end # Distributional tests
end
