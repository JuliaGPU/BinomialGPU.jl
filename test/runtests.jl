using BinomialGPU
using Test
using CUDA
using BenchmarkTools

@testset "BinomialGPU.jl" begin
    A = CUDA.zeros(Int, 1024, 1024)
    n = 128
    p = 0.5
    ns = CUDA.fill(128, (1024, 1024))
    ps = CUDA.rand(1024, 1024)

    # normal tests
    @test rand_binomial!(A, count = n, prob = p) isa CuArray{Int}
    @test rand_binomial!(A, count = ns, prob = ps) isa CuArray{Int}

    # benchmarks
    println("")
    println("Benchmarking constant parameter array: should run in less than 2ms on an RTX20xx card")
    display(@benchmark CUDA.@sync rand_binomial!($A, count = $n, prob = $p))
    println("")
    println("Benchmarking full parameter array: should run in less than 2ms on an RTX20xx card")
    display(@benchmark CUDA.@sync rand_binomial!($A, count = $ns, prob = $ps))

    println("")
    #tests with wrong dimensions
    A = CUDA.zeros(Int, 16, 16)
    ns = CUDA.fill(10, (16, 16))
    ps = CUDA.rand(16, 16, 2)
    @test_throws DimensionMismatch rand_binomial!(A, count = ns, prob = ps)
    ns = CUDA.fill(10, (16, 16, 16))
    ps = CUDA.rand(16, 16)
    @test_throws DimensionMismatch rand_binomial!(A, count = ns, prob = ps)
    ns = CUDA.fill(10, (16, 15))
    ps = CUDA.rand(16, 16)
    @test_throws DimensionMismatch rand_binomial!(A, count = ns, prob = ps)
    ns = CUDA.fill(10, (16, 16))
    ps = CUDA.rand(16, 15)
    @test_throws DimensionMismatch rand_binomial!(A, count = ns, prob = ps)
end
