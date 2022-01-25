# BinomialGPU

[![Build status](https://badge.buildkite.com/70a8c11259658ad6f836a4981791ed144bac80e65302291d0d.svg?branch=master)](https://buildkite.com/julialang/binomialgpu-dot-jl)
[![Coverage](https://codecov.io/gh/JuliaGPU/BinomialGPU.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaGPU/BinomialGPU.jl)

This package exports two functions `rand_binomial` and `rand_binomial!` that produce `CuArrays` with binomially distributed elements, analogous to `CUDA.rand_poisson` and `CUDA.rand_poisson!` for Poisson-distributed ones.
The sampling occurs natively on the GPU and is implemented using custom GPU kernels.  

The performance of this implementation seems to be very competitive with other libraries.
Sampling a 1024x1024 matrix on an RTX2070 GPU: BinomialGPU.jl 0.8ms, PyTorch 11ms, CuPy 18ms, tensorflow 400ms. Benchmarking results for other samplers are very welcome; please open an issue if you find one, especially if is faster than this package.


## Installation

In a Julia 1.6 or 1.7 REPL, type `]` to use the built-in package manager and then run:

```julia
pkg> add BinomialGPU
```


## Usage

Sample `CuArrays` with binomial random variates of various dimensions:
```julia
julia> using BinomialGPU
julia> rand_binomial(3, count = 10, prob = 0.5)
3-element CuArray{Int64, 1, CUDA.Mem.DeviceBuffer}:
 4
 3
 7
julia> rand_binomial(4, 4, count = 10, prob = 0.5)
4×4 CuArray{Int64, 2, CUDA.Mem.DeviceBuffer}:
 5  5  6  4
 5  7  6  7
 6  4  4  6
 7  2  4  5
```
The function also supports arrays of parameters of suitable (compatible) sizes:
```julia
julia> counts = [5, 10, 20]
julia> probs = [0.3, 0.4, 0.8]
julia> rand_binomial(count = counts, prob = probs)
3-element CuArray{Int64, 1, CUDA.Mem.DeviceBuffer}:
  0
  7
 19
julia> probs = CUDA.rand(3, 2);
julia> rand_binomial(count = counts, prob = probs)
3×2 CuArray{Int64, 2, CUDA.Mem.DeviceBuffer}:
 3   1
 4   0
 3  18
```
The function with exclamation mark samples random numbers in-place:
```julia
julia> using CUDA
julia> A = CUDA.zeros(Int, 4, 4);
julia> rand_binomial!(A, count = 10, prob = 0.5)
4×4 CuArray{Int64, 2, CUDA.Mem.DeviceBuffer}:
 6  4  1  8
 4  6  6  6
 4  3  2  4
 5  7  3  5
```
This also allows for non-standard types to be preserved:
```julia
julia> A = CUDA.zeros(UInt16, 4, 4);
julia> rand_binomial!(A, count = 10, prob = 0.5)
4×4 CuArray{UInt16, 2, CUDA.Mem.DeviceBuffer}:
 0x0005  0x0004  0x0003  0x0005
 0x0006  0x0006  0x0006  0x0003
 0x0006  0x0005  0x0006  0x0005
 0x0007  0x0005  0x0006  0x0006
```
Alternatively, pass the desired type as the first argument:
```julia
julia> rand_binomial(UInt32, 4, 4, count = 10, prob = 0.5)
4×4 CuArray{UInt32, 2, CUDA.Mem.DeviceBuffer}:
 0x00000004  0x00000005  0x00000008  0x00000005
 0x00000003  0x00000007  0x00000005  0x00000005
 0x00000007  0x00000005  0x00000005  0x00000004
 0x00000001  0x00000005  0x00000005  0x00000003
```
