# BinomialGPU

[![Build status](https://badge.buildkite.com/70a8c11259658ad6f836a4981791ed144bac80e65302291d0d.svg?branch=master)](https://buildkite.com/julialang/binomialgpu-dot-jl)
[![Coverage](https://codecov.io/gh/JuliaGPU/BinomialGPU.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaGPU/BinomialGPU.jl)

This package provides a function `rand_binomial!` to produce `CuArrays` with binomially distributed entries, analogous to `CUDA.rand_poisson!` for Poisson-distributed ones.


## Installation

Use the built-in package manager:

```julia
import Pkg; Pkg.add("BinomialGPU")
```


## Usage

Sample `CuArrays` with binomial random variates in-place:
```julia
using CUDA, BinomialGPU

A = CUDA.zeros(Int, 16)
rand_binomial!(A, count = 10, prob = 0.5)
```
The function currently also supports broadcast over arrays of parameters of the same size as the one to be filled:
```julia
A      = CUDA.zeros(Int, 8)
counts = [1,2,4,8,16,32,64,128]
probs  = CUDA.rand(8)
rand_binomial!(A, count = counts, prob = probs)
```
as well as broadcasts over arrays of parameters whose dimensions are a prefix of the dimensions of A, e.g.
```julia
A      = CUDA.zeros(Int, (2, 4, 8))
counts = rand(1:128, 2, 4)
probs  = CUDA.rand(2)
rand_binomial!(A, count = counts, prob = probs)
```


## Issues

* The speed is slower when using optimal thread allocation than when defaulting to 256 threads. See [issue #2](https://github.com/JuliaGPU/BinomialGPU.jl/issues/2)
* Are there any other samplers that are comparably fast or faster? I compared the following: sample an array of size `(1024, 1024)` with `count = 128` and `prob` of size `(1024, 1024)` with uniformly drawn entries. Timings on an RTX2070 card: BinomialGPU.jl 0.8ms, PyTorch 11ms, CuPy 18ms, tensorflow 400ms. Timings for other samplers are very welcome; please open an issue if you find one.
