# BinomialGPU

[![Build status](https://badge.buildkite.com/70a8c11259658ad6f836a4981791ed144bac80e65302291d0d.svg?branch=master)](https://buildkite.com/julialang/binomialgpu-dot-jl)
[![Coverage](https://codecov.io/gh/simsurace/BinomialGPU.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/simsurace/BinomialGPU.jl)

This package provides a function `rand_binomial!` to produce `CuArrays` with binomially distributed entries, analogous to `CUDA.rand_poisson!` for Poisson-distributed ones.


## Installation

Use the built-in package manager, accessed in Julia via `]`:

```julia
(@v1.6) pkg> add BinomialGPU#experimental_rand
```

in order this branch of this package, you need to also install this branch of CUDA.jl:

```julia
(@v1.6) pkg> add CUDA#tb/speedup_rand
```

If you do not want to do this, use the latest release, which has the same functionality, but is a bit slower:

```julia
(@v1.6) pkg> add BinomialGPU
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
or arbitrary combinations of scalars and arrays of parameters.  


## Issues

The sampler is fast: it is about one order of magnitude faster than other samplers. But it is still an open question whether it can be made faster, whether there are other samplers with competitive speed, and it shows some non-intuitive behavior:
* The speed is faster in Julia 1.5.4 than in the current Julia 1.6 release candidate. See [issue #8](https://github.com/JuliaGPU/BinomialGPU.jl/issues/8).
* The speed is slower when using optimal thread allocation than when defaulting to 256 threads. See [issue #2](https://github.com/JuliaGPU/BinomialGPU.jl/issues/2)
* Are there any other samplers that are comparably fast or faster? I compared the following: sample an array of size `(1024, 1024)` with `count = 128` and `prob` of size `(1024, 1024)` with uniformly drawn entries. Timings on an RTX2070 card: BinomialGPU.jl 0.6ms, PyTorch 11ms, CuPy 18ms, tensorflow 400ms. Please let me know if you know samplers that are not yet listed.