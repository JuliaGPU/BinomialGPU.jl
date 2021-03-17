# BinomialGPU

[![Build Status](https://travis-ci.com/simsurace/BinomialGPU.jl.svg?branch=master)](https://travis-ci.com/simsurace/BinomialGPU.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/simsurace/BinomialGPU.jl?svg=true)](https://ci.appveyor.com/project/simsurace/BinomialGPU-jl)
[![Coverage](https://codecov.io/gh/simsurace/BinomialGPU.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/simsurace/BinomialGPU.jl)
[![Coverage](https://coveralls.io/repos/github/simsurace/BinomialGPU.jl/badge.svg?branch=master)](https://coveralls.io/github/simsurace/BinomialGPU.jl?branch=master)

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
