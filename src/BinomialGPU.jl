module BinomialGPU

using CUDA
using GPUArrays
using Random

using CUDA: i32

# user-level API
include("rand_binomial.jl")
export rand_binomial!

# CUDA kernels
include("kernels.jl")

end
