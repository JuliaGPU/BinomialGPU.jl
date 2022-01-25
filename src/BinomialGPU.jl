module BinomialGPU

using CUDA
using Random

using CUDA: cuda_rng, i32


# user-level API
include("rand_binomial.jl")
export rand_binomial!

# CUDA kernels
include("kernels.jl")

end
