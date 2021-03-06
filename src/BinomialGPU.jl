module BinomialGPU

using CUDA
using Random

# user-level API
include("rand_binomial.jl")
export rand_binomial!

# CUDA kernels
include("kernels.jl")

end
