using Statistics
using LinearAlgebra
include("misc.jl")

function PCA(X,k)
    (n,d) = size(X)

    # Subtract mean
    mu = mean(X,dims=1)
    X -= repeat(mu,n,1)

    (U,S,V) = svd(X)
    W = V[:,1:k]'
    largest = 0
    largest_index = 0
    for i in 1:85
        if abs(W[1,i]) > largest
            largest = abs(W[1,i])
            largest_index = i
        end
    end
    # @show largest
    # @show largest_index

    largest2 = 0
    largest_index2 = 0
    for i in 1:85
        if abs(W[2,i]) > largest
            largest2 = abs(W[2,i])
            largest_index2 = i
        end
    end
    # @show largest2
    # @show largest_index2

    compress(Xhat) = compressFunc(Xhat,W,mu)
    expand(Z) = expandFunc(Z,W,mu)

    return CompressModel(compress,expand,W)
end

function compressFunc(Xhat,W,mu)
    (t,d) = size(Xhat)
    Xcentered = Xhat - repeat(mu,t,1)
    return Xcentered*W' # Assumes W has orthogonal rows
end

function expandFunc(Z,W,mu)
    (t,k) = size(Z)
    return Z*W + repeat(mu,t,1)
end
