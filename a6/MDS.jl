include("misc.jl")
include("PCA.jl")
include("findMin.jl")

function MDS(X)
    (n,d) = size(X)

    # Compute all distances
    D = distancesSquared(X,X)
    D = sqrt.(abs.(D))

    # Initialize low-dimensional representation with PCA
    model = PCA(X,2)
    Z = model.compress(X)

    funObj(z) = stress(z,D)

    Z[:] = findMin(funObj,Z[:])

    return Z
end

function stress(z,D)
    n = size(D,1)
    Z = reshape(z,n,2)

    f = 0
    G = zeros(n,2)
    for i in 1:n
        for j in i+1:n
            # Objective function
            Dz = norm(Z[i,:] - Z[j,:])
            s = D[i,j] - Dz
            f = f + (1/2)s^2

            # Gradient
            df = s
            dgi = (Z[i,:] - Z[j,:])/Dz
            dgj = (Z[j,:] - Z[i,:])/Dz
            G[i,:] -= df*dgi
            G[j,:] -= df*dgj
        end
    end
    return (f,G[:])
end

function ISOMAP(X)
    (n,d) = size(X)
    k = 3

    # Compute all distances
    D = distancesSquared(X,X)
    D = sqrt.(abs.(D))

    for i in 1:n
        D[i,i] = Inf
    end

    # get weighted shortest path
    ISO_Graph = fill(Inf,(n,n))
    for i in 1:n
        neighs = sortperm(D[i,:])[1:k]
        for j in neighs
            ISO_Graph[i,j] = D[i,j]
            ISO_Graph[j,i] = D[j,i]
        end
    end

    for i in 1:n
        for j in 1:n
            D[i,j] = dijkstra(ISO_Graph,i,j)
        end
    end

    # Initialize low-dimensional representation with PCA
    model = PCA(X,2)
    Z = model.compress(X)

    funObj(z) = stressISO(z,D)

    Z[:] = findMin(funObj,Z[:])

    return Z
end

function stressISO(z,D)
    n = size(D,1)
    Z = reshape(z,n,2)

    f = 0
    G = zeros(n,2)
    for i in 1:n
        for j in i+1:n
            # Objective function
            Dz = norm(Z[i,:] - Z[j,:])
            s = D[i,j] - Dz
            f = f + (1/2)s^2

            # Gradient
            df = s
            dgi = (Z[i,:] - Z[j,:])/Dz
            dgj = (Z[j,:] - Z[i,:])/Dz
            G[i,:] -= df*dgi
            G[j,:] -= df*dgj
        end
    end
    return (f,G[:])
end

function ISOMAP_Modified(X)
    (n,d) = size(X)
    k = 2

    # Compute all distances
    D = distancesSquared(X,X)
    D = sqrt.(abs.(D))

    for i in 1:n
        D[i,i] = Inf
    end

    # get weighted shortest path
    ISO_Graph = fill(Inf,(n,n))
    for i in 1:n
        neighs = sortperm(D[i,:])[1:k]
        for j in neighs
            ISO_Graph[i,j] = D[i,j]
            ISO_Graph[j,i] = D[j,i]
        end
    end

    for i in 1:n
        for j in 1:n
            D[i,j] = dijkstra(ISO_Graph,i,j)
        end
    end

    maxDis = 0
    for i in 1:n
        for j in 1:n
            if !isinf(D[i,j])
                if D[i,j] > maxDis
                    maxDis = D[i,j]
                end
            end
        end
    end

    for i in 1:n
        for j in 1:n
            if isinf(D[i,j])
                D[i,j] = maxDis
            end
        end
    end

    # Initialize low-dimensional representation with PCA
    model = PCA(X,2)
    Z = model.compress(X)

    funObj(z) = stressISO2(z,D)

    Z[:] = findMin(funObj,Z[:])

    return Z
end

function stressISO2(z,D)
    n = size(D,1)
    Z = reshape(z,n,2)

    f = 0
    G = zeros(n,2)
    for i in 1:n
        for j in i+1:n
            # Objective function
            Dz = norm(Z[i,:] - Z[j,:])
            s = D[i,j] - Dz
            f = f + (1/2)s^2

            # Gradient
            df = s
            dgi = (Z[i,:] - Z[j,:])/Dz
            dgj = (Z[j,:] - Z[i,:])/Dz
            G[i,:] -= df*dgi
            G[j,:] -= df*dgj
        end
    end
    return (f,G[:])
end
