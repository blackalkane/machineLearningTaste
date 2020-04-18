using DelimitedFiles

# Load data
dataTable = readdlm("animals.csv",',')
X = float(real(dataTable[2:end,2:end]))
(n,d) = size(X)
@show dataTable[1,13]
@show dataTable[1,59]
# Standardize columns
include("misc.jl")
(X,mu,sigma) = standardizeCols(X)

# Plot matrix as image
using PyPlot
figure(1)
clf()
imshow(X)

include("PCA.jl")
for k in 2:5
	model = PCA(X,k)
	Z = model.compress(X)
	gg = model.expand(Z)
	variance = 1 - (norm(gg.-X)^2/norm(X)^2)
	@show k
	@show variance
end

k=2
# get PCA fit model
model = PCA(X,k)
# get Z
Z = model.compress(X)

figure(2)
clf()
plot(Z[:,1],Z[:,2],".")
for i in 1:n
    annotate(dataTable[i+1,1],
	xy=[Z[i,1],Z[i,2]],
	xycoords="data")
end


gg = model.expand(Z)
#@show size(gg)
variance = 1 - (norm(gg.-X)^2/norm(X)^2)
#@show variance
