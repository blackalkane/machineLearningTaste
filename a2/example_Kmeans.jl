# Load data
using JLD
X = load("clusterData.jld","X")

# K-means clustering
k = 10
include("kMeans.jl")
model = kMeans(X,k,doPlot=false)
y = model.predict(X)
display("text/plain", kMeansError(X,y,model.W))
include("clustering2Dplot.jl")
clustering2Dplot(X,y,model.W)

using PyPlot
x = 1:10;
y = [122972.69789150477, 45434.98257066315, 9909.52054379199, 3071.4680526538586, 2532.4352089576782, 2061.6215954531017, 1672.4796791066044, 1657.9503069120249, 1290.610167257671, 1175.7140396261957]
plot(x, y)
