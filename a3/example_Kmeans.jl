# Load data
using JLD
X = load("clusterData2.jld","X")

# K-means clustering
function GG(a)
    for i in 1:50
        k = 10
        include("kMeans.jl")
        model = kMeans(X,k,doPlot=false)
        y = model.predict(X)
        display("text/plain", kMeansError(X,y,model.W))
    end
    return
end

function kMediansError(X,y,W)
	(n,d) = size(X)

	f = 0
	for i in 1:n
		for j = 1:d
			f += abs.(X[i,j] - W[y[i],j])
		end
	end
	return f
end

# K-medians clustering
function GG2(a)
    for i in 1:50
        k = 10
        include("kMeans.jl")
        model = kMedians(X,k,doPlot=false)
        y = model.predict(X)
        display("text/plain", kMediansError(X,y,model.W))
    end
    return
end
GG2(0)

k = 4
include("kMeans.jl")
model = kMedians(X,k,doPlot=true)
y = model.predict(X)
display("text/plain", kMediansError(X,y,model.W))
include("clustering2Dplot.jl")
clustering2Dplot(X,y,model.W)
#=using PyPlot
x = 1:10;
y = [203003.55519264052, 115999.27902504399, 76454.55428467982, 59116.295802958084, 42203.39335120093, 27211.32876295442, 17275.67342438407, 3071.4680526538586, 2532.4644558837776, 2055.697718510527]
plot(x, y)=#
using PyPlot
x = 1:10;
y = [10279.640065369174, 5303.9416817137935, 2854.140404795244, 2080.8964257454077, 1900.0042756155528, 1792.671735493119, 1639.4646288713884, 1507.7490251404995, 1473.9047578975387, 1399.3308404839886]
plot(x, y)
