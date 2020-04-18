using Printf

# Load X and y variable
using JLD
using PyPlot
X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
n = size(X,1)

depth = 15
Xtest = load("citiesSmall.jld","Xtest")
ytest = load("citiesSmall.jld","ytest")
t = size(Xtest,1)

plot_x = 1:15
plot_y_training = []
plot_z_test = []

# check errors from depth 1 to 15 for infoGain-based decision tree
for d in 1:depth
    include("decisionTree_infoGain.jl")
    model = decisionTree_infoGain(X,y,d)

    yhat = model.predict(X)
    trainError = sum(yhat .!= y)/n
    #@printf("Train error with depth-%d decision tree: %.3f\n",d,trainError)
    push!(plot_y_training,trainError)

    yhat_test = model.predict(Xtest)
    testError = sum(yhat_test .!= ytest)/t
    #@printf("Test error with depth-%d decision tree: %.3f\n",d,testError)
    push!(plot_z_test,testError)
end

plot(plot_x,plot_y_training)
plot(plot_x,plot_z_test)
