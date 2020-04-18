using Printf
using Statistics

# Load X and y variable
using JLD
dataName = "citiesBig1.jld"
X = load(dataName,"X")
y = load(dataName,"y")
Xtest = load(dataName,"Xtest")
ytest = load(dataName,"ytest")

# Fit a KNN classifier
k = 1
include("knn.jl")
model = cknn(X,y,k)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with %d-nearest neighbours: %.3f\n",k,trainError)
include("plot2Dclassifier.jl")
plot2Dclassifier(X, yhat, model)
# Evaluate test error
yhat = model.predict([-123.116226 49.246292])
@show yhat
testError = mean(yhat .!= ytest)
@printf("Test Error with %d-nearest neighbours: %.3f\n",k,testError)
plot2Dclassifier([-123.116226 49.246292], yhat, model)
