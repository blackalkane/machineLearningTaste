using Printf

# Load X and y variable
using JLD
X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
n = size(X,1)
# first half as training set
X_train = X[1:200,:]
y_train = y[1:200,:]
X_validation = X[201:400,:]
y_validation = y[201:400,:]
# second half as training set
X_train = X[201:400,:]
y_train = y[201:400,:]
X_validation = X[1:200,:]
y_validation = y[1:200,:]

# Maximum depth we will plot
maxDepth = 10

include("decisionTree.jl")
for depth in 1:maxDepth
	model = decisionTree(X_train,y_train,depth)

	yhat = model.predict(X_train)
	trainError = sum(yhat .!= y_train)/n
	@printf("Training error with depth-%d accuracy-based decision tree: %.2f\n",depth,trainError)

	yhat_validation = model.predict(X_validation)
	validationError = sum(yhat_validation .!= y_validation)/n
	@printf("Validation error with depth-%d accuracy-based decision tree: %.2f\n",depth,validationError)
end

@printf("Now let's try infogain instead of accuracy...\n")

include("decisionTree_infoGain.jl")
for depth in 1:maxDepth
	model = decisionTree_infoGain(X_train,y_train,depth)

	yhat = model.predict(X_train)
	trainError = sum(yhat .!= y_train)/n
	@printf("Training error with depth-%d infogain-based decision tree: %.2f\n",depth,trainError)

	yhat_validation = model.predict(X_validation)
	validationError = sum(yhat_validation .!= y_validation)/n
	@printf("Validation error with depth-%d accuracy-based decision tree: %.2f\n",depth,validationError)
end
