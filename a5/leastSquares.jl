using LinearAlgebra
include("misc.jl")

function leastSquares(X,y)

	# Find regression weights minimizing squared error
	w = (X'*X)\(X'*y)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return LinearModel(predict,w)
end

# function leastSquaresBiasL2(X,y,lambda)
#
# 	# Add bias column
# 	n = size(X,1)
# 	Z = [ones(n,1) X]
#
# 	# Find regression weights minimizing squared error
# 	v = (Z'*Z + lambda*I)\(Z'*y)
#
# 	# Make linear prediction function
# 	predict(Xhat) = [ones(size(Xhat,1),1) Xhat]*v
#
# 	# Return model
# 	return LinearModel(predict,v)
# end

function leastSquaresBiasL2(X,y,lambda)

	# Add bias column
	n = size(X,1)
	Z = [ones(n,1) X]

	# Find regression weights minimizing squared error
	u = (Z*Z' + lambda*I)\y

	# Make linear prediction function
	predict(Xhat) = ([ones(size(Xhat,1),1) Xhat]*Z')*u

	# Return model
	return LinearModel(predict,u)
end

function polyKernel(x1,x2,p)
	return (1 .+ x1*x2').^p
end

function leastSquaresKernelBasis(x,y,lambda,p)
	K = polyKernel(x,x,p)

	u = (K + lambda*I)\y

	predict(Xhat) = polyKernel(Xhat,x,p)*u

	return LinearModel(predict,u)
end

function rbfKernel(x1,x2,sigma)
	D = distancesSquared(x1,x2)
	return exp.(-D/(2*sigma^2))
end

function leastSquaresKernelRBF(x,y,lambda,sigma)
	K = rbfKernel(x,x,sigma)

	u = (K + lambda*I)\y

	predict(Xhat) = rbfKernel(Xhat,x,sigma)*u

	return LinearModel(predict,u)
end

function leastSquaresBasis(x,y,p)
	Z = polyBasis(x,p)

	v = (Z'*Z)\(Z'*y)

	predict(xhat) = polyBasis(xhat,p)*v

	return LinearModel(predict,v)
end

function polyBasis(x,p)
	n = length(x)
	Z = zeros(n,p+1)
	for i in 0:p
		Z[:,i+1] = x.^i
	end
	return Z
end

function weightedLeastSquares(X,y,v)
	V = diagm(v)
	w = (X'*V*X)\(X'*V*y)
	predict(Xhat) = Xhat*w
	return LinearModel(predict,w)
end

function binaryLeastSquares(X,y)
	w = (X'X)\(X'y)

	predict(Xhat) = sign.(Xhat*w)

	return LinearModel(predict,w)
end


function leastSquaresRBF(X,y,sigma)
	(n,d) = size(X)

	Z = rbf(X,X,sigma)

	v = (Z'*Z)\(Z'*y)

	predict(Xhat) = rbf(Xhat,X,sigma)*v

	return LinearModel(predict,v)
end

function rbf(Xhat,X,sigma)
	(t,d) = size(Xhat)
	n = size(X,1)
	D = distancesSquared(Xhat,X)
	return (1/sqrt(2pi*sigma^2))exp.(-D/(2sigma^2))
end
