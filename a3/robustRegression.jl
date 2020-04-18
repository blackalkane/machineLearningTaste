using Printf
include("misc.jl")
include("findMin.jl")

function robustRegression(X,y)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = robustRegressionObj(w,X,y)

	# This is how you compute the function and gradient:
	(f,g) = funObj(w)

	# Derivative check that the gradient code is correct:
	g2 = numGrad(funObj,w)

	if maximum(abs.(g-g2)) > 1e-4
		@printf("User and numerical derivatives differ:\n")
		@show([g g2])
	else
		@printf("User and numerical derivatives agree\n")
	end

	# Solve least squares problem
	w = findMin(funObj,w)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end

function robustRegressionObj(w,X,y)
	(n,d) = size(X)
	f = 0
	for i in 1:n
		temp = (X[i,:]*w)[1] - y[i]
		if (abs(temp) <= 1)
			f += 0.5*((temp)^2)
		else
			f += (abs(temp)-0.5)
		end
	end
	@show f
	g = zeros(size(w))
	for i in 1:n
		temp = (X[i,:]*w)[1] - y[i]
		if (temp < -1)
			g .-= X[i,:]
		elseif (temp > 1)
			g .+= X[i,:]
		else
			g .+= X[i,:] * temp
		end
	end
	@show g
	return (f,g)
end
