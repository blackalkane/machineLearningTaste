include("misc.jl")

function leastSquares(X,y)

	# Find regression weights minimizing squared error
	w = (X'X)\(X'y)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end

function leastSquaresBias(X,y)
	(n,d) = size(X)
	w0 = ones(n,1)
	Xtemp = hcat(w0, X)
	w = (Xtemp'Xtemp)\(Xtemp'y)

	function predict(Xhat)
		(t,) = size(Xhat)
		w0 = ones(t,1)
		yhat = hcat(w0, Xhat)*w
		return yhat
	end

	return GenericModel(predict)
end

function leastSquaresBasis(X,y,p)

	function polyBasis(X,p)
		(n,) = size(X)
		if p == 0
			return ones(n,1)
		end
		if p == 1
			return hcat(ones(n,1), X)
		end
		Xtemp = hcat(ones(n,1), X)
		for i in 2:p
			Xtemp2 = X
			Xtemp = hcat(Xtemp, Xtemp2.^i)
		end
		return Xtemp
	end

	Xtemp = polyBasis(X,p)
	w = (Xtemp'Xtemp)\(Xtemp'y)

	function predict(Xhat)
		Xhattemp = polyBasis(Xhat,p)
		return Xhattemp*w
	end

	return GenericModel(predict)
end

function weightedLeastSquare(X,y,v)
	w = (X'*v*X)\(X'*v*y)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end

function newLeastSquaresBasis(X,y,p,k)
	function nonLinearBasis(X,p,k)
		(n,) = size(X)
		if p == 0
   			return ones(n,1)
  		end
  		if p == 1
   			return hcat(ones(n,1), X)
  		end
		Xtemp = hcat(ones(n,1), X)
  		for i in 2:k
			if i<=p
   				Xtemp = hcat(Xtemp, X.^i)
			else
				Xtemp = hcat(Xtemp, (sin.(i.*X)))
			end
  		end
		return Xtemp
	end
	Xtemp = nonLinearBasis(X,p,k)
 	w = (Xtemp'Xtemp)\(Xtemp'y)

	function predict(Xhat)
  		Xhattemp = nonLinearBasis(Xhat,p,k)
  		return Xhattemp * w
 	end
	return GenericModel(predict)
end
