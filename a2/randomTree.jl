function randomStump(X,y)
	# Fits a decision stump based on inequality rule

	# Get the size of the data matrix
	(n,d) = size(X)

	# Initialize with the best rule with the baseline rule (no split)
	y_mode = mode(y)
	maxGain = 0
	splitVariable = [];
	splitValue = [];
	splitYes = y_mode;
	splitNo = [];

  # Compute number of classes (assumes we have classes {1,2,...,k})
  k = maximum(y)

	# Compute total entropy
  p = zeros(k)
  for i in 1:n
    p[y[i]] += 1
  end
  p ./= n
  Htotal = -sum(p.*log0(p))

	# Search for the best rule
	# (Uses O(n^2d) approach to keep code simple, doesn't use any sparisty)
	yhat = zeros(n)
	for j in rand(1:d,Int(floor(sqrt(d))))
		# Try unique values of column as split values
		for val in unique(X[:,j])

			# Test whether each object satisfies equality
			yes = X[:,j] .> val

			# Find correct label on both sides of split
			y_yes = mode(y[yes])
			y_no = mode(y[.!yes])

			# Make predictions
			yhat[yes] .= y_yes
			yhat[.!yes] .= y_no

			# Compute infogain
			countyes = sum(yes)
			countno = n-countyes
      pyes = zeros(k)
      pno = zeros(k)
      for i in 1:n
        if yes[i]
          pyes[y[i]] += 1
        else
          pno[y[i]] += 1
        end
      end
      pyes ./= countyes
      pno ./= countno
			Hyes = -sum(pyes.*log0(pyes))
			Hno = -sum(pno.*log0(pno))
			infoGain = Htotal - (countyes/n)*Hyes - (countno/n)*Hno

			# Update best rule
			if infoGain > maxGain
				maxGain = infoGain
				splitVariable = j
				splitValue = val
				splitYes = y_yes
				splitNo = y_no
			end
		end
	end

	# Now that we have the best rule,
	# let's build our splitting function
	function split(Xhat)
		(t,d) = size(Xhat)
		if isempty(splitVariable)
			return fill(true,t)
		else
			return (Xhat[:,splitVariable] .> splitValue)
		end
	end

	# Now that we have the best rule,
	# let's build our predict function
	function predict(Xhat)
		(t,d) = size(Xhat)
		yes = split(Xhat)
		yhat = fill(splitYes,t)
		if any(.!yes)
			yhat[.!yes] .= splitNo
		end
		return yhat
	end

	return StumpModel(predict,split,isempty(splitNo))
end

function randomTree_sub(X,y,depth)
	# Fits a decision tree using greedy recursive splitting
	# (uses recursion to make the code simpler)
    # (could be made more efficient by passing indices rather than data)

	(n,d) = size(X)
	# Learn a random decision stump (if we're at the last layer we should probably switch to accuracy)
	splitModel = randomStump(X,y)

	if depth <= 1 || splitModel.baseSplit
		# Base cases where we stop splitting:
		# - this stump gets us to the max depth
		# - this stump doesn't split the data
		return splitModel
	else
		# Use the decision stump to split the data
		yes = splitModel.split(X)

		# Recusively fit a decision tree to each split
		yesModel = randomTree_sub(X[yes,:],y[yes],depth-1)
		noModel = randomTree_sub(X[.!yes,:],y[.!yes],depth-1)

		# Make a predict function
		function predict(Xhat)
			(t,d) = size(Xhat)
			yhat = zeros(t)

			yes = splitModel.split(Xhat)

			yhat[yes] .= yesModel.predict(Xhat[yes,:])
			yhat[.!yes] .= noModel.predict(Xhat[.!yes,:])
			return yhat
		end

		return GenericModel(predict)
	end
end

function randomTree(X,y,depth)
	# Generates boostrap sample of X, then fit a random tree
	(n,d) = size(X)
	bootstrap = rand(1:n,n)
	randomTree_sub(X[bootstrap,:],y[bootstrap],depth)
end

function randomForest(X,y,depth, nTrees)
	subModels = Array{GenericModel}(undef,nTrees)
	for i in 1:nTrees
		tree = randomTree(X,y,depth)
		subModels[i] = tree
	end
	
	function predict(Xhat)
		(t,d) = size(Xhat)
		yhat = zeros(t,nTrees)
		for i in 1:nTrees
			yhat[:,i] = subModels[i].predict(Xhat)
		end
		final_yhat = zeros(t)
		for i in 1:t
			final_yhat[i] = mode(yhat[i,:])
		end
		return final_yhat
	end
	return GenericModel(predict)
end
