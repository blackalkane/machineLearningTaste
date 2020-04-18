include("misc.jl") # Includes GenericModel typedef

function naiveBayes(X,y)
	# Implementation of naive Bayes classifier for binary features

	(n,d) = size(X)

  # Compute number of classes, assuming y in {1,2,...,k}
  k = maximum(y)
  # @show k = 4

  # We will store p(y(i) = c) in p_y(c)
  counts = zeros(k)
  for i in 1:n
    counts[y[i]] += 1
  end
  p_y = counts ./ n
  # @show p_y probability of each newsgroup among all

  # We will store p(x(i,j) = 1 | y(i) = c) in p_xy(1,j,c)
  # We will store p(x(i,j) = 0 | y(i) = c) in p_xy(2,j,c)
  p_xy = zeros(2,d,k)
  # number of newsgroup
  for c in 1:k
	  # get matrix from each newsgroup, y(i) = c
	  x_temp = X[y[:,1] .== c, :]
	  colNum = 1
	  # iterate through all the words
	  for col in eachcol(x_temp)
		  ones_counter = 0
		  zeros_counter = 0
		  # store p(x(i,j) = 1,  y(i) = c) and p(x(i,j) = 0,  y(i) = c)
		  p_x1_yc = 0
		  p_x0_yc = 0
		  # iterate through all the newsgroup post from centain newsgroup
		  for i in eachindex(col)
			  if col[i] == 1
				  ones_counter += 1
			  elseif col[i] == 0
				  zeros_counter += 1
			  end
		  end
		  p_x1_yc = ones_counter / n
		  p_x0_yc = zeros_counter / n
		  # update p_xy for each newsgroup and each word
		  p_xy[1,colNum,c] = p_x1_yc / p_y[c]
		  p_xy[2,colNum,c] = p_x0_yc / p_y[c]
		  # update col num
		  colNum += 1
	  end
  end
  @show p_xy

  function predict(Xhat)
    (t,d) = size(Xhat)
    yhat = zeros(t)

    for i in 1:t
      # p_yx = p_y*prod(p_xy) for the appropriate x and y values
      p_yx = copy(p_y)
      for j in 1:d
        if Xhat[i,j] == 1
          for c in 1:k
            p_yx[c] *= p_xy[1,j,c]
          end
        else
          for c in 1:k
            p_yx[c] *= p_xy[2,j,c]
          end
        end
      (~,yhat[i]) = findmax(p_yx)
      end
    end
    return yhat
  end

	return GenericModel(predict)
end
