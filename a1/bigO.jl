using Printf

function func1(n)
	for i = 1:n
		@printf("I shall not fear gradients!\n");
	end
end

function func2(n)
	x = ones(100);
	return x *= n;
end

function func3(n)
	x = ones(n,1);
	return x .+ 100; 
end

function func4(n)
	X = zeros(n,n);
	for i in 1:n
		for j in 1:n
			X[i,j] = i+j;
		end
	end
	return X;
end

	
