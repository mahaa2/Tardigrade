function LP(lik::likelihood, y::Array{Float64, 1})
 # DESCRIPTION :
 # algorithm for the Laplace (LP) approxmation 

	# indexes, initial guess, dimension of the parametric space
	# l = length(fieldnames(lik.p))
	# ind = [getfield(lik.p, a) for a in fieldnames(lik.p)] .!= priorEmpty()
	# θini = [getfield(lik, a) for a in fieldnames(lik)[1:l]][ind]

	θini = lik.pak(lik)

	aux = 0; 
	θNew = θini
	while (norm(lik.dllik(lik, y, θNew)) > 0.01 || aux < 15) 
		# L = ctranspose(chol(lik.G(lik, y, θNew)));
		# Δ = L' \ (L \ lik.dllik(lik, y, θNew));
        Δ = inv(lik.G(lik, y, θNew))*lik.dllik(lik, y, θNew)
		θNew += Δ
		aux += 1
	end

	if any(isnan.(lik.dllik(lik, y, θNew)))
		println("Newton-algorithm")
		aux = 0;
		θNew = θini
	    while norm(lik.dllik(lik, y, θNew)) > 0.05 || aux < 15
	        Δ = -inv(Calculus.hessian(x -> lik.llik(lik, y, x), θNew))*lik.dllik(lik, y, θNew)
			θNew += Δ
			aux += 1
		end
	end

	# function logπ(lik, y, θ)
	# 	lik.llik(lik, y, θ)
	# end

	# function ∇logπ(lik, y, θ)
	# 	L = ctranspose(chol(lik.G(lik, y, θ)));
	# 	Δ = L' \ (L \ lik.dllik(lik, y, θ));
	# end

	# function ∇logπ1(lik, y, θ)
	#  	Δ = -inv(Calculus.hessian(x -> logπ(lik, y, x), θ))*lik.dllik(lik, y, θ)
	# end 

	# aux = 0
	# while norm(lik.dllik(lik, y, θNew)) > 0.01 || aux < 10
	# 	θNew += ∇logπ(lik, y, θNew)
	# 	aux += 1
	# end

	# optimize
	# optm = optimize(logπ, θini);

	# take the map-estimate (mean) and inverse-hessian (covariance-matrix)
	µ = θNew
	A = -Calculus.hessian(x -> lik.llik(lik, y, x), µ);
	iA = inv(A)
	Σ = (iA + iA')/2; # ensure computational symmetry 

	return(MvNormal(µ, Σ))
end 