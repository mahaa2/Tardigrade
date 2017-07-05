function EP(lik::likelihood, y::Array{Float64, 1})
 # DESCRIPTION :
 # algorithm for expectation-propagation (EP) approximation
 
 	# dimension of the parametric space
	d = sum([getfield(lik.p, a) for a in fieldnames(lik.p)] .== priorEmpty())

	# define likelihood
	L(θ, y) = exp(lik.lLik(lik, y, θ))

	# define prior distribution
	warn("not implemented yet")

    # initialize mean and covariances	
	n = size(y, 1)

	# site parameters (n + 1 if there is prior)
	Z = Array{Array{Float64, 1}}(n)
	μ = Array{Array{Float64, 1}}(n)
	Σ = Array{Array{Float64, 2}}(n)

	m = zeros(d)
	C = 5 .* eye(d)

	for j in 1:(n + 1)
		μ[j] = m; Σ[j] = C; Z[j] = 1
	end

	# create observations indices
	ind = collect(1:n)

	# loop likelihood terms
	for j in ind

		# exclude observation j
		indj = ind[ind .!= j]

		# take remanining site parameters
		# Z_j = Z[indj]
		# μ_j = μ[indj]
		# Σ_j = Σ[indj]

		for i in indj
			if i == indj[1]
				Σ_mj = Σ[i]

				L = chol(Σ_mj, Val{:L})
				μ_mj = L' \ (L \ μ[i])

			else
				# covariance site parameter (cavity distribution)
				# inv(inv(A) + inv(B)) = A - A * inv(B + A) * A
				# recursive update 
				Σ_m = Σ_mj + Σ[i]
				L = chol(Σ_m, Val{:L})
				Σ_mj = Σ_mj - Σ_mj * L' \ (L \ Σ_mj)

				# mean site parameter (cavity distribution)
				μ_mj += L' \ (L \ μ[i])
			end

			if i == indj[end]
				μ_mj = Σ_mj * μ_mj
			end
		end

		# cavity distribution
		q_mj(θ) = pdf(MvNormal(μ_mj, Σ_mj), θ)

		# unnormalizad tilded distribution
		qt(θ) = L(θ, [y[j]]) * q_mj(θ)

	end

	# loop prior
end