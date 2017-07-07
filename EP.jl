function EP(lik::likelihood, y::Array{Float64, 1})
 # DESCRIPTION :
 # algorithm for expectation-propagation (EP) approximation

  m0 = lik.m0(lik, y, Float64[])
  m1 = lik.m1(lik, y, Float64[])
  m2 = lik.m2(lik, y, Float64[])

  μ = m1/m0
  Σ = m2/m0 - μ*μ' 

  return(MvNormal(µ, Σ))

  	# # take the priors and remove from the likelihood structure 
	# indf = fieldnames(lik.p)
	# p = Array{Function}(0)

	# for i = 1:length(indf)
	# 	pf = getfield(lik.p, indf[i])
	# 	if ~isequal(pf, priorEmpty())
	# 		p = [p; pf]
	# 		setfield!(lik.p, indf[i], priorEmpty())
	# 	end
	# end

	# # dimension of the parametric space
	# d = length(p)

	# # define likelihood
	# L(θ, y) = exp(lik.llik(lik, y, θ))

	# # initialize mean and covariances	
	# n = length(y)

	# # site parameters (n + 1 if there is prior)
	# Z = ones(n + d)
	# μ = Array{Array{Float64, 1}}(n + d)
	# Σ = Array{Array{Float64, 2}}(n + d)

	# m = zeros(d)
	# C = 5 .* eye(d)

	# for j in 1:(n + d)
	# 	μ[j] = m; Σ[j] = C;
	# end

	# # create observations indices
	# ind = collect(1:n)

	# # loop likelihood terms
	# for j in ind

	# 	# exclude observation j
	# 	indj = ind[ind .!= j]

	# 	# take remanining site parameters
	# 	# Z_j = Z[indj]
	# 	# μ_j = μ[indj]
	# 	# Σ_j = Σ[indj]

	# 	for i in indj
	# 		if i == indj[1]
	# 			Σ_mj = Σ[i]

	# 			L = chol(Σ_mj, Val{:L})
	# 			μ_mj = L' \ (L \ μ[i])

	# 		else
	# 			# covariance site parameter (cavity distribution)
	# 			# inv(inv(A) + inv(B)) = A - A * inv(B + A) * A
	# 			# recursive update 
	# 			Σ_m = Σ_mj + Σ[i]
	# 			L = chol(Σ_m, Val{:L})
	# 			Σ_mj = Σ_mj - Σ_mj * L' \ (L \ Σ_mj)

	# 			# mean site parameter (cavity distribution)
	# 			μ_mj += L' \ (L \ μ[i])
	# 		end

	# 		if i == indj[end]
	# 			μ_mj = Σ_mj * μ_mj
	# 		end
	# 	end

	# 	# cavity distribution
	# 	q_mj(θ) = pdf(MvNormal(μ_mj, Σ_mj), θ)

	# 	# unnormalizad tilded distribution
	# 	qt(θ) = L(θ, [y[j]]) * q_mj(θ)

	# end

	# loop prior
end

# indg = [getfield(lik.p, a) for a in indf] .== priorEmpty()
# d = sum(indg)