type pplikPois <: prior
	lamb::prior
end

type likPois <: likelihood
	λ::Float64
	z::Float64
	llik::Function
	dllik::Function
	d2llik::Function
	p::pplikPois
end

function likPois(λ::Float64, lambprior::prior, z::Float64)

	if ~(z > 0)
		error("z is the sampling effort. Real positive number")
	end

	if ~(λ > 0)
		error("λ is positive")
	end

	function llik(lik::likPois, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # log-likelihood function

	 	z = lik.z
	 	λ = θ[]
	 	n = length(y)

		llik = -n*z*λ + sum(y).*log(z*λ) - sum(lgamma(y+1))

		if ~isequal(typeof(lik.p.lamb), priorEmpty)
			llik += lik.p.lamb.lp(lik.p.lamb, λ)
		end

		return(llik)
	end

	function dllik(lik::likPois, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # gradient vector log-likelihood function

 	 	z = lik.z
	 	λ = θ[]
	 	n = length(y)

		dl = Float64[]

		if ~isequal(typeof(lik.p.lamb), priorEmpty)
			dll = -n*z + sum(y)./λ 
			dll += lik.p.lamb.dlp(lik.p.lamb, λ)
			dl = [dl; dll]
		end

		return(dl)
	end

	function d2llik(lik::likPois, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # hessian matrix of the log-likelihood function

 	 	z = lik.z
	 	λ = θ[]
	 	n = length(y)

	 	d2l = Float64[]

		if ~isequal(typeof(lik.p.lamb), priorEmpty)
			d2ll = -sum(y)./(λ.^2)
			d2ll += lik.p.lamb.d2lp(lik.p.lamb, λ)
			d2l = [d2l; d2ll]
		end

		return(d2l)
	end

	# pass the structure
	likPois(λ, z, llik, dllik, d2llik, pplikPois(lambprior))
end

lik = likPois(1.0, priorGaussT(1.0, 1.0, priorEmpty(), priorEmpty(), 0.0, Inf), 1.0)
# lik = likPois(1.0, 0.1, priorEmpty())