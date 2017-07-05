type pplikBern <: prior
	# N::prior
	pr::prior
end

type likBern <: likelihood
	π::Float64
	N::Float64
	llik::Function
	dllik::Function
	d2llik::Function
	p::pplikBern
end

function likBern(π::Float64, prprior::prior, N::Float64)

	if ~isinteger(N) | ~(N > 0)
		error("N is the number of trials. Natural number")
	end

	if (π < 0) | (π > 1)
		error("π is a probability parameter")
	end

	function llik(lik::likBern, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # log-likelihood function

	 	N = lik.N
	 	π = θ[]

		llik = sum(lgamma(N+1)-lgamma(N-y+1)-lgamma(y+1) + y.*log(π) + (N - y).*log(1-π))

		if ~isequal(typeof(lik.p.pr), priorEmpty)
			llik += lik.p.pr.lp(lik.p.pr, π)
		end

		return(llik)
	end

	function dllik(lik::likBern, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # gradient vector log-likelihood function

		N = lik.N
	 	π = θ[]
	 	n = length(y)

		dl = Float64[]

		if ~isequal(typeof(lik.p.pr), priorEmpty)
			dll = (sum(y)./π - (n*N - sum(y))./(1-π))
			dll += lik.p.pr.dlp(lik.p.pr, π)
			dl = [dl; dll]
		end

		return(dl)
	end

	function d2llik(lik::likBern, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # hessian matrix of the log-likelihood function

	 	N = lik.N
	 	π = θ[]
	 	n = length(y)

	 	d2l = Float64[]

		if ~isequal(typeof(lik.p.pr), priorEmpty)
			d2ll = (-sum(y)./(π.^2) -(n*N - sum(y))./(1-π).^2)
			d2ll += lik.p.pr.d2lp(lik.p.pr, π)
			d2l = [d2l; d2ll]
		end

		return(d2l)
	end

	# pass the structure
	likBern(π, N, llik, dllik, d2llik, pplikBern(prprior))
end

lik = likBern(1.0, priorGaussT(0.5, 0.2, priorEmpty(), priorEmpty(), 0.0, 1.0), 0.1)
# lik = likBern(1.0, 0.1, priorEmpty())