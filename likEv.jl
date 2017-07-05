type pplikEv <: prior
	mu::prior
	sig::prior
end

type likEv <: likelihood
	μ::Float64
	σ::Float64
	llik::Function
	dllik::Function
	d2llik::Function
	G::Function
	p::pplikEv
end

function likEv(µ::Float64, σ::Float64, muprior::prior, sigprior::prior)

	if ~(σ > 0) 
	  error("σ must be greater than zero")
	end

	function llik(lik::likEv, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # log-likelihood function

		(μ, σ) = θ

		w = (y - μ)./σ
		n = length(y)

		llik = -n*log(σ) + sum(w - exp(w))

		if ~isequal(typeof(lik.p.mu), priorEmpty)
			llik += lik.p.mu.lp(lik.p.mu, μ)
		end

		if ~isequal(typeof(lik.p.sig), priorEmpty)
			llik += lik.p.sig.lp(lik.p.sig, σ)
		end

		return(llik)
	end

	function dllik(lik::likEv, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # gradient vector log-likelihood function

	 	(μ, σ) = θ

		w = (y - μ)./σ
		n = length(y)

		dl = Float64[]

		if ~isequal(typeof(lik.p.mu), priorEmpty)
			dll = 1./σ .*(-n + sum(exp(w)))
			dll += lik.p.mu.dlp(lik.p.mu, μ)
			dl = [dl; dll]
		end

		if ~isequal(typeof(lik.p.sig), priorEmpty)
			dll = -n./σ - 1./σ.*sum(w) + 1./σ .* sum(w.*exp(w))
			dll += lik.p.sig.dlp(lik.p.sig, σ)
			dl = [dl; dll]
		end

		return(dl)
	end

	function d2llik(lik::likEv, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # hessian matrix of the log-likelihood function

	 	(μ, σ) = θ

	 	w = (y - μ)./σ
	 	n = length(y)
 
 	 	H = zeros(2, 2)
     	H[1, 1] = -1./σ.^2.*sum(exp(w))
     	H[1, 2] = H[2, 1] = -1./σ.^2.*(-n + sum(exp(w))) - 1./σ.^2.*sum(w.*exp(w))
     	H[2, 2] = n./σ.^2 + 2./σ.^2.*sum(w) - sum(2./σ.^2.*w.*exp(w) + 1./σ.^2 .*w.^2.*exp(w))

	 return(H)	
	end

	function G(lik::likEv1, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # hessian matrix of the log-likelihood function

	 	(μ, σ) = θ
	 	σ = exp(ø)

	 	w = (y - μ)./σ
	 	n = length(y)
 
 	 	g = zeros(2, 2)
 	 	g[1, 1] = n./σ.^2
 	 	g[1, 2] = g[2, 1] = n./σ^2 * (1 + digamma(1))
 	 	g[2, 2] = n./σ.^2 *(1 + digamma(2) + digamma(2)^2)

	 return(G)	
	end

	# pass the structure
	likEv(µ, σ, llik, dllik, d2llik, G, pplikEv(muprior, sigprior))
end

# lik = likEv(1.0, 1.0, pGauss, pGauss)
# lik = likEv(1.0, 1.0, priorEmpty(),priorEmpty())