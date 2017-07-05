type pplikEvH <: prior
	th1::prior
	th2::prior
end

type likEvH <: likelihood
	μ::Float64
	σ::Float64
	pak::Function
	unpak::Function
	llik::Function
	dllik::Function
	d2llik::Function
	G::Function
	p::pplikEvH
end

function likEvH(θ¹::Float64, θ²::Float64, th1prior::prior, th2prior::prior)

	function pak(lik::likEvH)
	 # DESCRIPTION :
	 # transform to actual parameterization

	 	k1 = 0.42278433509846847
	 	# (1 + digamma(1))

		θ¹ = lik.μ + k1*lik.σ
		θ² = log(lik.σ)

		return([θ¹, θ²])
	end

	function unpak(lik::likEvH, θ::Vector{Float64})
	 # DESCRIPTION :
	 # transform to the classical parameterization

		(θ¹, θ²) = θ

		k1 = 0.42278433509846847

		lik.μ = θ¹ - k1*exp(θ²)
		lik.σ = exp(θ²)

		return(lik)
	end

	function llik(lik::likEvH, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # log-likelihood function

		(θ¹, θ²) = θ

		k1 = 0.42278433509846847

    	μ = θ¹ - k1*exp(θ²)
		σ = exp(θ²)

		w = (y - μ)./σ
		n = length(y)

		llik = -n*log(σ) + sum(w - exp(w))

		if ~isequal(typeof(lik.p.th1), priorEmpty)
			llik += lik.p.th1.lp(lik.p.th1, θ¹)
		end

		if ~isequal(typeof(lik.p.th2), priorEmpty)
			llik += lik.p.th2.lp(lik.p.th2, θ²)
		end

		return(llik)
	end

	function dllik(lik::likEvH, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # gradient vector log-likelihood function

	 	(θ¹, θ²) = θ

	 	k1 = 0.42278433509846847

    	μ = θ¹ - k1*exp(θ²)
		σ = exp(θ²)

		w = (y - μ)./σ
		w1 = (y - θ¹)./σ
		n = length(y)

		dl = Float64[]

		if ~isequal(typeof(lik.p.th1), priorEmpty)
			dll = 1./σ .*(-n + sum(exp(w)))
			dll += lik.p.th1.dlp(lik.p.th1, θ¹)
			dl = [dl; dll]
		end

		if ~isequal(typeof(lik.p.th2), priorEmpty)
			# dll = (-n./σ - 1./σ.*sum(w1) + 1./σ .* sum(w1.*exp(w))).*σ
			dll = (-n - sum(w1) + sum(w1.*exp(w)))
			dll += lik.p.th2.dlp(lik.p.th2, θ²)
			dl = [dl; dll]
		end

		return(dl)
	end

	function d2llik(lik::likEvH, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # hessian matrix of the log-likelihood function

	 	(θ¹, θ²) = θ

	 	k1 = 0.42278433509846847

    	μ = θ¹ - k1.*exp(θ²)
		σ = exp(θ²)

		w = (y - μ)./σ
		n = length(y)
 
 		# not implemented yet
 	 	# H = eye(2, 2)
 	 	error("not implemented yet")

		return(H)	
	end

	function G(lik::likEvH, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # fisher information matrix

		(θ¹, θ²) = θ

		μ = θ¹ - exp(θ²)
		σ = exp(θ²)

	 	n = length(y)
	 	k1 = 1 + digamma(1)
	 	k2 = 1 + digamma(2) + digamma(2)^2
 
 	 	g = zeros(2, 2)
 	 	g[1, 1] = n./σ.^2
 	 	g[2, 2] = n*(k2 - k1)

 	 	# add negative prior hessian
 	 	d2p = zeros(2, 2)

		if ~isequal(typeof(lik.p.th1), priorEmpty)
			d2p[1, 1] = -lik.p.th1.d2lp(lik.p.th1, θ¹)
		end

		if ~isequal(typeof(lik.p.th2), priorEmpty)
			d2p[2, 2] = -lik.p.th2.d2lp(lik.p.th2, θ²)
		end

		return(g + d2p)	
	end

	# pass the structure
	μ, σ = [θ¹-exp(θ²); exp(θ²)]
	likEvH(μ, σ, pak, unpak, llik, dllik, d2llik, G, pplikEvH(th1prior, th2prior))
end

# lik = likEv1(1.0, 1.0, pGauss, pGauss)
# lik = likEv1(1.0, 1.0, priorEmpty(),priorEmpty())