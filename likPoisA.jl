type pplikPoisA <: prior
	th::prior
end

type likPoisA <: likelihood
	λ::Float64
	z::Float64
	pak::Function
	unpak::Function
	llik::Function
	dllik::Function
	d2llik::Function
	m0::Function
	m1::Function
	m2::Function
	G::Function
	p::pplikPoisA
end

function likPoisA(θ::Float64, thprior::prior, z::Float64)

	if (z <= 0)
		error("z is the sampling effort. Real positive number")
	end

	function pak(lik::likPoisA)
	 # DESCRIPTION :
	 # transform to actual parameterization

		θ = [log(lik.λ)]
		return(θ)
	end

	function unpak(lik::likPoisA, θ::Vector{Float64})
	 # DESCRIPTION :
	 # transform to the classical parameterization

		lik.λ = exp(θ[])
		return(lik)
	end

	function llik(lik::likPoisA, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # log-likelihood function

	 	z = lik.z
	 	λ = exp.(θ)
	 	n = length(y)

		llik = -n.*z.*λ + sum(y).*log.(z*λ) - sum(lgamma.(y+1))

		if ~isequal(typeof(lik.p.th), priorEmpty)
			llik += lik.p.th.lp(lik.p.th, θ)
		end

		return(llik)
	end

	function dllik(lik::likPoisA, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # gradient vector log-likelihood function

 	 	z = lik.z
	 	λ = exp.(θ)
	 	n = length(y)

		dl = Float64[]

		if ~isequal(typeof(lik.p.th), priorEmpty)
			dll = (-n*z + sum(y)./λ).*λ
			dll += lik.p.th.dlp(lik.p.th, θ)
			dl = [dl; dll]
		end

		return(dl)
	end

	function d2llik(lik::likPoisA, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # hessian matrix of the log-likelihood function

 	 	z = lik.z
	 	λ = exp.(θ)
	 	nth, n = length(θ), length(y)

	 	d2l = Float64[]

		if ~isequal(typeof(lik.p.th), priorEmpty)
			d2ll = -n*z*λ 
			d2ll += lik.p.th.d2lp(lik.p.th, θ)
			d2l = [d2l; d2ll]
		end

		return(d2l)
	end

	function G(lik::likPoisA, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # metric 

 	 	z = lik.z
	 	λ = exp.(θ)
	 	nth, n = length(θ), length(y)
	 	
	 	g = zeros(nth, nth)
	 	d2p = zeros(nth, nth)

		g[1:nth+1:nth^2] = n*z*λ

		if ~isequal(typeof(lik.p.th), priorEmpty)
			d2p[1:nth+1:nth^2] = -lik.p.th.d2lp(lik.p.th, θ)
		end

		return(g + d2p)
	end

	function m0(lik::likPoisA, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION:
	 # calculate the 0th moment (the normalizing constant) 
	 # here we use the transformation θ = x/(1-x²) to shrink the real line interval

		f0 = (x, v) -> v[:] = exp.(lik.llik(lik, y, x./(1-x.^2))).*(1+x.^2)./((1-x.^2).^2)
		m0 = hquadrature_v(f0, -0.992, 0.992)[1]

		return(m0)
	end

	function m1(lik::likPoisA, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION:
	 # calculate the 1st moment (unnormalized)

		f1 = (x, v) -> v[:] = x./(1-x.^2).*exp.(lik.llik(lik, y, x./(1-x.^2))).*(1+x.^2)./((1-x.^2).^2)
		m1 = hquadrature_v(f1, -0.992, 0.992)[1]

		return(m1)
	end

	function m2(lik::likPoisA, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION:
	 # calculate the 2nd moment (unnormalized)

		f2 = (x, v) -> v[:] = (x./(1-x.^2)).^2 .*exp.(lik.llik(lik, y, x./(1-x.^2))).*(1+x.^2)./((1-x.^2).^2)
		m2 = hquadrature_v(f2, -0.992, 0.992)[1]

		# return([m1 m3; m3 m2])
	end


	# pass the structure
	λ = exp(θ[])
	likPoisA(λ, z, pak, unpak, llik, dllik, d2llik, m0, m1, m2, G, pplikPoisA(thprior))
end

pGauss = priorGauss(0.0, 1.0, priorEmpty(), priorEmpty());
# lik = likPoisA(1.0, priorEmpty(), 1.0)
lik = likPoisA(1.0, pGauss, 1.0)