type pplikEv1 <: prior
	mu::prior
	phi::prior
end

type likEv1 <: likelihood
	μ::Float64
	σ::Float64
	pak::Function
	unpak::Function
	llik::Function
	dllik::Function
	d2llik::Function
	m0::Function
	m1::Function
	m2::Function
	G::Function
	p::pplikEv1
end

function likEv1(µ::Float64, ø::Float64, muprior::prior, phiprior::prior)

	function pak(lik::likEv1)
	 # DESCRIPTION :
	 # transform to actual parameterization

		μ = lik.μ 
		ø = log(lik.σ)

		return([μ, ø])
	end

	function unpak(lik::likEv1, θ::Vector{Float64})
	 # DESCRIPTION :
	 # transform to the classical parameterization

		(μ, ø) = θ
		
    	lik.μ = μ
		lik.σ = exp(ø)

		return(lik)
	end

	function llik(lik::likEv1, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # log-likelihood function

		(μ, ø) = θ
		σ = exp(ø)

		w = (y - μ)./σ
		n = length(y)

		llik = -n*log(σ) + sum(w - exp.(w))

		if ~isequal(typeof(lik.p.mu), priorEmpty)
			llik += lik.p.mu.lp(lik.p.mu, μ)
		end

		if ~isequal(typeof(lik.p.phi), priorEmpty)
			llik += lik.p.phi.lp(lik.p.phi, ø)
		end

		return(llik)
	end

	function dllik(lik::likEv1, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # gradient vector log-likelihood function

	 	(μ, ø) = θ
	 	σ = exp(ø)

		w = (y - μ)./σ
		n = length(y)

		dl = Float64[]

		if ~isequal(typeof(lik.p.mu), priorEmpty)
			dll = 1./σ .*(-n + sum(exp.(w)))
			dll += lik.p.mu.dlp(lik.p.mu, μ)
			dl = [dl; dll]
		end

		if ~isequal(typeof(lik.p.phi), priorEmpty)
			dll = (-n./σ - 1./σ.*sum(w) + 1./σ .* sum(w.*exp.(w))).*σ
			dll += lik.p.phi.dlp(lik.p.phi, ø)
			dl = [dl; dll]
		end

		return(dl)
	end

	function d2llik(lik::likEv1, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # hessian matrix of the log-likelihood function

	 	(μ, σ) = θ
	 	σ = exp(ø)

	 	w = (y - μ)./σ
	 	n = length(y)
 
 	 	H = zeros(2, 2)
     	H[1, 1] = -1./σ.^2.*sum(exp.(w))
     	H[1, 2] = H[2, 1] = (-1./σ.^2.*(-n + sum(exp.(w))) - 1./σ.^2.*sum(w.*exp.(w))).*σ
     	H[2, 2] = (n./σ.^2 + 2./σ.^2.*sum(w) - sum(2./σ.^2.*w.*exp.(w) + 1./σ.^2 .*w.^2.*exp.(w))).*σ.^2 +
     	(-n./σ - 1./σ.*sum(w) + 1./σ .* sum(w.*exp.(w))).*σ

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
 	 	g[1, 2] = g[2, 1] = n./σ * (1 + digamma(1))
 	 	g[2, 2] = n*(1 + digamma(2) + digamma(2)^2)

 	 	# add negative prior hessian
 	 	d2p = zeros(2, 2)

		if ~isequal(typeof(lik.p.mu), priorEmpty)
			d2p[1, 1] = -lik.p.mu.d2lp(lik.p.mu, μ)
		end

		if ~isequal(typeof(lik.p.phi), priorEmpty)
			d2p[2, 2] = -lik.p.phi.d2lp(lik.p.phi, ø)
		end
	 return(g + d2p)	
	end

	function m0(lik::likEv1, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION:
	 # calculate the 0th moment (the normalizing constant)

		f = t -> exp(lik.llik(lik, y, t./(1-t.^2))).*prod((1+t.^2)./((1-t.^2).^2))
		m0 = hcubature(f, -[0.99; 0.92], [0.99; 0.92])[1]

		return(m0)
	end

	function m1(lik::likEv1, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION:
	 # calculate the 1st moment (unnormalized)

		f1 = t -> t[1]/(1-t[1].^2)*exp(lik.llik(lik, y, t./(1-t.^2)))*prod((1+t.^2)./((1-t.^2).^2))
		f2 = t -> t[2]/(1-t[2].^2)*exp(lik.llik(lik, y, t./(1-t.^2)))*prod((1+t.^2)./((1-t.^2).^2))

		m1 = hcubature(f1, -[0.99; 0.95], [0.99; 0.92])[1]
		m2 = hcubature(f2, -[0.99; 0.95], [0.99; 0.92])[1]

		return([m1; m2])
	end

	function m2(lik::likEv1, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION:
	 # calculate the 2nd moment (unnormalized)

		f1 = t -> (t[1]./(1-t[1].^2))^2*exp(lik.llik(lik, y, t./(1-t.^2))).*prod((1+t.^2)./((1-t.^2).^2))
		f2 = t -> (t[2]./(1-t[2].^2))^2*exp(lik.llik(lik, y, t./(1-t.^2))).*prod((1+t.^2)./((1-t.^2).^2))
		f3 = t -> prod(t./(1-t.^2))*exp(lik.llik(lik, y, t./(1-t.^2))).*prod((1+t.^2)./((1-t.^2).^2))

		m1 = hcubature(f1, -[0.99; 0.92], [0.99; 0.92])[1]
		m2 = hcubature(f2, -[0.92; 0.92], [0.92; 0.92])[1]
		m3 = hcubature(f3, -[0.99; 0.92], [0.99; 0.92])[1]

		return([m1 m3; m3 m2])
	end

	# pass the structure
	µ, σ = [µ; exp(ø)]
	likEv1(µ, σ, pak, unpak, llik, dllik, d2llik, m0, m1, m2, G, pplikEv1(muprior, phiprior))
end

# lik = likEv1(1.0, 1.0, pGauss, pGauss)
# lik = likEv1(1.0, 1.0, priorEmpty(),priorEmpty())