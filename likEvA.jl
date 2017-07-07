type pplikEvA <: prior
	th1::prior
	th2::prior
end

type likEvA <: likelihood
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
	p::pplikEvA
	a::Float64
end

function likEvA(θ¹::Float64, θ²::Float64, th1prior::prior, th2prior::prior, a::Float64)

	function pak(lik::likEvA)
	 # DESCRIPTION :
	 # transform to actual parameterization

		θ¹ = (lik.μ + lik.a)/lik.σ
		θ² = log(lik.σ)

		return([θ¹, θ²])
	end

	function unpak(lik::likEvA, θ::Vector{Float64})
	 # DESCRIPTION :
	 # transform to the classical parameterization

		(θ¹, θ²) = θ
		
    	lik.μ = θ¹*exp(θ²) - lik.a
		lik.σ = exp(θ²)

		return(lik)
	end

	function llik(lik::likEvA, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # log-likelihood function

		(θ¹, θ²) = θ
		a = lik.a

    	μ = θ¹*exp(θ²) - a
		σ = exp(θ²)

		w = (y - μ)./σ
		n = length(y)

		llik = -n*log(σ) + sum(w - exp.(w))

		if ~isequal(typeof(lik.p.th1), priorEmpty)
			llik += lik.p.th1.lp(lik.p.th1, θ¹)
		end

		if ~isequal(typeof(lik.p.th2), priorEmpty)
			llik += lik.p.th2.lp(lik.p.th2, θ²)
		end

		return(llik)
	end

	function dllik(lik::likEvA, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # gradient vector log-likelihood function

	 	(θ¹, θ²) = θ
		a = lik.a

    	μ = θ¹*exp(θ²) - a
		σ = exp(θ²)

		w = (y - μ)./σ
		w1 = (y + a)./σ.^2
		n = length(y)

		dl = Float64[]

		if ~isequal(typeof(lik.p.th1), priorEmpty)
			dll = 1./σ .*(-n + sum(exp.(w))).*σ
			dll += lik.p.th1.dlp(lik.p.th1, θ¹)
			dl = [dl; dll]
		end

		if ~isequal(typeof(lik.p.th2), priorEmpty)
			dll = (-n./σ - sum(w1) + sum(w1.*exp.(w))).*σ
			dll += lik.p.th2.dlp(lik.p.th2, θ²)
			dl = [dl; dll]
		end

		return(dl)
	end

	function d2llik(lik::likEvA, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # hessian matrix of the log-likelihood function

	 	(θ¹, θ²) = θ
		a = lik.a

    	μ = θ¹*exp(θ²) - a
		σ = exp(θ²)

		w = (y - μ)./σ
		n = length(y)
 
 	 	H = eye(2, 2)

		return(H)	
	end

	function G(lik::likEvA, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # fisher information matrix

		(θ¹, θ²) = θ
		a = lik.a

	 	n = length(y)
	 	k1 = 1 + digamma(1)
	 	k2 = 1 + digamma(2) + digamma(2)^2
 
 	 	g = zeros(2, 2)
 	 	g[1, 1] = n
 	 	g[1, 2] = g[2, 1] = n*(θ¹ + k1)
 	 	g[2, 2] = n*(θ¹.^2 + 2*k1*θ¹ + k2)

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

	function m0(lik::likEvA, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION:
	 # calculate the 0th moment (the normalizing constant)

		f = t -> exp(lik.llik(lik, y, t./(1-t.^2))).*prod((1+t.^2)./((1-t.^2).^2))
		m0 = hcubature(f, -[0.99; 0.92], [0.99; 0.92])[1]

		return(m0)
	end

	function m1(lik::likEvA, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION:
	 # calculate the 1st moment (unnormalized)

		f1 = t -> t[1]/(1-t[1].^2)*exp(lik.llik(lik, y, t./(1-t.^2)))*prod((1+t.^2)./((1-t.^2).^2))
		f2 = t -> t[2]/(1-t[2].^2)*exp(lik.llik(lik, y, t./(1-t.^2)))*prod((1+t.^2)./((1-t.^2).^2))

		m1 = hcubature(f1, -[0.99; 0.95], [0.99; 0.92])[1]
		m2 = hcubature(f2, -[0.99; 0.95], [0.99; 0.92])[1]

		return([m1; m2])
	end

	function m2(lik::likEvA, y::Vector{Float64}, θ::Vector{Float64})
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
	μ, σ = [θ¹*exp(θ²)-a; exp(θ²)]
	likEvA(μ, σ, pak, unpak, llik, dllik, d2llik, m0, m1, m2, G, pplikEvA(th1prior, th2prior), a)
end

# lik = likEv1(1.0, 1.0, pGauss, pGauss)
# lik = likEv1(1.0, 1.0, priorEmpty(),priorEmpty())