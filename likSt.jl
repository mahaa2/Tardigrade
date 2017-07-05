type pplikSt <: prior
	mu::prior
	sig::prior
	nu::prior
end

type likSt <: likelihood
	μ::Float64
	σ::Float64
	ν::Float64
	llik::Function
	dllik::Function
	d2llik::Function
	p::pplikSt
end

function likSt(µ::Float64, σ::Float64, ν::Float64, muprior::prior, sigprior::prior, nuprior::prior)

	if ~(σ > 0) | ~(ν > 0)
	  error("σ or ν  must be greater than zero")
	end

	# function randSt(lik::likSt, n::Int64)
	#  # DESCRIPTION:
	#  # generates random sample by the inversion method
	#
	# 	(μ, σ, ν) = [lik.μ, lik.σ, lik.ν]
	# 	warning("not implemented yet")
	# end

	function llik(lik::likSt, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # log-likelihood function

		(μ, σ, ν) = θ

		z = (y - μ)./σ
		zA = z.^2
		zB = 1 + 1./ν .* zA

		llik = sum(lgamma((ν + 1)/2) - lgamma(ν/2) - log(σ) - log(sqrt(pi*ν)) - (ν + 1)/2 .* log(zB))

		if ~isequal(typeof(lik.p.mu), priorEmpty)
			llik += lik.p.mu.lp(lik.p.mu, μ)
		end

		if ~isequal(typeof(lik.p.sig), priorEmpty)
			llik += lik.p.sig.lp(lik.p.sig, σ)
		end

		if ~isequal(typeof(lik.p.nu), priorEmpty)
			llik += lik.p.nu.lp(lik.p.nu, ν)
		end

		return(llik)
	end

	function dllik(lik::likSt, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # gradient vector log-likelihood function

		(μ, σ², ν) = θ

		z = (y - μ)./σ
		zA = z.^2
		zB = 1 + 1./ν .* zA

		dl = Float64[]

		if ~isequal(typeof(lik.p.mu), priorEmpty)
			dll = sum((1 + 1/ν).*z./(σ .* zB))
			dll += lik.p.mu.dlp(lik.p.mu, μ)
			dl = [dl; dll]
		end

		if ~isequal(typeof(lik.p.sig), priorEmpty)
			dll = sum((zA - 1)./(σ.*zB));
			dll += lik.p.sig.dlp(lik.p.sig, σ)
			dl = [dl; dll]
		end

		if ~isequal(typeof(lik.p.nu), priorEmpty)
			dll = 0.5 * sum(digamma((ν + 1)./2) - digamma(ν./2) - 1./ν - log(1 + 1/ν .* zA) + (ν + 1) .* 1/ν^2 .* zA./zB)
			dll += lik.p.nu.dlp(lik.p.nu, ν)
			dl = [dl; dll]
		end

		return(dl)
	end

	function d2llik(lik::likSt, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # hessian matrix of the log-likelihood function

		(μ, σ, ν) = θ

		z = (y - μ)./σ;
		zA = z.^2
		zB = 1 + 1./ν .* zA

		H = zeros(3, 3)
		H[1, 1] = sum(-1/σ^2*(1 + 1/ν)*(2./zB.^2 - 1./zB))
		H[2, 2] = sum(-1/σ^2*(-1 + (1 + 1/ν)*(zA./zB - 2*zA./zB.^2)))
		# H[2, 2] = sum(-1/σ^2 *(-1 + (1 + 1/ν)*z.^2 ./ (1 + 1/ν*z.^2) + (1 + 1/ν)*2*z.^2 ./ ((1 + 1/ν*z.^2).^2)))
		H[3, 3] = sum(polygamma(1, (ν + 1)/2)/4 - polygamma(1, ν/2)/4 + 1./(2*ν^2) + (1/ν^2)*zA./zB +
        (ν + 1)/2 * (-z.^2/(ν^3).*(2 + 1/ν*zA))./(zB.^2))

		H[1, 2] = sum(-2/σ^2 * (1 + 1/ν)*z./(zB.^2))
		H[1, 3] = sum(1/(σ*ν^2)*z.*(z.^2 - 1)./zB.^2)
		H[2, 3] = sum(1/(σ*ν^2)*(zA - 1).*(zA)./(zB.^2))

		H[3, 2] = H[2, 3]; H[3, 1] = H[1, 3]; H[2, 1] = H[1, 2]

		return(H)
	end

	function G(lik::likSt, y::Vector{Float64}, θ::Vector{Float64})
	 # DESCRIPTION :
	 # hessian matrix of the log-likelihood function

		(μ, σ, ν) = θ
		n = length(y)

		z = (y - μ)./σ;
		zA = z.^2
		zB = 1 + 1./ν .* zA

		g = zeros(3, 3)
		g[1, 1] = n/σ.^2*(ν + 1)./(ν + 3)
		g[2, 2] = 2*n/σ.^2*ν./(ν + 3)
		g[3, 3] = n*(polygamma(1, (ν/2))/4 - polygamma(1, (ν+1)/2)/4 - 2*(ν+5)./(ν*(ν+1)(ν+3)))
		g[2, 3] = g[3, 2] = -2*n./σ * 1/((ν+1)*(ν+3))

		return(g)
	end

	# pass the structure
	likSt(µ, σ, ν, llik, dllik, d2llik, pplikSt(muprior, sigprior, nuprior))
end

lik = likSt(1.0, 1.0, 4.0, pGauss, pGauss, pGauss)
# lik = likSt(0.0, 1.0, 4.0, priorEmpty(), priorEmpty(), priorEmpty())
