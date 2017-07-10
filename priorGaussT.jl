type ppGaussT <: prior
	mu::prior
	sig2::prior
end

type priorGaussT <: prior
	μ::Float64
	σ²::Float64
	a::Float64
	b::Float64
	Phi::Function
	lp::Function
	dlp::Function
	d2lp::Function
	p::ppGaussT
end

function priorGaussT(µ::Float64, σ²::Float64, muprior::prior, sig2prior::prior, a::Float64,	b::Float64)
 # DESCRIPTION :
 # creates a Gaussian prior distribution

	if ~(σ² > 0)
		error("σ² (scale parameter) must be greater than zero")
	end

 	if a < b
		if ~isfinite(a) && ~isfinite(b)
			warn("non-finite bounds")
			throw(BoundsError())
		end
	else
		warn("lower bound greater than upper bound")
		throw(BoundsError())
 	end

 	function Φ(x::Vector{Float64})
	 # DESCRIPRION:
	 # standart gaussian distribution function

		0.5 + 0.5 .* erf(x./sqrt(2))
	end

	function lp(p::priorGaussT, x::Vector{Float64})
	 # DESCRIPTION:
	 # log-prior

		σ = sqrt(p.σ²)
		a = p.a
		b = p.b

		if ~any((x .< a) | (x .> b))
			lp = -0.5*log(2*pi*p.σ²) - 0.5*(x - p.μ).^2 ./ p.σ² - log(p.Phi((b - p.μ)/σ) - p.Phi((a - p.μ)/σ))

			if ~isequal(typeof(p.p.mu), priorEmpty)
				lp += p.p.mu.lp(p.p.mu, p.μ)
			end

			if ~isequal(typeof(p.p.sig2), priorEmpty)
				lp += p.p.sig2.lp(p.p.sig2, p.σ²)
			end
		else
			lp = -Inf
		end

		return(lp)
	end

	function dlp(p::priorGaussT, x::Vector{Float64})
	 # DESCRIPTION:
	 # derivative of log-prior

		if ~any((x .< p.a) | (x .> p.b))
			dlp = -(x - p.μ) / p.σ²

			if ~isequal(typeof(p.p.mu), priorEmpty)
				dlpdmu = (x - p.μ) / p.σ²
				dlpdmu += p.p.mu.dlp(p.p.mu, p.μ)
				dlp = [dlp; dlpdmu]
			end

			if ~isequal(typeof(p.p.sig2), priorEmpty)
				dlpdsig2 = -0.5/p.σ² + 0.5 * (x - p.μ).^2 / p.σ²^2
				dlpdsig2 += p.p.sig2.dlp(p.p.sig2, p.σ²)
				dlp = [dlp; dlpdsig2]
			end

		else
			dlp = -Inf
		end

		return(dlp)
	end

	function d2lp(p::priorGaussT, x::Vector{Float64})
	 # DESCRIPTION:
	 # 2nd derivative of log-prior

		if ~any((x .< p.a) | (x .> p.b))
			d2lp = -1/p.σ²*ones(length(x))
		else
			d2lp = -Inf*ones(length(x))
		end

		return(d2lp)
  end

  # pass the structure
  priorGaussT(µ, σ², a, b, Φ, lp, dlp, d2lp, ppGaussT(muprior, sig2prior))
end

# pgt = priorGaussT(1.2, 2.0, priorEmpty(), priorEmpty(), 0.0, Inf);
