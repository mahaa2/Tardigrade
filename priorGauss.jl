type ppGauss <: prior
	mu::prior
	sig2::prior
end

type priorGauss <: prior
	μ::Float64
	σ²::Float64
	lp::Function
	dlp::Function
	d2lp::Function
	p::ppGauss
end

function priorGauss(µ::Float64, σ²::Float64, muprior::prior, sig2prior::prior)
 # DESCRIPTION :
 # creates a structure for the Gaussian prior distribution

 if ~(σ² > 0)
	 error("σ² must be greater than zero")
 end

  function lp{T <: Real}(p::priorGauss, x::T)
	 # DESCRIPTION:
	 # log-prior for the Gaussian distribution

		lp = -0.5 .* (log(2.*pi) + log(p.σ²) + (x - p.μ).^2 ./ p.σ²)

		if ~isequal(typeof(p.p.mu), priorEmpty)
			lp += p.p.mu.lp(p.p.mu, p.μ)
		end

		if ~isequal(typeof(p.p.sig2), priorEmpty)
			lp += p.p.sig2.lp(p.p.sig2, p.σ²)
		end

		return(lp)
  end

  function dlp{T <: Real}(p::priorGauss, x::T)
	 # DESCRIPTION:
	 # derivative of log-prior for the Gaussian distribution

		dlp = -(x - p.μ)/p.σ²

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

		return(dlp)
	end

	function d2lp{T <: Real}(p::priorGauss, x::T)
	 # DESCRIPTION:
	 # 2nd derivative of log-prior Gaussian distribution

		d2lp = -1/p.σ²
	end

  # pass the structure
  priorGauss(µ, σ², lp, dlp, d2lp, ppGauss(muprior, sig2prior))
end

# pGauss = priorGauss(0.0, 1, priorEmpty(), priorEmpty());
