type ppSt <: prior
	mu::prior
	sig::prior
	nu::prior
end

type priorSt <: prior
	μ::Float64
	σ::Float64
	ν::Float64
	lp::Function
	dlp::Function
	d2lp::Function
	p::ppSt
end

function priorSt(µ::Float64, σ::Float64, ν::Float64, muprior::prior, sigprior::prior, nuprior::prior)
 # DESCRIPTION :
 # creates a Student-t prior distribution

  if ~(σ > 0) | ~(ν > 0)
	  error("σ or ν  must be greater than zero")
  end

	function lp(p::priorSt, x::Vector{Float64})
	 # DESCRIPTION:
	 # log-prior for Student-t prior

		z = (x - p.μ)./ p.σ
		zA = z.^2
		zB = 1 + 1./p.ν .* zA

		lp = lgamma((p.ν + 1)/2)-lgamma(p.ν/2)-log(p.σ)-log(sqrt(pi*p.ν))-(ν + 1)/2
		.*log(zB)

		if ~isequal(typeof(p.p.mu), priorEmpty)
			lp += p.p.mu.lp(p.p.mu, p.μ)
		end

		if ~isequal(typeof(p.p.sig), priorEmpty)
			lp += p.p.sig.lp(p.p.sig, p.σ)
		end

		if ~isequal(typeof(p.p.nu), priorEmpty)
			lp += p.p.nu.lp(p.p.nu, p.ν)
		end

		return(lp)
	end

	function dlp(p::priorSt, x::Vector{Float64})
	 # DESCRIPTION:
	 # derivative of log-prior for Student-t prior

	  z = (x - p.μ)./ p.σ
	  zA = z.^2
	  zB = 1 + 1./p.ν .* zA

		dlp = -(1 + 1/p.ν).*z./(p.σ .* zB);

		if ~isequal(typeof(p.p.mu), priorEmpty)
			dlpdmu = (1 + 1/p.ν).*z./(p.σ .* zB)
			dlpdmu += p.p.mu.dlp(p.p.mu, p.μ)
			dlp = [dlp; dlpdmu]
		end

		if ~isequal(typeof(p.p.sig), priorEmpty)
			dlpdsig = (zA - 1)./(zB .* p.σ)
			dlpdsig += p.p.sig.dlp(p.p.sig2, p.σ)
			dlp = [dlp; dlpdsig]
		end

		if ~isequal(typeof(p.p.nu), priorEmpty)
			dlpdnu = 0.5 * (digamma((p.ν + 1)./2) - digamma(p.ν./2) - 1./p.ν - log(1 + 1/p.ν .* zA) + (p.ν + 1) .* 1./p.ν^2 .* zA./zB)
			dlpdnu += p.p.nu.dlp(p.p.nu, p.ν)
			dlp = [dlp; dlpdnu]
		end

		return(dlp)
	end

	function d2lp(p::priorSt, x::Vector{Float64})
	 # DESCRIPTION:
	 # 2nd derivative of log-prior for Student-t prior

      z = (x - p.μ)./ p.σ
	  zA = z.^2
	  zB = 1 + 1./p.ν .* zA

		d2lp = -1./(p.σ.^2) .* (1 + 1/p.ν) .* (2./zB.^2 - 1./zB);

		return(d2lp)
	end

  # pass the structure
	priorSt(µ, σ, ν, lp, dlp, d2lp, ppSt(muprior, sigprior, nuprior))
end

pSt = priorSt(0.0, 1.0, 4.0, priorEmpty(), priorEmpty(), priorEmpty());
