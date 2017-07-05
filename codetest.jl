# · code testing

# sample data from Lawless 1982 Statistical models and methods for lifetime-data
y = log.([32.0; 35.4; 36.2; 39.8; 41.2; 43.3; 
45.5; 46.0; 46.2; 46.4; 46.5; 46.8; 47.3; 
47.3; 47.6; 49.5; 50.4; 50.9; 52.4; 56.3]);

# define priors
pGauss = priorGauss(0.0, 500.0, priorEmpty(), priorEmpty());

# define models
likA = likEvA(0.0, 0.0, pGauss, pGauss, -3.92)
likH = likEvH(0.0, 0.0, pGauss, pGauss)
lik1 = likEv1(0.0, 0.0, pGauss, pGauss)

# Laplace approximation
dA = LP(likA, y)
dH = LP(likH, y)
d1 = LP(lik1, y)

# define delta
Δa = sqrt(diag(dA.Σ.mat))
Δh = sqrt(diag(dH.Σ.mat))
Δ1 = sqrt(diag(d1.Σ.mat))

# normalization constants
ka = pcubature(x -> exp(likA.llik(likA, y, x)), dA.μ-3.*Δa, dA.μ+3.*Δa)[1]
kh = pcubature(x -> exp(likH.llik(likH, y, x)), dH.μ-3.*Δh, dH.μ+3.*Δh)[1]
k1 = pcubature(x -> exp(lik1.llik(lik1, y, x)), d1.μ-3.*Δ1, d1.μ+3.*Δ1)[1]

# true posteriors
fa(x) = exp(likA.llik(likA, y, x))/ka
fh(x) = exp(likH.llik(likH, y, x))/kh
f1(x) = exp(lik1.llik(lik1, y, x))/k1

# countour plots
np = 200
lvl = [0.1; 0.5; 0.9];

xsa = linspace(dA.μ[1]-3.*Δa[1], dA.μ[1]+3.*Δa[1], np); ysa = linspace(dA.μ[2]-3.*Δa[2], dA.μ[2]+3.*Δa[2], np);
zsa = Float64[fa([x1; x2]) for x1 in xsa, x2 in ysa]; # true countours
zsla = Float64[pdf(dA, [x1; x2]) for x1 in xsa, x2 in ysa]; # approximate Gaussian (elliptical) countours
clvla = maximum(zsa).*lvl;

xsh = linspace(dH.μ[1]-5.*Δh[1], dH.μ[1]+3.*Δh[1], 2*np); ysh = linspace(dH.μ[2]-3.*Δh[2], dH.μ[2]+3.*Δh[2], 2*np);
zsh = Float64[fh([x1; x2]) for x1 in xsh, x2 in ysh]; # true countours
zslh = Float64[pdf(dH, [x1; x2]) for x1 in xsh, x2 in ysh]; # approximate Gaussian (elliptical) countours
zs1 = Float64[f1([x1; x2]) for x1 in xsh, x2 in ysh]; # true countours
zsl1 = Float64[pdf(d1, [x1; x2]) for x1 in xsh, x2 in ysh]; # approximate Gaussian (elliptical) countours
clvlh = maximum(zsh).*lvl;

# countour plots with functions countour
ca = curves(collect(xsa), collect(ysa), zsa, clvla);
cla = curves(collect(xsa), collect(ysa), zsla, clvla);
cal = [ca; cla];

ch = curves(collect(xsh), collect(ysh), zsh, clvlh);
clh = curves(collect(xsh), collect(ysh), zslh, clvlh);
chl = [ch; clh];

c1 = curves(collect(xsh), collect(ysh), zs1, clvlh);
cl1 = curves(collect(xsh), collect(ysh), zsl1, clvlh);
c1l = [c1; cl1];

cols = repeat(["black"; "red"], inner = length(lvl))

pa = plot([layer(x = cal[i][:, 1], y = cal[i][:, 2], Geom.path, 
           Theme(default_color = color(cols[i]))) for i in 1:length(cal)] ...,
           Theme(background_color = color("White")), Guide.yticks(label=false),
           Guide.xticks(label=false), Guide.ylabel(nothing), Guide.xlabel("Constant volume/area element (Achcar)"),
           Coord.Cartesian(xmin=dA.μ[1]-3.*Δa[1], xmax=dA.μ[1]+3.*Δa[1],
           ymin=dA.μ[2]-3.*Δa[2], ymax=dA.μ[2]+3.*Δa[2]));

ph = plot([layer(x = chl[i][:, 1], y = chl[i][:, 2], Geom.path, 
           Theme(default_color = color(cols[i]))) for i in 1:length(chl)] ...,
           Theme(background_color = color("White")), Guide.yticks(label=false),
           Guide.xticks(label=false), Guide.ylabel(nothing), Guide.xlabel("Orthogonal (Jeffreys)"),
           Coord.Cartesian(xmin=dH.μ[1]-1.1.*Δh[2], xmax=dH.μ[1]+1.1.*Δh[2],
           ymin=dH.μ[2]-3.*Δh[2], ymax=dH.μ[2]+3.*Δh[2]));

p1 = plot([layer(x = c1l[i][:, 1], y = c1l[i][:, 2], Geom.path, 
           Theme(default_color = color(cols[i]))) for i in 1:length(c1l)] ...,
           Theme(background_color = color("White")), Guide.yticks(label=false),
           Guide.xticks(label=false), Guide.ylabel(nothing), Guide.xlabel("Real line"),
           Coord.Cartesian(xmin=d1.μ[1]-0.9.*Δ1[2], xmax=d1.μ[1]+0.9.*Δ1[2],
           ymin=d1.μ[2]-3.*Δ1[2], ymax=d1.μ[2]+3.*Δ1[2]));

pn = plot(x = [0.0], y = [0.0], Geom.point, Guide.yticks(label=false), Guide.ylabel(nothing),
    Guide.xticks(label=false), Guide.xlabel(nothing), Theme(background_color = color("White")));

myplot1 = vstack(hstack(p1, ph), hstack(pa, pn))
draw(PNG("myplot.png", 8inch, 8inch, dpi = 100), myplot1)