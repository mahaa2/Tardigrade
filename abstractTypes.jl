# module
# module tardigrade

# packages
using Calculus,
      SpecialFunctions,
      Distributions,
      Cubature
      # Cairo,
      # Gadfly,
      # Contour
      # Optim

 # set_default_plot_format(:png)
 # set_default_plot_size(15cm, 15cm)

# abstract types
 abstract GP
 abstract likelihood <: GP
 abstract prior <: GP

# directories
 cd("$(homedir())/Downloads/tardigrade")

# include other files
 # include("priorGaussT.jl")
 # include("likBern.jl")
 # include("likPois.jl")
 # include("likSt.jl")
 # include("likEv.jl")
 # include("priorSt.jl")
 include("priorEmpty.jl")
 include("priorGauss.jl")
 # include("likEv1.jl")
 # include("likEvA.jl")
 # include("likEvH.jl")
 include("likPoisA.jl")
 include("LP.jl")
 # include("EP.jl")
 # include("curves.jl")
 # include("modelSet.jl")
