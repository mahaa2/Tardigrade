type options <: GP
    numberIter::Int64
    errorMeth::Float64
end

type modelSet <: GP
    lik::likelihood
    infer::String
    opt::options
end

function modelSet(lik::likelihood, infer::String)
 # DESCRIPTION
 # Set the options for the inference algorithms

 if ~any(["MCMC" "LP" "EP"] .== infer)
    error("One must choose MCMC, EP or LP")
 end

 # pass the structure
 modelSet(lik, infer, options(10, 0.1))
end

ps = modelSet(lik, "EP");
