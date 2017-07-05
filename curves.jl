function curves(x::Vector{Float64}, y::Vector{Float64}, z::Matrix{Float64}, lvls::Vector{Float64})
 # DESCRIPTION:
 # return the coordinates of contour lines

 n = length(lvls)
 p = Array{Array{Float64, 2}}(n)
 c = [contour(x, y, z, lvls[i]) for i = 1:n]

 i = 0
 for cl in c
    lvl = level(cl) 
    i += 1
    for line in lines(cl)
        xc, yc = coordinates(line)
        p[i] = [xc yc]
    end
 end

 return(p)
end