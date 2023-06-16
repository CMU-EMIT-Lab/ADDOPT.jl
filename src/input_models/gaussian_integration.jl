
# Fully naive (single sample per point)
(l^2 / (2π*σ^2)) * exp(-((xₙ - xₜ)^2 + (zₙ - zₜ)^2)/(2σ^2))

# for NxN subgrid, N odd

function gau(xₙ, zₙ, σ, N, xₜ, zₜ)
    a = zeros(size(xₙ)...)

    for i in 0:(N-1)
        for j in 0:(N-1)
            xₛ = (l/N)*i - (l/2)
            zₛ = (l/N)*j - (l/2)

            @. a += ((l/N)^2 / (2π*σ^2)) * exp(-((xₙ + xₛ - xₜ)^2 + (zₙ + zₛ - zₜ)^2)/(2σ^2))
        end
    end

    return a
end

## simpler is better^ for some reason

function simpson(xₙ, zₙ, σ, N, xₜ, zₜ)
    a = zeros(size(xₙ)...)
    W = [1; 4; 2; 4; 1]

    for i in 0:(N-1)
        for j in 0:(N-1)
            xₛ = (l/N)*i - (l/2)
            zₛ = (l/N)*j - (l/2)

            @. a += (1/ (2π*σ^2)) * (W[i+1]*W[j+1])*exp(-((xₙ + xₛ - xₜ)^2 + (zₙ + zₛ - zₜ)^2)/(2σ^2))
        end
    end

    @. a *= (l^2/(9*(N-1)^2)) 

    return a
end