include("../src/LindbladEq.jl")
using DelimitedFiles, LinearAlgebra, Main.LindbladEq
#--------------------------------------------------------------------------------
"""
H = ∑ cᵢ⁺cᵢ₊₁ + cᵢ₊₁⁺cᵢ.
"""
function hopping_ham(L::Integer; PBC::Bool=false)
    H = diagm(1 => ones(L-1), 1 => ones(L-1))
    PBC && (H[L,1] = H[1,L] = 1)
    H
end
#--------------------------------------------------------------------------------
"""
Jump operator:
    Lᵢ = √γ * dᵢ⁺dᵢ,
    dᵢ⁺ = (cᵢ₋₁⁺ + α cᵢ⁺ + cᵢ₊₁⁺) / √2.
"""
function nodal_jump(L::Integer; a=1.0)
    V = [1 , a, 1] / sqrt(2+abs2(a))
    l1 = [QuasiMode([i, mod(i,L)+1, mod(i+1,L)+1], V, L) for i in 1:3:L-2]
    l2 = [QuasiMode([i, mod(i,L)+1, mod(i+1,L)+1], V, L) for i in 2:3:L-1]
    l3 = [QuasiMode([i, mod(i,L)+1, mod(i+1,L)+1], V, L) for i in 3:3:L]
    l1, l2, l3
end

#--------------------------------------------------------------------------------
# Main
#--------------------------------------------------------------------------------
function ent_evo(;L=32, γ=0.1, dt=0.05, T=1000, a=1im, config="Z2")
    N = round(Int, T/dt)
    evo = evo_operator(Hermitian(hopping_ham(L, PBC=true)), dt)
    l1, l2, l3 = nodal_jump(L; a)
    s = FreeFermionState(L=L, N=L÷2, config=config)
    EE = zeros(N)
    for k = 1:N 
        s = evo * s
        wiener!(l1, s, γ*dt)
        wiener!(l2, s, γ*dt)
        wiener!(l3, s, γ*dt)
        EE[k] = ent_S(s, 1:L÷2)
        iszero(mod(k, 200)) && println("L=$L, t=$(k*dt).")
    end
    EE
end

function main(L, T)
    e = ent_evo(;L, T)
    writedlm("2-L=$L.dat", e)
end

const L = [
    32
    45
    64
    91
    128
    181
    256
    362
    512
    724
    1024
]
for l in L 
    main(l, 1000)
end
