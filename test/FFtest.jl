using Distributed

@everywhere include("../src/LindbladEq.jl")
@everywhere begin # to all kernel
using LinearAlgebra, Main.LindbladEq, DifferentialEquations
#--------------------------------------------------------------------------------
"""
Lindbladian
    H  = ∑ cᵢ⁺cᵢ₊₁ + cᵢ₊₁⁺cᵢ.
    Lᵢ = cᵢ⁺cᵢ
"""
function lindbladian(L::Integer, γ::Real, dt::Real; PBC::Bool=false)
    H = diagm(1 => ones(L-1), -1 => ones(L-1))
    PBC && (H[L,1] = H[1,L] = 1)
    Ut = evo_operator(Hermitian(H), dt)
    ls = [QJParticle([i], [1], L, γ*dt) for i in 1:L]
    Ut, ls
end

end
#--------------------------------------------------------------------------------
function jump_den_evo(;
    L=32, γ=0.5, dt=0.05, T=30, config="left", times=64
)
    out = pmap(1:times) do _
        N = round(Int, T/dt)
        Ut, ls = lindbladian(L, γ, dt)
        s = FreeFermionState(L=L, N=L÷2, config=config)
        den = zeros(L, N)
        for k = 1:N 
            s = ls * (Ut * s)
            den[:, k] = diag(s)
            GC.gc()
        end
        den
    end
    sum(out) / times
end
#--------------------------------------------------------------------------------
function jump_den_evo2(;
    L=32, γ=0.5, dt=0.05, T=30, config="left", times=64
)
    out = pmap(1:times) do _
        N = round(Int, T/dt)
        Ut, ls = lindbladian(L, γ, dt)
        s = FreeFermionState(L=L, N=L÷2, config=config)
        den = zeros(L, N)
        for k = 1:N 
            s = Ut * s
            for l in ls 
                s = l * s
            end
            den[:, k] = diag(s)
        end
        den
    end
    sum(out) / times
end

function majorana_den_evo(
    L=32, γ=0.5, dt=0.05, T=30
)
    H = diagm(1 => ones(L-1), -1 => ones(L-1))
    M = fill([sqrt(γ);;], L)
    I = [[i] for i in 1:L]
    lind = quadraticlindblad_from_fermion(;H, M, I)
    Γ = covariancematrix([ones(Int,L÷2); zeros(Int,L÷2)])
    
    tspan = (0.0, T)
    prob = ODEProblem((u,p,t) -> (lind*u), Γ, tspan)
    sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8,saveat = dt:dt:T)

    N = round(Int, T/dt)
    den = zeros(L,N)
    for i in 1:N
        den[:, i] = real.(diag(fermioncorrelation(sol.u[i], 1)))
    end
    den
end

function fermion_den_evo(
    L=32, γ=0.5, dt=0.05, T=30
)
    H = diagm(1 => ones(L-1), -1 => ones(L-1))
    M = fill([sqrt(γ);;], L)
    I = [[i] for i in 1:L]
    lind = fermionlindblad(H, M, I)
    Γ = diagm([ones(ComplexF64,L÷2); zeros(ComplexF64,L÷2)])
    
    tspan = (0.0, T)
    prob = ODEProblem((u,p,t) -> (lind*u), Γ, tspan)
    sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8,saveat = dt:dt:T)

    N = round(Int, T/dt)
    den = zeros(L,N)
    for i in 1:N
        den[:, i] = real.(diag(sol.u[i]))
    end
    den
end