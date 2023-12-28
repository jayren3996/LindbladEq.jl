include("LinAlg.jl")
#----------------------------------------------------------------------------------------------------
# Free fermion States
#----------------------------------------------------------------------------------------------------
export FreeFermionState
"""
Particle-number-preserving free fermion state

Represented by a matrix `B`,
    |ψ⟩ = ⨂ₖ bₖ⁺|0⟩
    bₖ⁺ = ∑ᵢ cᵢ⁺ Bᵢₖ.
A `FreeFermionState` object `s` can be access as a matrix.
The element `s[i,j]` is the two-point function ⟨cᵢ⁺cⱼ⟩.

The boolean value `N` keeps track of the orthogonality of the representation.
"""
struct FreeFermionState{T<:Number}
    B::Matrix{T}    # Matrix representation of free modes
    N::Bool         # Whether normalized
end
#----------------------------------------------------------------------------------------------------
"""
    FreeFermionState(dtype::DataType, pos::AbstractVector{<:Integer}, L::Integer)

Initialize FreeFermionState with given data type, particle positions and system length.

Inputs:
-------
dtype: data type of representing matrix B.
pos  : vector of occupied positions.
L    : length.
"""
function FreeFermionState(dtype::DataType, pos::AbstractVector{<:Integer}, L::Integer)
    B = zeros(dtype, L, length(pos))
    for i = 1:length(pos)
        B[pos[i], i] = 1
    end
    FreeFermionState(B, true)
end
#----------------------------------------------------------------------------------------------------
"""
    FreeFermionState(pos::AbstractVector{<:Integer}, L::Integer) 

Initialize FreeFermionState with given particle positions and system length.
"""
function FreeFermionState(pos::AbstractVector{<:Integer}, L::Integer) 
    FreeFermionState(ComplexF64, pos, L)
end
#----------------------------------------------------------------------------------------------------
"""
    FreeFermionState(dtype::DataType, vec::AbstractVector{Bool})

Initialize FreeFermionState with vector indicating particle positions.
"""
function FreeFermionState(dtype::DataType, vec::AbstractVector{Bool})
    pos = Int64[]
    for i in eachindex(vec)
        vec[i] && push!(pos, i)
    end
    FreeFermionState(dtype, pos, length(vec))
end
#----------------------------------------------------------------------------------------------------
function FreeFermionState(vec::AbstractVector{Bool}) 
    FreeFermionState(ComplexF64, vec)
end
#----------------------------------------------------------------------------------------------------
"""
    FreeFermionState(;L::Integer, N::Integer, config::String="Z2")

More initial configuration.
"""
function FreeFermionState(;L::Integer, N::Integer, config::String="Z2")
    pos = if config == "center"
        l = (L-N)÷2
        l+1:l+N
    elseif config == "left"
        1:N
    elseif config == "right"
        L-N+1:L
    elseif config == "random"
        shuffle(1:L)[1:N]
    elseif config[1] == 'Z'
        n = parse(Int, config[2])
        1:n:1+n*(N-1)
    else
        error("Invalid configureation: $config.")
    end
    FreeFermionState(pos, L)
end

#----------------------------------------------------------------------------------------------------
# Properties
#----------------------------------------------------------------------------------------------------
function Base.eltype(::FreeFermionState{T}) where T 
    return T
end
#----------------------------------------------------------------------------------------------------
function Base.length(s::FreeFermionState) 
    return size(s.B, 1)
end
#----------------------------------------------------------------------------------------------------
"""
Compute correlation ⟨cᵢ⁺cⱼ⟩
"""
function Base.getindex(s::FreeFermionState, i::Integer, j::Integer) 
    return dot(s.B[i,:], s.B[j,:])
end
#----------------------------------------------------------------------------------------------------
"""
Compute correlation matrix
"""
function Base.getindex(s::FreeFermionState, ::Colon) 
    return conj(s.B) * transpose(s.B)
end
function Base.getindex(s::FreeFermionState, ::Colon, ::Colon) 
    return conj(s.B) * transpose(s.B)
end
#----------------------------------------------------------------------------------------------------
"""
Compute particle density on each site.
"""
function LinearAlgebra.diag(s::FreeFermionState)
    n = Vector{Float64}(undef,length(s))
    for i in 1:length(s)
        n[i] = real(s[i,i])
    end
    n
end
#----------------------------------------------------------------------------------------------------
"""
Return particle number.
"""
function LinearAlgebra.rank(s::FreeFermionState) 
    return size(s.B, 2)
end
#----------------------------------------------------------------------------------------------------
orthogonalize(B::AbstractMatrix) = Matrix(qr(B).Q)
"""
Return the normalized FreeFermionState.
"""
function LinearAlgebra.normalize(s::FreeFermionState) 
    s.N ? s : FreeFermionState(orthogonalize(s.B), true)
end
#----------------------------------------------------------------------------------------------------
export ent_S
"""
    entropy(s::FreeFermionState, i::AbstractVector{<:Integer})

Entaglement entropy of gaussian state with system A chosen to be the sites {i}.
"""
function ent_S(s::FreeFermionState, i::AbstractVector{<:Integer})
    B = s.B[i,:]
    vals = svdvals(B).^2
    EE = 0.0
    for x in vals
        x > 1.0 && x-1.0 < 1e-6 && error("Got a Schmidt value λ = $x.")
        x < 1e-14 && continue
        (y = 1.0 - x) < 1e-14 && continue
        EE -= x * log(x) + y * log(y)
    end
    EE
end
#----------------------------------------------------------------------------------------------------
"""
    entropy(s::FreeFermionState, i::AbstractVector{<:Integer}, j::AbstractVector{<:Integer})

Mutual information for FreeFermionState.
"""
function ent_S(
    s::FreeFermionState, 
    i::AbstractVector{<:Integer}, 
    j::AbstractVector{<:Integer}
)
    SA = ent_S(s, i)
    SB = ent_S(s, j)
    SAB = ent_S(s, vcat(i,j))
    SA + SB - SAB
end
#----------------------------------------------------------------------------------------------------
"""
Multiply Unitary matrix to FreeFermionState.
"""
function *(U::Unitary, s::FreeFermionState) 
    FreeFermionState(U.M * s.B, true)
end
#----------------------------------------------------------------------------------------------------
"""
Multiply general matrix to FreeFermionState, and by default normalize the output.
"""
function *(M::AbstractMatrix, s::FreeFermionState; normalize::Bool=true)
    B_new = normalize ? orthogonalize(M * s.B) : M * s.B
    FreeFermionState(B_new, true)
end
#----------------------------------------------------------------------------------------------------
export apply!
"""
    apply!(U::Unitary, s::FreeFermionState, ind::AbstractVector{<:Integer})

Apply local unitary gate to FreeFermionState `s` on sites `inds`.
"""
function apply!(U::Unitary, s::FreeFermionState, inds::AbstractVector{<:Integer})
    s.B[inds, :] = U.M * s.B[inds, :]
    s
end

#----------------------------------------------------------------------------------------------------
# Quasi paritcle
#----------------------------------------------------------------------------------------------------
export QuasiMode
"""
Quasi Mode d⁺ = ∑ Vⱼ cⱼ⁺
"""
struct QuasiMode{T<:Number}
    I::Vector{Int64}
    V::Vector{T}
    L::Int64
    function QuasiMode(I::AbstractVector{<:Integer}, V::AbstractVector{T}, L::Integer) where T <: Number 
        ind = [mod(i-1, L)+1 for i in I]
        new{T}(ind, V, Int64(L))
    end
end
#----------------------------------------------------------------------------------------------------
vector(qm::QuasiMode) = sparsevec(qm.I, qm.V, qm.L)
inner(qm::QuasiMode, s::FreeFermionState) = vec(qm.V' * s.B[qm.I, :])



#----------------------------------------------------------------------------------------------------
# Quantum Jumps
#----------------------------------------------------------------------------------------------------
"""
abstract type for quantum jumps that preserve free fermion structure.
"""
abstract type QuantumJump end
(qj::QuantumJump)(I, V, L, γdt) = qj(QuasiMode(I, V, L), γdt)
#----------------------------------------------------------------------------------------------------
export QJParticle
"""
QJParticle: L = d⁺d
"""
struct QJParticle{T1, T2, T3<:Real} <: QuantumJump
    M::QuasiMode{T1}
    P::Matrix{T2}
    γdt::T3
end
QJParticle(M::QuasiMode, γdt::Real) = QJParticle(M, particle_exp(M, γdt), γdt)
#----------------------------------------------------------------------------------------------------
export QJHole
"""
QJHole: L = dd⁺
"""
struct QJHole{T1, T2, T3<:Real} <: QuantumJump
    M::QuasiMode{T1}
    P::Matrix{T2}
    γdt::T3
end
QJHole(M::QuasiMode, γdt::Real) = QJHole(M, hole_exp(M, γdt), γdt)
#----------------------------------------------------------------------------------------------------
export QJDrain
"""
QJDrain: L = d 
"""
struct QJDrain{T1, T2, T3<:Real} <: QuantumJump
    M::QuasiMode{T1}
    P::Matrix{T2}
    γdt::T3
end
QJDrain(M::QuasiMode, γdt::Real) = QJDrain(M, particle_exp(M, γdt), γdt)
#----------------------------------------------------------------------------------------------------
export QJSource
"""
QJSource: L = d⁺
"""
struct QJSource{T1, T2, T3<:Real} <: QuantumJump
    M::QuasiMode{T1}
    P::Matrix{T2}
    γdt::T3
end
QJSource(M::QuasiMode, γdt::Real) = QJSource(M, hole_exp(M, γdt), γdt)
#----------------------------------------------------------------------------------------------------
"""
Correaction operator if jump does not happen
    P = sqrt(1-γ⋅dt)d⁺d + dd⁺
"""
function particle_exp(qm::QuasiMode, γdt::Real)
    v = hcat(qm.V, nullspace(qm.V'))
    d = ones(length(qm.I))
    d[1] = sqrt(1-γdt)
    v * Diagonal(d) * v'
end
#----------------------------------------------------------------------------------------------------
"""
Correaction operator if jump does not happen
    P = d⁺d + sqrt(1-γ⋅dt)dd⁺
"""
function hole_exp(qm::QuasiMode, γdt::Real)
    v = hcat(qm.V, nullspace(qm.V'))
    d = fill(sqrt(1-γdt), length(qm.I))
    d[1] = 1
    v * Diagonal(d) * v'
end


#----------------------------------------------------------------------------------------------------
# Weak measurement
#----------------------------------------------------------------------------------------------------
function jump(qj::QJParticle, s::FreeFermionState, p::AbstractVector; renorm::Bool=true)    # L = d⁺d
    FreeFermionState(replace_vector(s.B, vector(qj.M), p; renorm), renorm)
end
function jump(qj::QJHole, s::FreeFermionState, p::AbstractVector; renorm::Bool=true)        # L = dd⁺
    FreeFermionState(avoid_vector(s.B, vector(qj.M), p; renorm), renorm)
end
function jump(qj::QJDrain, s::FreeFermionState, p::AbstractVector; renorm::Bool=true)       # L = d
    FreeFermionState(delete_vector(s.B, vector(qj.M), p; renorm), renorm)
end
function jump(qj::QJSource, s::FreeFermionState, p::AbstractVector; renorm::Bool=true)      # L = d⁺
    FreeFermionState(insert_vector(s.B, vector(qj.M); renorm), renorm)
end
#----------------------------------------------------------------------------------------------------
"""
Apply quantum jump on a free fermion state `s`

First decide whether jump happen:
    Y: return the jumped state.
    N: apply a correction `P` to `s`.
"""
function *(qj::QuantumJump, s::FreeFermionState)
    v = inner(qj.M, s)
    if rand() < real(dot(v, v)) * qj.γdt
        return jump(qj, s, v)
    else
        s.B[qj.M.I, :] .= qj.P * s.B[qj.M.I, :]
        return FreeFermionState(orthogonalize(s.B), true)
    end
end
#----------------------------------------------------------------------------------------------------
"""
Apply multiple quantum jumps on a free fermion state.

Note that indices of the jumps should have no overlap.
"""
function *(qjs::AbstractVector{<:QuantumJump}, s::FreeFermionState)
    normQ = true
    for qj in qjs
        v = inner(qj.M, s)
        if rand() < real(dot(v, v)) * qj.γdt
            s = jump(qj, s, v)
            normQ = true
        else
            s.B[qj.M.I, :] .= qj.P * s.B[qj.M.I, :]
            normQ = false
        end
    end
    normQ ? s : FreeFermionState(orthogonalize(s.B), true)
end

#----------------------------------------------------------------------------------------------------
# Conditional Jump
#----------------------------------------------------------------------------------------------------
export ConditionalJump
"""
Quantum jump with conditional feedback 

First decided if jump happens (true/false), then apply (Ut/Uf) feedback.
"""
struct ConditionalJump{T<:QuantumJump, Tm<:AbstractMatrix}
    J::T
    U::Tm
end
#----------------------------------------------------------------------------------------------------
function *(cj::ConditionalJump, s::FreeFermionState)
    v = inner(cj.J.M, s)
    ind = cj.J.M.I
    if rand() < real(dot(v, v)) * cj.J.γdt
        s = jump(cj.J, s, v)
        s.B[ind, :] .= cj.U * s.B[ind, :] 
    else
        s.B[ind, :] .= cj.J.P * s.B[ind, :]
        s = FreeFermionState(orthogonalize(s.B), true)
    end
    s
end
#----------------------------------------------------------------------------------------------------
"""
Apply a list of quantum jumps with condional feedback.

Note that indices of the jumps should have no overlap.
"""
function *(cjs::AbstractVector{<:ConditionalJump}, s::FreeFermionState)
    normQ = true
    for cj in cjs
        qj = cj.J
        ind = qj.M.I
        v = inner(qj.M, s)
        if rand() < real(dot(v, v)) * qj.γdt
            s = jump(qj, s, v)
            s.B[ind, :] .= cj.U * s.B[ind, :]
            normQ = true
        else
            s.B[ind, :] .= qj.P * s.B[ind, :]
            normQ = false
        end
    end
    normQ ? s : FreeFermionState(orthogonalize(s.B), true)
end


#----------------------------------------------------------------------------------------------------
# Gaussian SSE
#----------------------------------------------------------------------------------------------------
export wiener!
"""
Wiener process:
    ψ → exp{∑ⱼ[δWⱼ + (2⟨nⱼ⟩-1)γ δt]nⱼ} ψ
"""
function wiener!(qms::AbstractVector{<:QuasiMode}, s::FreeFermionState, γdt::Real)
    sgdt = sqrt(γdt)
    #Threads.@threads 
    for qm in qms
        p = inner(qm, s)
        a = randn() * sgdt + (2 * norm(p)^2 - 1) * γdt
        cv = (exp(a) - 1) * qm.V
        for j in axes(s.B, 2)
            s.B[qm.I, j] += cv * p[j]
        end
    end
    s.B .= orthogonalize(s.B)
end

#----------------------------------------------------------------------------------------------------
# Projective measurement
#----------------------------------------------------------------------------------------------------
export measure
"""
Projective Measure
"""
function measure(qm::QuasiMode, s::FreeFermionState)
    v = vector(qm)
    p = (v' * s.B)[:]
    if rand() < real(dot(p, p))
        B = replace_vector(s.B, v, p)
        true, FreeFermionState(B, true)
    else
        B = avoid_vector(s.B, v, p)
        false, FreeFermionState(B, true)
    end
end
#----------------------------------------------------------------------------------------------------
export ConditionalMeasure
"""
Measurement with conditional feddback

First decided the outcome of the measurement (true/false), then apply (Ut/Uf) feedback.
"""
struct ConditionalMeasure{T<:QuasiMode, Tm1<:AbstractMatrix, Tm2<:AbstractMatrix}
    M::T
    Ut::Tm1
    Uf::Tm2
end
#----------------------------------------------------------------------------------------------------
function *(cm::ConditionalMeasure, s::FreeFermionState)
    q, s2 = measure(cm.M, s)
    ind = cj.M.I
    s2.B[ind, :] .= (q ? cm.Ut : cm.Uf) * s2.B[ind, :] 
    s2
end


