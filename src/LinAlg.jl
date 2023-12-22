#----------------------------------------------------------------------------------------------------
# Evolution under unitary matrices
#----------------------------------------------------------------------------------------------------
"""
Data type for Unitary matrices
"""
struct Unitary{T <: AbstractMatrix} 
    M::T
end
#----------------------------------------------------------------------------------------------------
export evo_operator
"""
    evo_operator(H::Hermitian, dt::Real)

Exponential for Hermitian matrix.
"""
function evo_operator(H::Hermitian, dt::Real)
    vals, vecs = eigen(H)
    D = exp.(-dt * im * vals) |> Diagonal
    Unitary(vecs * D * vecs')
end
#----------------------------------------------------------------------------------------------------
"""
    evo_operator(H::AbstractMatrix, dt::Real; order::Integer=10)

Exponential for general matrix using series expansion.
"""
function evo_operator(H::AbstractMatrix, dt::Real; order::Integer=10)
    N = round(Int, 10 * dt * maximum(abs, H))
    iszero(N) && (N = 1)
    expm(-dt*im*H/N, order)^N
end
#----------------------------------------------------------------------------------------------------
"""
    expm(A::AbstractMatrix, order::Integer=10)

Matrix exponential using Taylor expansion.
"""
function expm(A::AbstractMatrix, order::Integer=10)
    mat = I + A / order
    order -= 1
    while order > 0
        mat = A * mat
        mat ./= order
        mat += I
        order -= 1
    end
    mat
end
#----------------------------------------------------------------------------------------------------
"""
    expv(A, v::AbstractVecOrMat, order::Integer=10)

Compute exp(A)*v using Taylor expansion
"""
function expv(A, v::AbstractVecOrMat, order::Integer=10)
    vec = v + A * v / order
    order -= 1
    while order > 0
        vec = A * vec
        vec ./= order
        vec += v
        order -= 1
    end
    vec
end
#----------------------------------------------------------------------------------------------------
export rand_unitary
"""
    rand_unitary(N::Integer)

Generate random unitary matrix. 
"""
function rand_unitary(N::Integer)
    F = qr(randn(ComplexF64, N,N)) 
    Unitary(Matrix(F.Q))
end


#----------------------------------------------------------------------------------------------------
# Helper
#----------------------------------------------------------------------------------------------------
"""
Given a vector space spanned by column of `A`, and a vector `v`, find the nullspace 
of `v` inside A. The inpute `vA` is the pre-computed inner product ⟨v|Aᵢ⟩.

The algorithm first choose the pivot vector |Aₚ⟩ with largest overlap ‖⟨v|Aₚ⟩‖ with
vector `v`, then calculate the vectors 
    |Ãᵢ⟩ := |Aᵢ⟩ - (⟨v|Aᵢ⟩/⟨v|Aₚ⟩) * |Aₚ⟩,  i ≠ p.
The set of vectors {|Ãᵢ⟩} span the null space of `v`, though not orthogonal.
"""
function delete_vector(A::AbstractMatrix, v::AbstractVector, vA::AbstractVector; renorm::Bool=true)
    mat = Matrix{eltype(vA)}(undef, size(A, 1), size(A, 2)-1)
    p = argmax(abs.(vA))
    Ap = A[:, p] / vA[p]
    mat[:, 1] .= v 
    for i = 1:p-1
        mat[:, i] .= A[:, i] - vA[i] * Ap
    end
    for i = p+1:size(A, 2)
        mat[:, i-1] .= A[:, i] - vA[i] * Ap
    end
    renorm ? orthogonalize(mat) : mat
end
#----------------------------------------------------------------------------------------------------
function replace_vector(A::AbstractMatrix, v::AbstractVector, vA::AbstractVector; renorm::Bool=true)
    mat = Matrix{eltype(vA)}(undef, size(A, 1), size(A, 2))
    mat[:, 1] .= v 
    mat[:, 2:end] .= delete_vector(A, v, vA; renorm)
    mat
end
#----------------------------------------------------------------------------------------------------
"""
Compute null vectors:
    |A'ᵢ⟩ = |Aᵢ⟩ - ⟨v|Aᵢ⟩|v⟩
"""
function avoid_vector(A::AbstractMatrix, v::AbstractVector, vA::AbstractVector; renorm::Bool=true)
    mat = Matrix{eltype(vA)}(undef, size(A, 1), size(A, 2))
    for i in axes(A, 2)
        mat[:, i] .= A[:, i] - vA[i] * v
    end
    renorm ? orthogonalize(mat) : mat
end
#--------------------------------------------------------------------------------
function insert_vector(A::AbstractMatrix, v::AbstractVector; renorm::Bool=true)
    mat = hcat(v, A)
    renorm ? orthogonalize(mat) : mat
end
