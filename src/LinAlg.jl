#----------------------------------------------------------------------------------------------------
# Evolution under unitary matrices
#----------------------------------------------------------------------------------------------------
export evo_operator
"""
    evo_operator(H::Hermitian, dt::Real)

Exponential for Hermitian matrix.
"""
function evo_operator(H::Hermitian, dt::Real)
    vals, vecs = eigen(H)
    D = exp.(-dt * im * vals) |> Diagonal
    vecs * D * vecs'
end
#----------------------------------------------------------------------------------------------------
"""
    rand_unitary(N::Integer)

Generate random unitary matrix. 
"""
function rand_unitary(N::Integer)
    F = qr(randn(ComplexF64, N, N)) 
    Matrix(F.Q)
end
#----------------------------------------------------------------------------------------------------
function turbo_mul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    @turbo warn_check_args=false for n ∈ indices((C,B), 2), m ∈ indices((C,A), 1)
        Cmn = zero(eltype(C))
        for k ∈ indices((A,B), (2,1))
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] = Cmn
    end
end
#----------------------------------------------------------------------------------------------------
function turbo_mul!(C::AbstractVector, A::AbstractMatrix, B::AbstractVector)
    @turbo warn_check_args=false for n ∈ eachindex(C)
        Cn = zero(eltype(C))
        for k ∈ eachindex(B)
            Cn += A[n,k] * B[k]
        end
        C[n] = Cn
    end
end
#----------------------------------------------------------------------------------------------------
function tturbo_mul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    @tturbo warn_check_args=false for n ∈ indices((C,B), 2), m ∈ indices((C,A), 1)
        Cmn = zero(eltype(C))
        for k ∈ indices((A,B), (2,1))
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] = Cmn
    end
end
#----------------------------------------------------------------------------------------------------
function turbo_dot!(C::AbstractVector, v::AbstractVector, B::AbstractMatrix, inds::AbstractVector{<:Integer})
    @turbo warn_check_args=false for n in eachindex(C)
        Cn = zero(eltype(C))
        for (i, k) in enumerate(inds)
            Cn += conj(v[i]) * B[k, n]
        end
        C[n] = Cn
    end
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
function delete_vector(A::AbstractMatrix, vA::AbstractVector)
    mat = Matrix{eltype(vA)}(undef, size(A, 1), size(A, 2)-1)
    p = argmax(abs.(vA))
    Ap = A[:, p] / vA[p]
    inner_itr = eachindex(Ap)
    @tturbo warn_check_args=false for i in 1:p-1
        for j in inner_itr
            mat[j, i] = A[j, i] - vA[i] * Ap[j]
        end
    end
    @tturbo warn_check_args=false for i in p+1:size(A, 2)
        for j in inner_itr
            mat[j, i-1] = A[j, i] - vA[i] * Ap[j]
        end
    end
    mat
end
#----------------------------------------------------------------------------------------------------
function replace_vector!(A::AbstractMatrix, v::AbstractVector, vA::AbstractVector)
    p = argmax(abs.(vA))
    Ap = A[:, p] / vA[p]
    A[:, p] = v
    inner_itr = eachindex(Ap)
    @tturbo warn_check_args=false for i in 1:p-1
        for j in inner_itr
            A[j, i] -= vA[i] * Ap[j]
        end
    end
    @tturbo warn_check_args=false for i in p+1:size(A, 2)
        for j in inner_itr
            A[j, i] -= vA[i] * Ap[j]
        end
    end
    A
end
#----------------------------------------------------------------------------------------------------
"""
Compute null vectors:
    |A'ᵢ⟩ = |Aᵢ⟩ - ⟨v|Aᵢ⟩|v⟩
"""
function avoid_vector!(A::AbstractMatrix, v::AbstractVector, vA::AbstractVector)
    inner_itr = eachindex(v)
    @tturbo warn_check_args=false for i in axes(A, 2)
        for j in inner_itr
            A[j, i] -= vA[i] * v[j]
        end
    end
    A
end
#--------------------------------------------------------------------------------
function insert_vector(A::AbstractMatrix, v::AbstractVector)
    mat = hcat(v, A)
    mat
end
