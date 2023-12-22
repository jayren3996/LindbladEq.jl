# LindbladEq.jl
Simulation of Lindblad equation.

## Closed Hierarchy

For Lindblad equation
$$
\frac{d}{dt} \rho = -i[H,\rho] + \sum_{\mu=1}^{m} L_\mu \hat\rho L_\mu^\dagger -\frac{1}{2} \sum_{\mu=1}^{m} \{ L_\mu^\dagger L_\mu, \rho\}
$$
When the jump operator $L_\mu$ contains only the linear Majorana operator, the Lindblad equation preserves Gaussianity.  For jump operators containing up to quadratic Majorana terms, the evolution will break the Gaussian form, however, the $2n$-point correlation is still solvable for free fermion systems.

We assume that the jump operator has up to quadratic Majorana terms. In particular, we denote the linear terms and the Hermitian quadratic terms as
$$
L_r = \sum_{j=1}^{2N} L^r_{j} \omega_j, \quad
L_s = \sum_{j,k=1}^{2N} M^s_{jk} \omega_j \omega_k.
$$
We get the EOM of the variance matrix $\Gamma_{ij}(t)=i\langle\hat O_{ij}\rangle_t$:
$$
\partial_t \Gamma = \mathcal{M}[\Gamma]=X^T\cdot\Gamma + \Gamma \cdot X + \sum_s (Z^s)^T \cdot \Gamma\cdot Z^s + Y,
$$
where
$$
X = H - 2B^R + 8 \sum_s (\mathrm{Im} M^s)^2, \quad
	Y = 4B^I, \quad 
	Z = 4 \mathrm{Im} M^s.
$$
The nonequilibrium steady state is solved by $\mathcal{M}[\Gamma]=0$.
