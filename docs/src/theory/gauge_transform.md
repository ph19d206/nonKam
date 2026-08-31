# Gauge Transformation & Unitary Equivalence

The transformation between the real symmetric operator $A_0$ and the phase-dependent operator $A_{\omega\tau}$ is a **unitary similarity transformation** (gauge transformation).

## 1. Operator Formulations

Let $A_0$ be the real quadrature operator with $\tau = 0$:

$$A_0 = \frac{1}{\sqrt{2\omega}} \left( a + a^\dagger \right)$$

In the truncated Fock basis $\{|n\rangle\}_{n=0}^{N-1}$, $A_0$ has matrix elements:

$$\langle m | A_0 | n \rangle = \frac{1}{\sqrt{2\omega}} \left( \sqrt{n}\,\delta_{m, n-1} + \sqrt{n+1}\,\delta_{m, n+1} \right)$$

This corresponds to a `SymTridiagonal` matrix with zero diagonal and $e_n = \sqrt{n} / \sqrt{2\omega}$.

The phase-dependent operator is:

$$A_{\omega\tau} = \frac{1}{\sqrt{2\omega}} \left( e^{-i\omega\tau} a + e^{i\omega\tau} a^\dagger \right)$$

## 2. Unitary Phase Matrix

Define the diagonal phase rotation matrix $U$:

$$U = \operatorname{diag}\left(1, e^{-i\omega\tau}, e^{-2i\omega\tau}, \dots, e^{-i(N-1)\omega\tau}\right)$$

where $U_{m,n} = e^{-in\omega\tau}\delta_{m,n}$.

## 3. Proof of Similarity ($A_{\omega\tau} = U A_0 U^\dagger$)

$$\begin{aligned}
(U A_0 U^\dagger)_{m,n} &= \sum_{j,k} U_{m,j} (A_0)_{j,k} (U^\dagger)_{k,n} \\
&= e^{-i(m-n)\omega\tau} (A_0)_{m,n}
\end{aligned}$$

Evaluating non-zero entries:
* **Superdiagonal** ($m = n - 1$): $(U A_0 U^\dagger)_{n-1,n} = e^{i\omega\tau} (A_0)_{n-1,n}$
* **Subdiagonal** ($m = n + 1$): $(U A_0 U^\dagger)_{n+1,n} = e^{-i\omega\tau} (A_0)_{n+1,n}$

Thus, $A_{\omega\tau} = U A_0 U^\dagger$ exactly.

## 4. Eigenvalue & Eigenvector Mapping

For $A_0 \mathbf{v}_{\text{real}} = \lambda \mathbf{v}_{\text{real}}$, substituting $A_0 = U^\dagger A_{\omega\tau} U$ yields:

$$A_{\omega\tau} (U \mathbf{v}_{\text{real}}) = \lambda (U \mathbf{v}_{\text{real}})$$

* **Eigenvalues:** Unchanged ($\lambda$).
* **Eigenvectors:** $\mathbf{v}_{\text{complex}} = U \mathbf{v}_{\text{real}}$.

In Julia, multiplying row $n$ of $V_{\text{real}}$ by $e^{-in\omega\tau}$ executes $U \mathbf{v}_{\text{real}}$ exactly.

---

## API Reference