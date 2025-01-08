import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14  # That's 1×10⁻¹⁴ = 0.00000000000001

def gsBasis4(A):
    B = np.array(A, dtype=np.float64)
    # The zeroth column has no other vectors to make it orthogonal to.
    B[:, 0] = B[:, 0] / la.norm(B[:, 0])

    # For the first column, subtract the projection onto the zeroth vector.
    B[:, 1] = B[:, 1] - B[:, 1] @ B[:, 0] * B[:, 0]
    if la.norm(B[:, 1]) > verySmallNumber:
        B[:, 1] = B[:, 1] / la.norm(B[:, 1])
    else:
        B[:, 1] = np.zeros_like(B[:, 1])

    # Column 2: Subtract overlaps with columns 0 and 1.
    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 0] * B[:, 0]
    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 1] * B[:, 1]
    if la.norm(B[:, 2]) > verySmallNumber:
        B[:, 2] = B[:, 2] / la.norm(B[:, 2])
    else:
        B[:, 2] = np.zeros_like(B[:, 2])

    # Column 3: Subtract overlaps with columns 0, 1, and 2.
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 0] * B[:, 0]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 1] * B[:, 1]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 2] * B[:, 2]
    if la.norm(B[:, 3]) > verySmallNumber:
        B[:, 3] = B[:, 3] / la.norm(B[:, 3])
    else:
        B[:, 3] = np.zeros_like(B[:, 3])

    return B

def gsBasis(A):
    B = np.array(A, dtype=np.float64)
    for i in range(B.shape[1]):  # Loop over all vectors, labeled by i
        for j in range(i):  # Loop over all previous vectors, labeled by j
            B[:, i] = B[:, i] - B[:, i] @ B[:, j] * B[:, j]  # Subtract projection onto B[:, j]
        if la.norm(B[:, i]) > verySmallNumber:  # Normalization condition
            B[:, i] = B[:, i] / la.norm(B[:, i])
        else:
            B[:, i] = np.zeros_like(B[:, i])  # Set to zero if norm is too small

    return B

def dimensions(A):
    return np.sum(la.norm(gsBasis(A), axis=0))

# Example usage:
A = np.array([[1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0]], dtype=float)
gs_result = gsBasis(A)
print("Orthonormal basis:")
print(gs_result)
print("Dimension spanned by the vectors:", dimensions(A))

C = np.array([[1,0,2],
              [0,1,-3],
              [1,0,2]], dtype=np.float64)
print("Orthonormal basis:")
print(gsBasis(C))
print("Dimension spanned by the vectors:", dimensions(C))# because we have one vector
                                                         # that is a linear combination of the others
