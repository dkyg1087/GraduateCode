import matplotlib.pyplot as plt
import cv2
import numpy as np
import numpy.matlib

dataDir = 'factorization_data/measurement_matrix.txt'
with open(dataDir) as file:
    lines = file.readlines()
matrix = np.array([line.rstrip().split(' ') for line in lines]).astype(float)

# Load the data matrix and normalize the point coordinates by translating them to the mean of the points in each view

m, n = int(matrix.shape[0] / 2), matrix.shape[1]
mean = numpy.matlib.repmat(np.sum(matrix,1)/matrix.shape[1], matrix.shape[1], 1).T
D = matrix - mean

# Apply SVD to the 2M x N data matrix to express it as D = U @ W @ V' (using NumPy notation) where U is a 2Mx3 matrix, 
# W is a 3x3 matrix of the top three singular values, and V is a Nx3 matrix. 
# You can use numpy.linalg.svd to compute this decomposition. 
# Next, derive structure and motion matrices from the SVD as explained in the lecture.

U, W, V = np.linalg.svd(D)
U = U[:,0:3]
W = np.diag(W)[:3,:3]
V = V[0:3,:]

M = U @ np.sqrt(W)
S = np.sqrt(W) @ V

#Find the matrix Q to eliminate the affine ambiguity using the method described on slide 32 of the lecture.

#below method refferenced from Evan Luo(evanluo2)

M_i,M_j = M[:m, :],M[m:, :]
def A_p(X, Y, m):
    A = []
    for i in range(m):
        [a, b, c] = X[i, :]
        [x, y, z] = Y[i, :]
        A.append([a*x, 2*a*y, 2*a*z, b*y, 2*b*z, c*z])
    return A

A = np.vstack((A_p(M_i, M_i,m), A_p(M_j, M_j,m), A_p(M_i, M_j,m)))
B = np.vstack((np.ones((m, 1)),np.ones((m, 1)),np.zeros((m, 1)))).squeeze()

L = np.linalg.lstsq(A, B,rcond=None)[0]
Q_Qt = [[L[0], L[1], L[2]],
       [L[1], L[3], L[4]],
       [L[2], L[4], L[5]]]

#---end of refferencing---

Q = np.linalg.cholesky(Q_Qt)

print(Q)

M = M @ Q
S = np.linalg.pinv(Q) @ S


X, Y, Z = S[0, :], S[1, :], S[2, :]
Z = -Z
ax = plt.axes(projection='3d')
ax.scatter(X, Y, Z, color="b")
plt.show()


# Display three frames with both the observed feature points and the estimated projected 3D points overlayed. 
# Report your total residual (sum of squared Euclidean distances, in pixels, 
# between the observed and the reprojected features) over all the frames, 
# and plot the per-frame residual as a function of the frame number.

De = M @ S
residual = []
x,y = 0,1
while y < 2*m+1: #sum of squared Euclidean distances
    sum_r = 0
    for j in range(n):
        sum_r += (De[x, j] - D[x, j])**2 + (De[y, j] - D[y, j])**2
    residual.append(sum_r)
    x+=2
    y = x + 1
print(sum(residual))
plt.plot(residual)
plt.savefig("residual.jpg")
plt.clf()

for frame in [40, 45, 50]:
    ax = plt.axes()
    img = cv2.imread('factorization_data/frame000000'+str(frame)+'.jpg')
    ax.imshow(img)
    frame *= 2
    ax.scatter(matrix[frame, :], matrix[frame+1, :], color='r')
    ax.scatter(De[frame,:]+mean[frame,:], De[frame+1,:]+mean[frame+1,:],marker='^', edgecolors='g')
    plt.savefig("frame"+str(frame/2)+'.jpg')
    plt.clf()