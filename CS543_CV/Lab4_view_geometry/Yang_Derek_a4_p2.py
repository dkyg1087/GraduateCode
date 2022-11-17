from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt
from scipy.spatial import distance
import cv2

##
## load images and match files for the first example
##
I1 = Image.open('MP4_part2_data/lab1.jpg')
I2 = Image.open('MP4_part2_data/lab2.jpg')
matches = np.loadtxt('MP4_part2_data/lab_matches.txt')

# this is a N x 4 file where the first two numbers of each row
# are coordinates of corners in the first image and the last two
# are coordinates of corresponding corners in the second image: 
# matches(i,1:2) is a point in the first image
# matches(i,3:4) is a corresponding point in the second image

N = len(matches)

##
## display two images side-by-side with matches
## this code is to help you visualize the matches, you don't need
## to use it to produce the results for the assignment
##

I3 = np.zeros((I1.size[1],I1.size[0]*2,3) )
I3[:,:I1.size[0],:] = I1
I3[:,I1.size[0]:,:] = I2
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.imshow(np.array(I3).astype(np.uint8))
ax.plot(matches[:,0],matches[:,1],  '+r')
ax.plot( matches[:,2]+I1.size[0],matches[:,3], '+r')
ax.plot([matches[:,0], matches[:,2]+I1.size[0]],[matches[:,1], matches[:,3]], 'r')
plt.show()

##
## display second image with epipolar lines reprojected 
## from the first image
##

def fit_fundamental(matches, normalize = True):     
    # Solve homogeneous linear system using eight or more matches  
    # no need to change to homogeneous style since we reform it to svd form later 
    p1 = matches[:, 0:2]
    p2 = matches[:, 2:4]

    # normalize the points if we use normalized eight points algo
    if normalize:
        p1, T1 = normalization(p1)
        p2, T2 = normalization(p2)

    # select randomly eight points to perform the algo
    rand_idx = random.sample(range(p1.shape[0]), k=8)
    eight_p1 = p1[rand_idx]
    eight_p2 = p2[rand_idx]

    # fitting the fundamental using eight point algo to solve F matrix
    A = []
    for i in range(eight_p1.shape[0]):
        p1 = eight_p1[i]
        p2 = eight_p2[i]
        
        row = [p2[0]*p1[0], p2[0]*p1[1], p2[0], p2[1]*p1[0], p2[1]*p1[1], p2[1], p1[0], p1[1], 1]
        A.append(row)

    A = np.array(A)

    U, s, V = np.linalg.svd(A)
    F = V[len(V)-1].reshape(3, 3)
    # normalized F
    F = F / F[2, 2] 

    # Enforce rank-2 constraint.
    U, s, v = np.linalg.svd(F)
    # Vector(s) with the singular values, within each vector sorted in descending order
    s_throw_out = np.diag(s)
    # throw out the smallest value
    s_throw_out[-1] = 0
    F = np.dot(U, np.dot(s_throw_out, v))

    # recover the unnormalized F matrix
    if normalize:
        F = np.dot(np.dot(T2.T, F), T1)
    
    residual = calculate_residual(matches, F)
    #print('residual of method: ' + str(residual))
    return F,residual

def normalization(pts):
    """Helper function to normalized data in image."""
    # "Center the image data at the origin". 
    # You can do this by just subtracting the mean of the data from each point.
    mean = np.mean(pts, axis=0)
    pts_x_centered = pts[:, 0] - mean[0]
    pts_y_centered = pts[:, 1] - mean[1]

    #Scale so the mean squared distance between origin and data point is 2
    scale = sqrt(1 / (2 * len(pts)) * np.sum(pts_x_centered**2 + pts_y_centered**2))
    scale = 1 / scale

    transform = np.array([[scale, 0, -scale*mean[0]], 
                           [0, scale, -scale*mean[1]], 
                           [0, 0, 1]])
    # do homogeneous transform
    pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
    normalized = np.dot(transform, pts.T).T

    return normalized[:, 0:2], transform

def calculate_residual(matches, F):
    p1 = matches[:, 0:2]
    p2 = matches[:, 2:4]
    p1_homo = np.concatenate((p1, np.ones((p1.shape[0], 1))), axis=1)
    p2_homo = np.concatenate((p2, np.ones((p2.shape[0], 1))), axis=1)

    residual = 0
    for i in range(p1.shape[0]):
        residual += abs(np.dot(np.dot(p2_homo[i], F), p1_homo[i].T))

    residual = residual / matches.shape[0]
    return residual

## Camera Calibration

def evaluate_points(M, points_2d, points_3d):
    """
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to
    :param M: projection matrix 3 x 4
    :param points_2d: 2D points N x 2
    :param points_3d: 3D points N x 3
    :return:
    """
    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual

def camera_calibration(points_3D, points_2D):
    A = np.zeros((len(points_3D) * 2, 12))

    for i in range(len(points_3D)):
        temp = np.concatenate((points_3D[i], [1]), axis = 0)
        A[2*i, 4:8] = temp
        A[2*i, 8:12] = -points_2D[i, 1] * temp
        A[2*i+1, 0:4] = temp
        A[2*i+1, 8:12] = -points_2D[i, 0] * temp

    _, _, v = np.linalg.svd(A)
    A = v[len(v)-1].reshape(3, 4)

    return A

## Camera Centers

def camera_center(M):
    _, _, v = np.linalg.svd(M)
    center = v[-1,:]
    center /= center[-1]

    return center

## Triangulation

def triangulation(matches, proj_1, proj_2):
    N = len(matches)
    pts_3d = np.zeros((N, 3))
    for i in range(N):
        x1, y1 = matches[i, :2]
        x2, y2 = matches[i, 2:]
        r1 = np.array([[0, -1, y1],[1, 0, -x1],[-y1, x1, 0]]).dot(proj_1)
        r2 = np.array([[0, -1, y2],[1, 0, -x2],[-y2, x2, 0]]).dot(proj_2)
        A = np.vstack((r1, r2))
        _, _, v = np.linalg.svd(A)
        p = v[-1,:]
        p /= p[-1]
        pts_3d[i] = p[:3]
    
    return pts_3d 


def plot_inlier_matches(ax, img1, img2, inliers):
    """
    Plot the matches between two images according to the matched keypoints
    :param ax: plot handle
    :param img1: left image
    :param img2: right image
    :inliers: x,y in the first image and x,y in the second image (Nx4)
    """
    res = np.hstack([img1, img2])
    ax.set_aspect('equal')
    ax.imshow(res, cmap='gray')
    
    ax.plot(inliers[:,0], inliers[:,1], '+r')
    ax.plot(inliers[:,2] + img1.shape[1], inliers[:,3], '+r')
    ax.plot([inliers[:,0], inliers[:,2] + img1.shape[1]],
            [inliers[:,1], inliers[:,3]], 'r', linewidth=0.4)
    ax.axis('off')

def ransac(match_coords, ransac_threshold):
    max_iters = 5000
    max_inlier = 0
    best_inlier = None

    for i in range(max_iters):
        F,_ = fit_fundamental(match_coords)
        ones = np.ones((match_coords.shape[0], 1))
        all_p1 = np.concatenate((match_coords[:, 0:2], ones), axis=1)
        all_p2 = np.concatenate((match_coords[:, 2:4], ones), axis=1)
        F_p1 = np.dot(F, all_p1.T).T
        F_p2 = np.dot(F.T, all_p2.T).T
        p1_line2 = np.sum(all_p1 * F_p2, axis=1)[:, np.newaxis]
        p2_line1 = np.sum(all_p2 * F_p1, axis=1)[:, np.newaxis]
        d1 = np.absolute(p1_line2) / np.linalg.norm(F_p2, axis=1)[:, np.newaxis]
        d2 = np.absolute(p2_line1) / np.linalg.norm(F_p1, axis=1)[:, np.newaxis]
        error = (d1 + d2) / 2
        idx = np.where(error < ransac_threshold)[0]
        inliers = match_coords[idx]
        inlier_count = len(inliers)
        if inlier_count > max_inlier:
            max_inlier = inlier_count
            best_inlier = inliers.copy()
            best_F = F.copy()
            residual = error[idx].sum() / max_inlier
    
    return best_inlier, best_F, residual

def get_matches(img1, img2, dist_threshold):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    pair_dist = distance.cdist(des1, des2, 'sqeuclidean')
    data = []
    tmp = []
    h, w = pair_dist.shape
    for i in range(h):
        for j in range(w):
            if pair_dist[i][j] <= dist_threshold:
                tmp = list(kp1[i].pt + kp2[j].pt)
                data.append(tmp)
    data = np.array(data)
    return data

def get_residual(F, p1, p2):
    P1 = np.c_[p1, np.ones((p1.shape[0],1))].transpose()
    P2 = np.c_[p2, np.ones((p2.shape[0],1))].transpose()
    L2 = np.matmul(F, P1).transpose()
    L2_norm = np.sqrt(L2[:,0]**2 + L2[:,1]**2)
    L2 = L2 / L2_norm[:,np.newaxis]
    pt_line_dist = np.multiply(L2, P2.T).sum(axis = 1)
    return np.mean(np.square(pt_line_dist))

for normalize in ["True", "False"]:
    # first, fit fundamental matrix to the matches
    F,r = fit_fundamental(matches,normalize); # this is a function that you should write
    print('residual of method: ' + str(r))
    M = np.c_[matches[:,0:2], np.ones((N,1))].transpose()
    L1 = np.matmul(F, M).transpose() # transform points from 
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:,0]**2 + L1[:,1]**2)
    L = np.divide(L1,np.kron(np.ones((3,1)),l).transpose())# rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:,2:4], np.ones((N,1))]).sum(axis = 1)
    closest_pt = matches[:,2:4] - np.multiply(L[:,0:2],np.kron(np.ones((2,1)), pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:,1], -L[:,0]]*10# offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:,1], -L[:,0]]*10

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I2).astype(np.uint8))
    ax.plot(matches[:,2],matches[:,3],  '+r')
    ax.plot([matches[:,2], closest_pt[:,0]],[matches[:,3], closest_pt[:,1]], 'r')
    ax.plot([pt1[:,0], pt2[:,0]],[pt1[:,1], pt2[:,1]], 'g')
    plt.show()

matches = np.loadtxt('MP4_part2_data/lab_matches.txt')
pts_3d = np.loadtxt('MP4_part2_data/lab_3d.txt')
lab1_proj = camera_calibration(pts_3d, matches[:,:2])
lab2_proj = camera_calibration(pts_3d, matches[:,2:])
print(lab1_proj)
print(lab2_proj)

proj1, residual1 = evaluate_points(lab1_proj, matches[:,:2], pts_3d)
proj2, residual2 = evaluate_points(lab2_proj, matches[:,2:], pts_3d)
print(residual1)
print(residual2)

lib1_proj = np.loadtxt('MP4_part2_data/library1_camera.txt')
lib2_proj = np.loadtxt('MP4_part2_data/library2_camera.txt')

lab1_c = camera_center(lab1_proj)
lab2_c = camera_center(lab2_proj)
print(lab1_c)
print(lab2_c)

lib1_c = camera_center(lib1_proj)
lib2_c = camera_center(lib2_proj)
print(lib1_c)
print(lib2_c)

lab_matches = np.loadtxt('MP4_part2_data/lab_matches.txt')
points_3d_gt = np.loadtxt('MP4_part2_data/lab_3d.txt')
lab_3d = triangulation(lab_matches, lab1_proj, lab2_proj)
#print(lab_3d)
lab_3d_pts = np.sum((lab_3d - points_3d_gt)**2, axis=1)
print('Mean 3D reconstuction error for the lab data: ', round(np.mean(lab_3d_pts), 5))
_, lab1_2d_pts = evaluate_points(lab1_proj, lab_matches[:, :2], lab_3d)
_, lab2_2d_pts = evaluate_points(lab2_proj, lab_matches[:, 2:], lab_3d)
print('2D reprojection error for the lab 1 data: ', np.mean(lab1_2d_pts))
print('2D reprojection error for the lab 2 data: ', np.mean(lab2_2d_pts))
camera_centers = np.vstack((lab1_c, lab2_c))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(lab1_c[0], lab1_c[1], lab1_c[2])
ax.scatter(lab2_c[0], lab2_c[1], lab2_c[2])
ax.scatter(lab_3d[:, 0], lab_3d[:, 1], lab_3d[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
fig.savefig('lab_triangulation.jpg')

library_matches = np.loadtxt('MP4_part2_data/library_matches.txt')
library_3d = triangulation(library_matches, lib1_proj, lib2_proj)
#print(library_3d)
_, library1_2d_pts = evaluate_points(lib1_proj, library_matches[:, :2], library_3d)
_, library2_2d_pts = evaluate_points(lib2_proj, library_matches[:, 2:], library_3d)
print('2D reprojection error for the library 1 data: ', np.mean(library1_2d_pts))
print('2D reprojection error for the library 2 data: ', np.mean(library2_2d_pts))
camera_centers = np.vstack((lib1_c, lib2_c))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(lib1_c[0], lib1_c[1], lib1_c[2])
ax.scatter(lib2_c[0], lib2_c[1], lib2_c[2])
ax.scatter(library_3d[:, 0], library_3d[:, 1], library_3d[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(azim=-45, elev=45)
fig.savefig('library_triangulation.jpg')

house1 = cv2.imread('MP4_part2_data/house1.jpg')
house2 = cv2.imread('MP4_part2_data/house2.jpg')
house1_grey = cv2.cvtColor(house1, cv2.COLOR_RGB2GRAY)
house2_grey = cv2.cvtColor(house2, cv2.COLOR_RGB2GRAY)

matched_coords = get_matches(house1_grey, house2_grey, 20000)
inliers, F, residual = ransac(matched_coords, 150)
fig, ax = plt.subplots(figsize=(20,10))
plot_inlier_matches(ax, house1, house2, inliers)
print("Average residual:", residual)
print("Number of inliers:", len(inliers))
fig.savefig('ransac_match_house.jpg', bbox_inches='tight')

gaudi1 = cv2.imread('MP4_part2_data/gaudi1.jpg')
gaudi2 = cv2.imread('MP4_part2_data/gaudi2.jpg')
gaudi1_grey = cv2.cvtColor(gaudi1, cv2.COLOR_RGB2GRAY)
gaudi2_grey = cv2.cvtColor(gaudi2, cv2.COLOR_RGB2GRAY)

matched_coords = get_matches(gaudi1_grey, gaudi2_grey, 15000)
inliers, F, residual = ransac(matched_coords, 150)
fig, ax = plt.subplots(figsize=(20,10))
plot_inlier_matches(ax, gaudi1_grey, gaudi2_grey, inliers)
print("Average residual:", residual)
print("Number of inliers:", len(inliers))
fig.savefig('ransac_match_gaudi.jpg', bbox_inches='tight')