# imports

import os
import sys
import glob
import re
import numpy as np
import random
import time
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


#####################################
### Provided functions start here ###
#####################################

# Image loading and saving

def LoadFaceImages(pathname, subject_name, num_images):
    """
    Load the set of face images.  
    The routine returns
        ambimage: image illuminated under the ambient lighting
        imarray: a 3-D array of images, h x w x Nimages
        lightdirs: Nimages x 3 array of light source directions
    """

    def load_image(fname):
        return np.asarray(Image.open(fname))

    def fname_to_ang(fname):
        yale_name = os.path.basename(fname)
        return int(yale_name[12:16]), int(yale_name[17:20])

    def sph2cart(az, el, r):
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    ambimage = load_image(
        os.path.join(pathname, subject_name + '_P00_Ambient.pgm'))
    im_list = glob.glob(os.path.join(pathname, subject_name + '_P00A*.pgm'))
    if num_images <= len(im_list):
        im_sub_list = np.random.choice(im_list, num_images, replace=False)
    else:
        print(
            'Total available images is less than specified.\nProceeding with %d images.\n'
            % len(im_list))
        im_sub_list = im_list
    im_sub_list.sort()
    imarray = np.stack([load_image(fname) for fname in im_sub_list], axis=-1)
    Ang = np.array([fname_to_ang(fname) for fname in im_sub_list])

    x, y, z = sph2cart(Ang[:, 0] / 180.0 * np.pi, Ang[:, 1] / 180.0 * np.pi, 1)
    lightdirs = np.stack([y, z, x], axis=-1)
    return ambimage, imarray, lightdirs

def save_outputs(subject_name, albedo_image, surface_normals):
    im = Image.fromarray((albedo_image*255).astype(np.uint8))
    im.save("%s_albedo.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,0]*128+128).astype(np.uint8))
    im.save("%s_normals_x.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,1]*128+128).astype(np.uint8))
    im.save("%s_normals_y.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,2]*128+128).astype(np.uint8))
    im.save("%s_normals_z.jpg" % subject_name)


# Plot the height map

def set_aspect_equal_3d(ax):
    """https://stackoverflow.com/questions/13685386"""
    """Fix equal aspect bug for 3D plots."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)
    plot_radius = max([
        abs(lim - mean_)
        for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
        for lim in lims
    ])
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def display_output(albedo_image, height_map):
    fig = plt.figure()
    plt.imshow(albedo_image, cmap='gray')
    plt.axis('off')
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(20, 20)
    X = np.arange(albedo_image.shape[0])
    Y = np.arange(albedo_image.shape[1])
    X, Y = np.meshgrid(Y, X)
    H = np.flipud(np.fliplr(height_map))
    A = np.flipud(np.fliplr(albedo_image))
    A = np.stack([A, A, A], axis=-1)
    ax.xaxis.set_ticks([])
    ax.xaxis.set_label_text('Z')
    ax.yaxis.set_ticks([])
    ax.yaxis.set_label_text('X')
    ax.zaxis.set_ticks([])
    ax.yaxis.set_label_text('Y')
    surf = ax.plot_surface(
        H, X, Y, rcount=100, ccount=100, cmap='gray', facecolors=A, linewidth=0, antialiased=False)
    set_aspect_equal_3d(ax)
    plt.show()


# Plot the surface normals

def plot_surface_normals(surface_normals):
    """
    surface_normals: h x w x 3 matrix.
    """
    fig = plt.figure()
    ax = plt.subplot(1, 3, 1)
    ax.axis('off')
    ax.set_title('X')
    im = ax.imshow(surface_normals[:,:,0])
    ax = plt.subplot(1, 3, 2)
    ax.axis('off')
    ax.set_title('Y')
    im = ax.imshow(surface_normals[:,:,1])
    ax = plt.subplot(1, 3, 3)
    ax.axis('off')
    ax.set_title('Z')
    im = ax.imshow(surface_normals[:,:,2])


#######################################
### Your implementation starts here ###
#######################################

def preprocess(ambimage, imarray):
    """
    preprocess the data: 
        1. subtract ambient_image from each image in imarray.
        2. make sure no pixel is less than zero.
        3. rescale values in imarray to be between 0 and 1.
    Inputs:
        ambimage: h x w
        imarray: h x w x Nimages
    Outputs:
        processed_imarray: h x w x Nimages
    """
    #* Done
    imarray_p = imarray - ambimage[:,:,np.newaxis]    
    imarray_p[imarray_p < 0] = 0
    imarray_p = imarray_p / 255.0
    return imarray_p



def photometric_stereo(imarray, light_dirs , approach="single"):
    """
    Inputs:
        imarray:  h x w x Nimages
        light_dirs: Nimages x 3
    Outputs:
        albedo_image: h x w
        surface_norms: h x w x 3
    """
    #* Done
    
    h, w, n = imarray.shape
    
    if approach == "each": 
        albedo_image = np.zeros((h, w))
        surface_normals = np.zeros((h, w, 3))
        
        for x in range(h):
            for y in range(w):
                
                g = np.linalg.lstsq(light_dirs, imarray[x, y, :], rcond = -1)[0]
                
                albedo_image[x][y] = np.linalg.norm(g, axis=0)

                surface_normals[x, y, :] = g / albedo_image[x][y]
            
        return albedo_image, surface_normals
    else:
        
        imarray = imarray.reshape(h*w, n).transpose()
    

        results = np.linalg.lstsq(light_dirs, imarray,rcond = -1)
        g = results[0]


        albedo_image = np.linalg.norm(g, axis=0)
        surface_normals = g / albedo_image


        surface_normals = surface_normals.transpose().reshape(h, w, 3)
        albedo_image = albedo_image.reshape(h, w)

        return albedo_image, surface_normals



def random_path(surface_normals,loops,fx,fy):

        h,w= surface_normals.shape[:2]
        
        height_map = np.zeros((h, w))
        
        for y in range(h):
            for x in range(w):
                
                if x != 0 or y != 0:
                    
                    for _ in range(loops):
                        zeros = [0] * x
                        ones = [1] * y
                        bit_stream = np.array(zeros + ones)

                        np.random.shuffle(bit_stream)
                        
                        current_x = 0
                        current_y = 0
                        cumsum = 0

                        for step in bit_stream:
                            if step == 0:
                                cumsum += fx[current_y, current_x]
                                current_x += 1
                            else:
                                cumsum += fy[current_y, current_x]
                                current_y += 1
                        
                        height_map[y, x] += cumsum
                    
                    height_map[y, x] = height_map[y, x]/loops
        return height_map
        

def get_surface(surface_normals, integration_method):
    """
    Inputs:
        surface_normals:h x w x 3
        integration_method: string in ['average', 'column', 'row', 'random']
    Outputs:
        height_map: h x w
    """
    h, w, n = surface_normals.shape

    fx = surface_normals[:, :, 0] / surface_normals[:, :, 2]
    fy = surface_normals[:, :, 1] / surface_normals[:, :, 2]
    
    row_sum_x = np.cumsum(fx, axis=1)
    col_sum_y = np.cumsum(fy, axis=0)
    
    if integration_method == 'row':
        return row_sum_x[0] + col_sum_y
    elif integration_method == 'column':
        return col_sum_y[:, 0][:, np.newaxis] + row_sum_x
    elif integration_method == 'average':
        return (col_sum_y[:, 0][:, np.newaxis] + row_sum_x + row_sum_x[0] + col_sum_y) / 2
    else:
        return random_path(surface_normals,25,fx,fy)



# Main function
if __name__ == '__main__':
    root_path = 'data/croppedyale/'
    subject_name = 'yaleB07'
    integration_method = 'row'
    save_flag = False

    full_path = '%s%s' % (root_path, subject_name)
    ambient_image, imarray, light_dirs = LoadFaceImages(full_path, subject_name,
                                                        64)

    processed_imarray = preprocess(ambient_image, imarray)

    albedo_image, surface_normals = photometric_stereo(processed_imarray,
                                                    light_dirs,"single")

    st = time.time()
    height_map = get_surface(surface_normals, 'average')
    et = time.time()
    
    print(et-st)
    
    if save_flag:
        save_outputs(subject_name, albedo_image, surface_normals)

    plot_surface_normals(surface_normals)

    display_output(albedo_image, height_map)
