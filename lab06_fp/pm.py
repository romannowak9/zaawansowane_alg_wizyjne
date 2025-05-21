import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import cv2
from copy import copy


def descriptors(gray, points, k_size):
    Y = gray.shape[0]
    X = gray.shape[1]
    # Central points
    pts = list(
        filter(
            lambda pt: pt[0] >= k_size
            and pt[0] < Y - k_size
            and pt[1] >= k_size
            and pt[1] < X - k_size,
            zip(points[0], points[1]),
        )
    )

    # Descriptors
    descs = list()
    for x, y in pts:
        # patch = image[y-size:y+size+1, x-size:x+size+1]
        window = gray[x - k_size : x + k_size + 1, y - k_size : y + k_size + 1]
        desc = window.flatten()
        descs.append((desc - np.mean(desc)) / np.std(desc))

    return list(zip(descs, pts))


def match_descriptors(desc1, desc2, n):
    descs_with_dist = dict()
    for d1, coord1 in desc1:
        min_dist = float('inf')
        min_coord = (None, None)
        for d2, coord2 in desc2:
            dist = np.linalg.norm(d1 - d2)
            if dist < min_dist:
                min_dist = dist
                min_coord = coord2
        
        descs_with_dist[(coord1, min_coord)] = min_dist
        
    n_matches = sorted(descs_with_dist.items(), key=lambda x: x[1])[:n]
    return [(coord1, coord2, dist) for (coord1, coord2), dist in n_matches]
    
    


def compute_H(gray, filter_size=5, k=0.05, normalize=True):
    sobel_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=filter_size)
    sobel_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=filter_size)

    # Autocorrelation matrix elems
    m00 = cv2.GaussianBlur(
        sobel_x * sobel_x, ksize=(filter_size, filter_size), sigmaX=0
    )
    m01 = cv2.GaussianBlur(
        sobel_x * sobel_y, ksize=(filter_size, filter_size), sigmaX=0
    )
    m10 = m01
    m11 = cv2.GaussianBlur(
        sobel_y * sobel_y, ksize=(filter_size, filter_size), sigmaX=0
    )

    out_img = np.zeros(gray.shape, dtype=np.float64)

    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            m_matrix = np.array([[m00[y, x], m01[y, x]], [m10[y, x], m11[y, x]]])
            harris = np.linalg.det(m_matrix) - k * np.trace(m_matrix) ** 2
            # print(harris)
            out_img[y, x] = harris

    out_img_norm = cv2.normalize(out_img, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
    return out_img_norm if normalize else out_img


def find_max(image, size, threshold):  # size - maximum filter mask size
    data_max = filters.maximum_filter(image, size)
    maxima = image == data_max
    diff = image > threshold
    maxima[diff == 0] = 0
    return np.nonzero(maxima)


def appendimages(im1, im2):
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1 - rows2, im2.shape[1]))), axis=0)

    return np.concatenate((im1, im2), axis=1)


def plot_matches(im1, im2, matches):
    colors = ["r", "g", "b", "c", "m", "y"]
    im3 = appendimages(im1, im2)

    plt.figure()
    plt.imshow(im3, cmap='gray')

    cols1 = im1.shape[1]
    for i, m in enumerate(matches):
        plt.plot(
            [m[0][1], m[1][1] + cols1], [m[0][0], m[1][0]], colors[i % 6], linewidth=0.5
        )
    plt.axis("off")





def show_img_with_points(img, points):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.plot(points[1], points[0], '*r')
    plt.show()
    
    
def non_maximum_suppression(points):
    filtered_points = copy(points)
    coords = [(y, x) for y, x, _ in points]
    points_dict = {(y, x) : metric for y, x, metric in points}
    for y in [y for y, _ in coords]:
        for x in [x for _, x in coords]:
            close_points = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (y+i, x+j) in coords:
                        close_points.append((y+i, x+j, points_dict[(y+i, x+j)]))
            if len(close_points) > 1:
                close_points = sorted(close_points, key=lambda x: x[2], reverse=True)
                for point in close_points[1:]:
                    if point in filtered_points:
                        filtered_points.remove(point)
            
    return filtered_points


