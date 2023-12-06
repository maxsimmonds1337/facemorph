import cv2
import sys
import os
import dlib
import glob
import copy
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import crop

def match_triangles_by_verticies(triangles_1, triangles_2, facial_points_1, facial_points_2, coded_dict_1, coded_dict_2):
    paired_points = [] # this will hold the paired points for the two triangles
    for triangle_1 in triangles_1:
        found = False
        target_ids = get_triangle_points(triangle_1, coded_dict_1)
        for triangle_2 in triangles_2:
            test_ids = get_triangle_points(triangle_2, coded_dict_2)
            if test_ids == target_ids:
                points_1 = [facial_points_1[id] for id in target_ids]
                points_2 = [facial_points_2[id] for id in test_ids]

                paired_points.append((points_1, points_2))
                found =  True
                break
        if not found:
            print(f'Couldn\'t find pair for id {target_ids}')
    return paired_points

def get_triangle_centroid(triangle_points):

    x = np.mean(triangle_points[0::2])
    y = np.mean(triangle_points[1::2])

    return (x, y)

def is_triangle_unique(triangle: list, used_triangles: list) -> bool:
    
    for used_triangle in used_triangles:
        if np.array_equal(used_triangle, triangle):
            return False
    return True

# to do, could make this better, less hard coded lists
def match_triangles_by_centroid(triangles_list, facial_points_list, coded_dict_list):
    paired_points = [] # this will hold the paired points for the two triangles
    id_index = -1 
    min_distance = float('inf')
    used_triangles = []

    image = np.zeros((450,450,3))

    for triangle_1 in triangles_list[0]:
        # plot the triangle so we can see it
        image = draw_triangular_mesh([triangle_1], image, "single mesh", show_image_bool=False, colour=(0,0,255))
        
        target_centroid = get_triangle_centroid(triangle_1) # This gets the centroid of the current triangle
        for i, triangle_2 in enumerate(triangles_list[1]): # make a copy of the list, since we're going to be modifying it in the loop
            image = draw_triangular_mesh([triangle_2], image, "single mesh", show_image_bool=True, colour=(255,0,0))
            cv2.waitKey(0)
            test_centroid = get_triangle_centroid(triangle_2) # This also gets a centroid, for each triangle in the second list
            distance_check = distance.euclidean(target_centroid, test_centroid)
            if (distance_check < min_distance):
                if(is_triangle_unique(triangle_2, used_triangles)):
                    min_distance = distance_check
                    id_index = i  # here we save the id with the smallest index, to be used later
                    used_triangles.append(triangle_2)

        # we should now have the id of the minimum distance
        target_ids = get_triangle_points(triangle_1, coded_dict_list[0])
        test_ids = get_triangle_points(triangles_list[1][id_index], coded_dict_list[1])

        points_1 = [facial_points_list[0][id] for id in target_ids]
        points_2 = [facial_points_list[1][id] for id in test_ids]

        paired_points.append((points_1, points_2))

        # Reset the things we use to determine distance
        min_distance = float('inf')
        id_index = -1
         
    return paired_points

def add_facial_points(facial_points_list, image, string_title, show_image_bool):
    for point in facial_points_list:
        cv2.circle(image, point, 1, (200,200,200), 1)
    if show_image_bool:
        cv2.imshow(string_title, image)
        # cv2.waitKey(0)

    return image

# Draws the mesh over an image
def draw_triangular_mesh(triangles, image, string_title, show_image_bool, colour = (50,50,50)):
    for triangle in triangles:
        pt1 = (int(triangle[0]), int(triangle[1]))
        pt2 = (int(triangle[2]), int(triangle[3]))
        pt3 = (int(triangle[4]), int(triangle[5]))
        cv2.line(image, pt1, pt2, colour, 1)
        cv2.line(image, pt2, pt3, colour, 1)
        cv2.line(image, pt3, pt1, colour, 1)

    if show_image_bool:
        cv2.imshow(string_title, image)
        # cv2.waitKey(0)

    return image

def add_numbers_to_mesh(facial_points, image, string_title, show_image_bool, colour=(255,255,255)):
    for i, point in enumerate(facial_points):
        cv2.putText(image, str(i), point, color = colour, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.25)
    if show_image_bool:
        cv2.imshow(string_title, image)
        # cv2.waitKey(0)

    return image

def add_centroids(triangles, image, string_title, show_image_bool, colour=(0,0,255)):
    for triangle in triangles:
        centroid = get_triangle_centroid(triangle)
        centroid = (int(centroid[0]), int(centroid[1]))  # Convert to integers
        cv2.circle(image, centroid, 1, colour, 1)
    if show_image_bool:
        cv2.imshow(string_title, image)
        # cv2.waitKey(0)
    return image

def get_triangle_points(triangle, coded_dict):
    points = ((triangle[0],triangle[1]),(triangle[2],triangle[3]),(triangle[4],triangle[5]))
    id = []
    for point in points:
        id.append(coded_dict[point])
    return id
def get_intermediate_shape(tri_1: list, tri_2: list, a: int) -> list:

    inter_shape = []
    for coord_pair in zip(tri_1, tri_2):

        xi = coord_pair[0][0]
        yi = coord_pair[0][1]

        xj = coord_pair[1][0]
        yj = coord_pair[1][1]

        x = int(((1-a)*xi) + (a*xj))
        y = int(((1-a)*yi) + (a*yj))

        inter_shape.append((x,y))
    return inter_shape

def flatten_coords(coord_pairs):
    return  [element for pair in coord_pairs for element in pair]


def unflatten_coords(coord_list):
    return [(coord_list[i], coord_list[i + 1]) for i in range(0, len(coord_list), 2)]