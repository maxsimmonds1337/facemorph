import dlib
import cv2
import numpy as np

class NoFaceFound(Exception):
   """Raised when there is no face found"""
   pass

def get_facial_points(images: list) -> list:
    """
    Takes a list of images, finds a face, and returns a 76 point list back.
    68 points for the facial features, and 8 to encorparate a background
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    facial_points_list = []

    for image in images:

        facial_points = []
       

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.

        facial_bounding_box = detector(image, 1) # returns top left and bottom right coors of a face

        try:
            if len(facial_bounding_box) == 0:
                raise NoFaceFound
        except NoFaceFound:
            print("Sorry, but I couldn't find a face in the image.")

        for k, rect in enumerate(facial_bounding_box):

            # Get the landmarks/parts for the face in rect.
            shape = predictor(image, rect)

            for i in range(0,68):
                x = shape.part(i).x
                y = shape.part(i).y
                facial_points.append((x, y))
                # cv2.circle(image, (x, y), 1, (0, 255, 0), 1)

        facial_points_list.append(facial_points)    # we then add the facial points to a list containing all the fps

    return facial_points_list

def add_background_points(height, width) -> list:

    """
    Adds 8 points to a list, the four corners and the 4 midpoints between them, and returns the list based on the width and height of an image
    """

    facial_points = []

    TOP_LEFT = (0,0)
    TOP_MIDDLE = (width//2, 0)
    TOP_RIGHT = (width-1, 0)
    LEFT_MIDDLE = (0, height//2-1)
    RIGHT_MIDDLE = (width-1, height//2-1)
    BOTTOM_LEFT = (0, height-1)
    BOTTOM_MIDDLE = (width//2, height-1)
    BOTTOM_RIGHT = (width-1, height-1)

    # Add back the background
    facial_points.append(TOP_LEFT)
    facial_points.append(TOP_MIDDLE)
    facial_points.append(TOP_RIGHT)
    facial_points.append(LEFT_MIDDLE)
    facial_points.append(RIGHT_MIDDLE)
    facial_points.append(BOTTOM_LEFT)
    facial_points.append(BOTTOM_MIDDLE)
    facial_points.append(BOTTOM_RIGHT)

    return facial_points
    

def find_facial_center(facial_points_list: list) -> list:
    """
    Returns a list of rectanges that bound the points given in the facial_points_list
    """
    center_points_list = []

    for facial_point in facial_points_list:
        center_points_list.append(cv2.boundingRect(np.array(facial_point))) 
    return center_points_list

def get_bounding_rects(facial_points):
    bounding_rects = []
    for facial_point in facial_points:
        x,y,w,h=cv2.boundingRect(np.array(facial_point)) # remove the last 8 points, as these define the bounds of the image
        bounding_rects.append((x,y,w,h))
    return bounding_rects

def scale_by_facial_rectangle(bounding_rects, images):
    sizes_x, sizes_y = {}, {}
    for i, (_, _, w, h) in enumerate(bounding_rects):
        sizes_x[i] = w
        sizes_y[i] = h

    minimum_res_x = min(sizes_x.values())
    minimum_res_y = min(sizes_y.values())
    min_aspect_ratio = minimum_res_x / minimum_res_y

    scaled_images = []

    for i, image in enumerate(images):
        curr_aspect_ratio = sizes_x[i] / sizes_y[i]
        if curr_aspect_ratio > min_aspect_ratio:
            scale_factor = minimum_res_x / sizes_x[i]
        else:
            scale_factor = minimum_res_y / sizes_y[i]
        scaled_images.append(cv2.resize(image, None, fx=scale_factor, fy=scale_factor))
    
    scaled_facial_points_list = get_facial_points(scaled_images)

    scaled_bounding_rects_list = get_bounding_rects(scaled_facial_points_list)

    return scaled_images, scaled_facial_points_list, scaled_bounding_rects_list

def get_rect_centroid(bounding_rect):
    x,y,w,h = bounding_rect
    # get centroid
    centroid_x = x + (w/2)
    centroid_y = y + (h/2)

    return centroid_x, centroid_y