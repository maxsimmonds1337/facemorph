import cv2
import numpy as np
import crop
import geo
from itertools import pairwise

# Load the images
image1 = cv2.imread("human1r.jpg")
image2 = cv2.imread("elon.jpg")
images = (image1, image2)

cv2.imshow("human 1", image1)
cv2.imshow("human 2", image2)
# cv2.waitKey(0)


# Attempt to find faces and bounding rects
image_facial_points_list = crop.get_facial_points(images)
bounding_rects = crop.get_bounding_rects(image_facial_points_list)

# Scale the images such that the bounding box around the faces are of equal size
scaled_images, scaled_facial_points_list, scaled_bounding_rects_list = crop.scale_by_facial_rectangle(bounding_rects, images)

# for each scaled image, plot the bounding rectangle around the face
for i, scaled_image in enumerate(scaled_images):
    x,y,h,w = scaled_bounding_rects_list[i]
    # cv2.rectangle(scaled_images[i],(x,y),(x+h,y+w),(0,255,0),1)
    cv2.imshow("Scaled image " + str(i+1), scaled_image)

# now we crop so that the bounding rectangles from all images are aligned
minus_x_distances = []
minus_y_distances = []
positive_x_distances = [] 
positive_y_distances =  []

bounding_rect_centroids = []

for i, scaled_image in enumerate(scaled_images):

    centroid_x, centroid_y = crop.get_rect_centroid(scaled_bounding_rects_list[i])

    w = scaled_image.shape[1]
    h = scaled_image.shape[0]

    minus_x_distances.append(centroid_x)
    positive_x_distances.append(w - centroid_x)
    minus_y_distances.append(centroid_y)
    positive_y_distances.append(h - centroid_y)

    bounding_rect_centroids.append((centroid_x, centroid_y))


min_minus_x = int(min(minus_x_distances))
min_positive_x = int(min(positive_x_distances))
min_minus_y = int(min(minus_y_distances))
min_positive_y = int(min(positive_y_distances))

cropped_scaled_images = []

for i, image in enumerate(scaled_images):

    bounding_rect_centroid_x, bounding_rect_centroid_y = bounding_rect_centroids[i]
    image_w, image_h = image.shape[1], image.shape[0]

    start_y = int((bounding_rect_centroid_y - min_minus_y))
    end_y = int((bounding_rect_centroid_y + min_positive_y))

    start_x = int((bounding_rect_centroid_x - min_minus_x))
    end_x = int((bounding_rect_centroid_x + min_positive_x))

    cropped_scaled_images.append(image[start_y:end_y , start_x:end_x])
    # cv2.imshow("Cropped and aligned image " + str(i + 1), cropped_scaled_images[i])
    # cv2.waitKey(0)

# now that they're cropped, we need to find the face again, this could probably be done by knowing the offset in x and y, but i'm lazy and will use my func again
cropped_scaled_facial_points_list = crop.get_facial_points(cropped_scaled_images)
cropped_scaled_bounding_rects_list = crop.get_bounding_rects(cropped_scaled_facial_points_list)

# for each scaled image, plot the bounding rectangle around the face
for i, cropped_scaled_image in enumerate(cropped_scaled_images):
    x,y,h,w = cropped_scaled_bounding_rects_list[i]
    # cv2.rectangle(cropped_scaled_image,(x,y),(x+h,y+w),(0,255,0),1)
    cv2.imshow("Cropped and aligned image " + str(i + 1), cropped_scaled_images[i])
    # cv2.waitKey(0)

# now we add the background 
width, height = cropped_scaled_images[0].shape[1], cropped_scaled_images[0].shape[0]
background_points = crop.add_background_points(height, width) # images are now scaled and cropped, to theu're the same sizie

coded_facial_points_list = []

for cropped_scaled_facial_points in cropped_scaled_facial_points_list:
    for background_point in background_points:
        cropped_scaled_facial_points.append(background_point)
    # make the coded dictionary, used for looking up the facial point and it's unique index
    coded_facial_points_list_temp = {}
    for index, pair in enumerate(list(cropped_scaled_facial_points)):
        coded_facial_points_list_temp[index]=pair
    coded_facial_points_list.append(coded_facial_points_list_temp)
    # coded_facial_points_list.append({x[0]:x[1] for x in list(zip(cropped_scaled_facial_points , range(len(cropped_scaled_facial_points))))})

cropped_scaled_image_triangles_list = []
for i, image in enumerate(cropped_scaled_images):
    geo.add_facial_points(cropped_scaled_facial_points_list[i], np.zeros_like(image), "image " + str(i) + " with facial points", True)

    rect=[0,0]
    rect.extend([image.shape[1], image.shape[0]])

    # start a subdiv
    image_subdiv = cv2.Subdiv2D(rect)

    # populate the subdiv
    for index in coded_facial_points_list[i]:
        image_subdiv.insert(coded_facial_points_list[i][index])

    # get the trianglees
    cropped_scaled_image_triangles_list.append(image_subdiv.getTriangleList())

    meshed_image = geo.draw_triangular_mesh(cropped_scaled_image_triangles_list[i], np.zeros_like(image), "Mesh with no image " + str(i), False)
    mesh_with_numbers = geo.add_numbers_to_mesh(cropped_scaled_facial_points_list[i], meshed_image, "Mesh with numbers " + str(i), True) # doesn't work
    # mesh_with_centroids = geo.add_centroids(cropped_scaled_image_triangles_list[i], mesh_with_numbers, "Mesh with centroids " + str(i), True)

# let's calc the mean mesh
mean_cropped_scaled_facial_points = []
for i in range(76):
    fp1 = cropped_scaled_facial_points_list[0][i]
    fp2 = cropped_scaled_facial_points_list[1][i]

    avg_fp = tuple(sum(x)//2 for x in zip(fp1,fp2))

    mean_cropped_scaled_facial_points.append(avg_fp)


mean_coded_facial_points = {x[0]:x[1] for x in list(zip(mean_cropped_scaled_facial_points , range(76)))}
mean_rect = cv2.boundingRect(np.array(mean_cropped_scaled_facial_points))
mean_image_subdiv = cv2.Subdiv2D(mean_rect)

# populate the subdiv
for (x, y) in mean_coded_facial_points:
    mean_image_subdiv.insert((x, y))

mean_cropped_scaled_image_triangles = mean_image_subdiv.getTriangleList()

destination_image = np.zeros(cropped_scaled_images[0].shape, dtype = cropped_scaled_images[0].dtype)

for mean_triangle in mean_cropped_scaled_image_triangles:

    # get the unique facial point numbers
    mean_verticies = geo.get_triangle_points(mean_triangle, mean_coded_facial_points)
    print(f'mean verticies: {mean_verticies}')

    # use the unique numbers to get the points for the first image triangle
    vertices_1 = []
    vertices_2 = []

    for mean_vertex in mean_verticies:
        # vertices_1_pts = list(coded_facial_points_list[0].keys())
        # vertices_2_pts = list(coded_facial_points_list[1].keys())

        # TODO: don't hard code this
        vertices_1.append(coded_facial_points_list[0][mean_vertex])
        vertices_2.append(coded_facial_points_list[1][mean_vertex])

    # vertices_1 = [coord for tup in vertices_1 for coord in tup]
    # vertices_2 = [coord for tup in vertices_2 for coord in tup]

    intermediate_shape = geo.get_intermediate_shape(vertices_1, vertices_2, 0.5) # TODO make this not hardcoded (the 0.5, same as the one below)

    # get a bounding box for each triangle

    r1_x, r1_y, r1_w, r1_h = cv2.boundingRect(np.float32(vertices_1))
    r2_x, r2_y, r2_w, r2_h = cv2.boundingRect(np.float32(vertices_2))
    r_x, r_y, r_w, r_h = cv2.boundingRect(np.float32(intermediate_shape))

    # now we offset the triangles by the upper left corner of their respective bounding boxes
    image_1_local_rect = []
    image_2_local_rect = []
    intermediate_shape_local_rect = []

    for i in range(0, 3):
        image_1_local_rect.append( ( (vertices_1[i][0] - r1_x), (vertices_1[i][1] - r1_y)))
        image_2_local_rect.append( ( (vertices_2[i][0] - r2_x), (vertices_2[i][1] - r2_y)))
        intermediate_shape_local_rect.append( ( (intermediate_shape[i][0] - r_x), (intermediate_shape[i][1] - r_y)))

    # Get mask by filling triangle
    mask = np.zeros((r_h, r_w, 3), dtype = np.float32) # get a mask the size of the intermediate shape (it's a rectangle that bounds the intermediate shape triangle)
    cv2.fillConvexPoly(mask, np.int32(intermediate_shape_local_rect), (1.0, 1.0, 1.0), 16, 0) 

    # we now crop the original images, with the bounding boxes
    image_1_rect = cropped_scaled_images[0][r1_y:r1_y+r1_h, r1_x:r1_x+r1_w]
    image_2_rect = cropped_scaled_images[1][r2_y:r2_y+r2_h, r2_x:r2_x+r2_w]

    # we then get the TM that goes from one local coord rect to the intermediate shape.
    print(f'getting TM for intershape to verticies_1, which is: \n {image_1_local_rect} --> {intermediate_shape_local_rect}')
    transformation_matrix_1 = cv2.getAffineTransform(np.float32(image_1_local_rect), np.float32(intermediate_shape_local_rect))

    print(f'getting TM for intershape to verticies_1, which is: \n {image_2_local_rect} --> {intermediate_shape_local_rect}')
    transformation_matrix_2 = cv2.getAffineTransform(np.float32(image_2_local_rect), np.float32(intermediate_shape_local_rect))

    # we then warp all the pixels from the small sqaure patch that bounds the triangle
    warped_image_1 = cv2.warpAffine(np.array(image_1_rect, dtype=np.float32), transformation_matrix_1, (r_w, r_h), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    warped_image_2 = cv2.warpAffine(np.array(image_2_rect, dtype=np.float32), transformation_matrix_2, (r_w, r_h), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # then we blend the the colours of the two patches
    blended_image_rect = 0.5 * warped_image_1 + 0.5 * warped_image_2 # TODO change 0.5 to something that can be adjusted, to change the blending
    # finally, we add back into the original image, the now blended sqaure:
    destination_image[r_y:r_y+r_h, r_x:r_x+r_w] = destination_image[r_y:r_y+r_h, r_x:r_x+r_w] * (1 - mask) + blended_image_rect * mask

combined_images_1 = cv2.hconcat([cropped_scaled_images[0], destination_image, cropped_scaled_images[1]])

mean_mesh = geo.draw_triangular_mesh(mean_cropped_scaled_image_triangles, np.zeros_like(cropped_scaled_images[0]), string_title="Meshes", show_image_bool=False, colour=(0,0,255))
human1_mesh = geo.draw_triangular_mesh(cropped_scaled_image_triangles_list[0], np.zeros_like(cropped_scaled_images[0]), string_title="Meshes", show_image_bool=False, colour=(0,255,0))
human2_mesh = geo.draw_triangular_mesh(cropped_scaled_image_triangles_list[1], np.zeros_like(cropped_scaled_images[1]), string_title="Meshes", show_image_bool=False, colour=(255,0,0))
combined_images_2 = cv2.hconcat([human1_mesh, mean_mesh, human2_mesh])

combined_images = cv2.vconcat([combined_images_1, combined_images_2])

cv2.imshow("Face Morph", combined_images_1)
cv2.imshow("Face Morph Meshes", combined_images_2)
cv2.imshow("Face Morph All", combined_images)


cv2.imshow("Face Morph single", destination_image)


print(f'done')
cv2.waitKey(0)
cv2.destroyAllWindows()
