from picamzero import Camera
from astro_pi_orbit import ISS
import time
from exif import Image
from datetime import datetime
import cv2
import math
#assign pi camera to cam
cam = Camera()

#counter for loop
i = 0

#metres/pixel
GSD = 26500

# Parameters for arc length 
radius_large = 6786000  # Radius of the iss's orbit in meters, from the center of the earth (6,786 km)
radius_small = 6378000  # Radius of the eath in meters (6,378 km)

"""
The base code is from the example however I have tweaked the GSD to suit our images and also added a function to account for the iss being on an orbit rather than straight line.

The code does not actively count it's running time but the photo taking loop runs for just over 8 minuts and takes 35 photos with an interval of 14 seconds.
The calculation loop tends to run for 1:30 maximum. I have done an extensive series of tests to confirm that the running time is not over 10 minutes.

The calculation is done by feeding the images in 2 at a time as follows: 1&2, 2&3, 3&4 etc.
Any outliers are removed, which is normally caused by clouds and causes readings over 10km/s throwing off the mean average.

I do rely on stats from google for the ISS's orbit height for calculating the arc length, however even if it is off it will be better than not having that calculation there.
"""



#------------------
# code begins

#get time from image meta data
def get_time(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time

# calculate difference in two image's time
def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
    return time_difference.seconds

#convert images to cv for processing
def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv

#find features in the image to match
def calculate_features(image_1, image_2, feature_number):
    orb = cv2.ORB_create(nfeatures=feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

#find the features across images and mark them as a match
def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

#testing function to display the matches on images
def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:50], None)
    resize = cv2.resize(match_img, (1600, 600), interpolation=cv2.INTER_AREA)
    cv2.imshow('matches', resize)
    cv2.waitKey(0)
    cv2.destroyWindow('matches')

#find the coordinates 
def find_matching_coordinates(keypoints_1, keypoints2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1, y1) = keypoints_1[image_1_idx].pt
        (x2, y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1, y1))
        coordinates_2.append((x2, y2))
    return coordinates_1, coordinates_2

#account for the fact that iss is on an orbit not travelling in a straight line
def calculate_arc_length(chord_length_small, radius_large, radius_small):
    # Calculate the angle theta in radians for the smaller circle
    theta_small = 2 * math.asin(chord_length_small / (2 * radius_small))

    # Calculate the arc length for the larger circle using the same angle
    arc_length_large = radius_large * theta_small
    return arc_length_large


#calculate the mean distance of matches
def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances = all_distances + distance
    return all_distances / len(merged_coordinates)

#calculate speed from time dif and distance
def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed

#combine all functions into one which returns speed
def calculate_speed(image_1, image_2, GSD):
    global image_1_cv
    global image_2_cv
    global keypoints_1
    global keypoints_2
    time_difference = get_time_difference(image_1, image_2)  # get time difference between images
    image_1_cv, image_2_cv = convert_to_cv(image_1, image_2)  # create opencfv images objects
    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv,1000)  # get keypoints and descriptors
    matches = calculate_matches(descriptors_1, descriptors_2)  # match descriptors
    #display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches)  # display matches - COMMENT OUT OR PROGRAM WILL NOT RUN (I spent way too long troubleshooting)
    coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
    average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
    distance_arc_adjusted = arc_length = calculate_arc_length(average_feature_distance, radius_large, radius_small)
    speed = calculate_speed_in_kmps(distance_arc_adjusted, GSD, time_difference)
    print(speed)
    return speed




#function for finding speed from list of images, 2 at a time
def calculate(image_names, GSD):
    results = []
    # Iterate through the list of image names, feeding pairs as (image[i], image[i + 1])
    for i in range(len(image_names)-1):
        image_1 = image_names[i]
        image_2 = image_names[i + 1]
        result = calculate_speed(image_1, image_2)# Call the function with the image pair
        if result < 10:
            results.append(result)  # Save the result to the list
        else:
            print("removed outlier")
    return results

#number of images to use, affects time
num_images = 35
image_names = [f"image_{i}.jpg" for i in range(1, num_images)]
print(f"image_names: {image_names}")
#loop to take image every 10 seconds
for i in range(num_images-1):
    # Generate image names based on the capture sequence
    print(f"index: {i}")
    cam.take_photo(image_names[i])
    time.sleep(14)  # Wait before taking photos again
    i = i+1

#output for debugging
results = calculate(image_names, GSD)
print("Results:", results)
mean = sum(results)/len(results)

#write to results file
with open("result.txt", "w") as file:
    file.write(str(round(mean, 4)))
    


