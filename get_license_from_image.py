import cv2
from get_license_number import *

image=cv2.imread("N:\Projects\Vehicle_update\image\IMG_2078.JPG")
license_plate_number=get_result_api(image)
if license_plate_number!=None:
    print(license_plate_number)