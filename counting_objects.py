"""
Course Number: ENGR 13300
Semester: Fall 2025

Description:
   

Assignment Information:
    Assignment:     individual project
    Team ID:        LC2 - 24
    Author:         Nhan Do, do113@purdue.edu;
    Date:           09/12/2025

Contributors:
    Name, login@purdue [repeat for each]

    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work. """


import cv2
import numpy as np
import math

from connected_component import otsu_thresholding
from connected_component import connected_component



def bgr_to_grayscale(img_array):
    img_array = img_array / 255 #Normalizes the pixel values
    linearize = img_array <= .0405 #Linearizes the pixels
    image_linearized = np.where(linearize, img_array/12.92,((img_array+0.055)/1.055)**2.4)
    Y = .2126*image_linearized[:, :, 2] + .7152*image_linearized[:, :, 1] + .0722*image_linearized[:, :, 0] #Uses the ITU-R Recommendation BT.709 standard for grayscale conversion
    gray_array = (Y * 255).round().astype(np.uint8) #Converts the pixel values back to 8-bit
    return gray_array
def resize_img(img_array, scale_factor):
    scale_factor = int(scale_factor)
    if img_array.ndim == 2:
        h, w = img_array.shape
        c = 1
        img_array = img_array[:,:,np.newaxis]
    elif img_array.ndim == 3:
        h, w, c = img_array.shape

    new_h = math.ceil(h/scale_factor)
    new_w = math.ceil(w/scale_factor)
    resized_img = np.zeros((new_h, new_w,c), dtype=img_array.dtype)
    array = []

    for i in range(0, h, scale_factor):
        for j in range(0, w, scale_factor):
            mean_value = img_array[i:i+scale_factor, j:j+scale_factor].mean(axis=(0,1))
            array.append(mean_value)
    n = 0
    for i in range(0, new_h):
        for j in range(0, new_w):
            resized_img[i,j] = array[n]
            n += 1
    if c == 1:
        return resized_img[:,:,0]
    else: 
        return resized_img

def main():
    while True:
        number = input("Select the image number: ")
        if not number.isdigit():
            print("ERROR: Input must be a whole number. Please try again.\n")
            continue   # Do NOT exit- let the user try again
        number = int(number)
        if number < 1 or number > 10:
            print("ERROR: Number must be between 1 and 10. Please try again.\n")
            continue   # Do NOT exit — let the user try again
        else :
            print("Valid input received:", number)
            print(f"Image {number} will be processed")
            break
    img = cv2.imread(f'test_{number}.jpg')
    H, W = img.shape[:2]
    print(f"Origin shape: {img.shape}")
    if H > 600:
        img = resize_img(img, math.ceil(H/600))
    print(f"Resized shape: {img.shape}")

    img_gray = bgr_to_grayscale(img)
    mask = (img_gray > otsu_thresholding(img_gray)).astype(np.uint8)
    if np.count_nonzero(mask == 1) > np.count_nonzero(mask == 0):
        mask = (img_gray < otsu_thresholding(img_gray)).astype(np.uint8)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    while True:
        a = input("Show images y/n: ")
        if a == 'y':
            mask_show = mask.astype(np.uint8)*255
            img[mask] = [255, 255, 255]
            mask_show = cv2.cvtColor(mask_show, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((mask_show, img))
            cv2.imshow("Binary image and Original image", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
        elif a == 'n': break
        print("Wrong input!")
        continue
    num = connected_component(mask)
    S = np.unique(num)
    #print(S)
    print("The number of objects in the image is: ", len(S) - 1)

if __name__ == "__main__":
        main()