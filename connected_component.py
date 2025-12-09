"""
Course Number: ENGR 13300
Semester: Fall 2025

Description:
   

Assignment Information:
    Assignment:     UDFs file 
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
import matplotlib.pyplot as plt


def matrix_to_binary_image(mat, cell_size=20):
    h, w = mat.shape
    img = np.zeros((h * cell_size, w * cell_size), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            v = mat[y, x]
            y1, y2 = y * cell_size, (y + 1) * cell_size
            x1, x2 = x * cell_size, (x + 1) * cell_size

            if v == 0:
                img[y1:y2, x1:x2] = 255       # đen
            else:
                img[y1:y2, x1:x2] = 0     # trắng

    return img

def visualize_matrix_cv2(binary, mat, cell_size=20):
    h, w = mat.shape
    img_h = h * cell_size
    img_w = w * cell_size

    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    for y in range(h):
        for x in range(w):
            value = mat[y, x]

            y1, y2 = y * cell_size, (y + 1) * cell_size
            x1, x2 = x * cell_size, (x + 1) * cell_size

            if value == 0:
                color = (255, 255, 255)  # trắng
            else:
                color = (0, 0, 0)        # đen

            img[y1:y2, x1:x2] = color

            text_color = (0, 0, 255) if value == 0 else (255, 255, 255)
            cv2.putText(
                img, str(int(value)),
                (x1 + 5, y1 + cell_size - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, text_color, 1, cv2.LINE_AA
            )

            cv2.rectangle(img, (x1, y1), (x2, y2), (150, 150, 150), 1)

    mat_img_color = cv2.cvtColor(matrix_to_binary_image(binary, cell_size), cv2.COLOR_GRAY2BGR)

    result = cv2.hconcat([mat_img_color, img])
    return result

def otsu_thresholding(img_array):
    pixels = img_array.ravel()
    hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
    prob = hist / (pixels.size)

    max_variance = 0
    threshold = 0
    for i in range(256):
        w0 = np.sum(prob[:i+1]) #cumulative probability
        w1 = np.sum(prob[i+1:])
        if w0 == 0 or w1 == 0:
            continue
        u0 = np.sum(np.arange(i+1)*prob[:i+1])/w0 #intensity 
        u1 = np.sum(np.arange(i+1, 256)*prob[i+1:])/w1

        variance = w0*w1*(u0-u1)**2
        if variance > max_variance:
            max_variance = variance
            threshold = i
    return threshold

def check_label(i, j, labels, w):
    """
    0 3 2
    1 x 0
    0 0 0 
    -> (i-1, j), (i-1, j+1), (i-1, j-1), (i, j-1)
    """
    if i == 0:
        if j == 0:
            return []
        else:
            return [labels[i, j-1]]
    elif j == 0:
        return [labels[i-1, j], labels[i-1, j+1]]
    elif j == w-1:
        return [labels[i, j-1], labels[i-1, j-1], labels[i-1, j]]
    
    return [labels[i, j-1], labels[i-1, j-1], labels[i-1, j], labels[i-1, j+1]]
        
def connected_component(binary_image):
    h, w = binary_image.shape
    labels = np.zeros((h, w))
    current_label = 0
    equivalence_table = {}

    for i in range(0, h):
        for j in range(0, w):
            if binary_image[i, j] == 0:
                continue

            # [0, 0, 0, 0] -> set(list) = {0}
            # [0, 0, 1, 2] -> set = {0, 1, 2}
            list_check_labels = np.unique(np.array(check_label(i, j, labels, w)))
            if len(list_check_labels) == 0:
                current_label += 1
                labels[i, j] = current_label
                equivalence_table[current_label] = current_label
            elif len(list_check_labels) == 1:
                if list_check_labels[0] == 0:
                    current_label += 1
                    labels[i, j] = current_label
                    equivalence_table[current_label] = current_label

                else:
                    labels[i, j] = list_check_labels[0]
            
            else:
                # 0 1 2 3 -> x=1 -> update equi[2]=1, equi[3]=1
                if list_check_labels[0] == 0:
                    labels[i, j] = list_check_labels[1]
                    for ii in range(2, len(list_check_labels)):
                        equivalence_table[list_check_labels[ii]] = list_check_labels[1]
                else: 
                    labels[i, j] = list_check_labels[0]
                    for ii in range(1, len(list_check_labels)):
                        equivalence_table[list_check_labels[ii]] = list_check_labels[0]
    for i in range(0, h):
        for j in range(0, w):
            if binary_image[i, j] == 0:
                continue
            while labels[i, j] != equivalence_table[labels[i, j]]:
                labels[i, j] = equivalence_table[labels[i, j]]
    return labels

def main():
    return
if __name__ == "__main__":
        main()