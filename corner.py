import cv2
import numpy as np
import pytesseract

import csv


class white_box():

    def order_corner_points(self, corners):
        # Separate corners into individual points
        # Index 0 - top-right
        #       1 - top-left
        #       2 - bottom-left
        #       3 - bottom-right
        corners = [(corner[0][0], corner[0][1]) for corner in corners]
        top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[
            2], corners[3]
        return (top_l, top_r, bottom_r, bottom_l)

    def perspective_transform(self, image, corners):
        # Order points in clockwise order
        ordered_corners = self.order_corner_points(corners)
        top_l, top_r, bottom_r, bottom_l = ordered_corners

        # Determine width of new image which is the max distance between
        # (bottom right and bottom left) or (top right and top left) x-coordinates
        width_A = np.sqrt(((bottom_r[0] - bottom_l[0])**2) +
                          ((bottom_r[1] - bottom_l[1])**2))
        width_B = np.sqrt(((top_r[0] - top_l[0])**2) +
                          ((top_r[1] - top_l[1])**2))
        width = max(int(width_A), int(width_B))

        # Determine height of new image which is the max distance between
        # (top right and bottom right) or (top left and bottom left) y-coordinates
        height_A = np.sqrt(((top_r[0] - bottom_r[0])**2) +
                           ((top_r[1] - bottom_r[1])**2))
        height_B = np.sqrt(((top_l[0] - bottom_l[0])**2) +
                           ((top_l[1] - bottom_l[1])**2))
        height = max(int(height_A), int(height_B))

        # Construct new points to obtain top-down view of image in
        # top_r, top_l, bottom_l, bottom_r order
        dimensions = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32")

        # Convert to Numpy format
        ordered_corners = np.array(ordered_corners, dtype="float32")

        # Find perspective transform matrix
        matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

        # Return the transformed image
        return cv2.warpPerspective(image, matrix, (width, height))

    def get_image_width_height(self, image):
        image_width = image.shape[1]  # current image's width
        image_height = image.shape[0]  # current image's height
        return image_width, image_height

    def calculate_scaled_dimension(self, scale, image):
        image_width, image_height = self.get_image_width_height(image)
        ratio_of_new_with_to_old = scale / image_width

        dimension = (scale, int(image_height * ratio_of_new_with_to_old))
        return dimension

    def scale_image(self, image, size):
        image_resized_scaled = cv2.resize(image,
                                          self.calculate_scaled_dimension(
                                              size, image),
                                          interpolation=cv2.INTER_AREA)
        return image_resized_scaled

    def rotate_image(self, image, angle):
        # Grab the dimensions of the image and then determine the center
        (h, w) = image.shape[:2]
        (cX, cY) = (w / 2, h / 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # Compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # Perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    def black_box(self, image):
        # image = cv2.imread('rotate.jpg')
        # cv2.imshow('',image)
        # cv2.waitKey()
        original_image = image.copy()

        image = self.scale_image(image, 800)

        # convert the image to grayscale, blur it, and find edges in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        screen_cnt = None

        # loop over our contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)

            if len(approx) == 4:
                screen_cnt = approx
                transformed = self.perspective_transform(image, screen_cnt)
                break

        # Draw ROI
        cv2.drawContours(image, [screen_cnt], -1, (0, 255, 0), 1)

        (h, w) = transformed.shape[:2]

        if (h > w):
            rotated = self.rotate_image(transformed, 270)
        else:
            rotated = transformed

        # cv2.imshow("image", original_image)
        # cv2.imshow("ROI", image)
        # cv2.imshow("transformed", transformed)
        return rotated

    def toText(self, image):
        return pytesseract.image_to_string(image,
                                           lang='vie',
                                           config='--psm 7 --oem 3')
# b=cv2.imread('b_left.jpg')
# img3 = cv2.resize(b, (0, 0), fx=0.2, fy=0.2)
# cv2.imshow('a',img3)
# a = white_box().black_box(cv2.imread('b_left.jpg'))

# cv2.imshow('a',a)
# # cv2.waitKey()
# # # university name
# b = a[27:27 + 58 - 27, 103:103 + 570 - 103]
# cv2.imshow('b', b)

# # # name
# c = a[127:127 + 173 - 127, 276:472]
# cv2.imshow('c',c)

# # # birthday
# d = a[165:209, 310:501]
# cv2.imshow('d', d)

# # # major
# e = a[205:244, 301:550]
# cv2.imshow('e',e)

# # # year of admission
# f = a[261:290, 332:410]
# cv2.imshow('f', f)

# # # ID
# g = a[374:433, 36:200]
# cv2.imshow('g',g)

# # cv2.imshow('b', b)
# # pytesseract.pytesseract.tesseract_cmd = 'F:/c++opencv/finalProject/tesseract/tesseract.exe'

# # text = pytesseract.image_to_string(b, lang='vie', config='--psm 7 --oem 3')
# # #  -c tessedit_char_whitelist=0123456789')
# # a = text.encode('utf-8')
# # # a = text

# # with open('innovators.csv', 'w', newline='',encoding="UTF-8") as file:
# #     writer = csv.writer(file)
# #     writer.writerow(["SN", "Name", "Contribution"])
# #     writer.writerow(["SN", "Name", a[:-2]])
# # f = open('recognized.txt', 'wb')
# # f.write(a)
# # f.close()
# cv2.waitKey()