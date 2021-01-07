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
        w_a = np.sqrt(((bottom_r[0] - bottom_l[0])**2) +
                      ((bottom_r[1] - bottom_l[1])**2))
        w_b = np.sqrt(((top_r[0] - top_l[0])**2) + ((top_r[1] - top_l[1])**2))
        width = max(int(w_a), int(w_b))

        h_A = np.sqrt(((top_r[0] - bottom_r[0])**2) +
                      ((top_r[1] - bottom_r[1])**2))
        h_B = np.sqrt(((top_l[0] - bottom_l[0])**2) +
                      ((top_l[1] - bottom_l[1])**2))
        height = max(int(h_A), int(h_B))

        dimensions = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32")

        ordered_corners = np.array(ordered_corners, dtype="float32")

        matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

        return cv2.warpPerspective(image, matrix, (width, height))

    def get_image_width_height(self, image):
        img_w = image.shape[1]
        img_h = image.shape[0]
        return img_w, img_h

    def calculate_scaled_dimension(self, scale, image):
        image_width, image_height = self.get_image_width_height(image)
        ratio_of_new_with_to_old = scale / image_width

        dimension = (scale, int(image_height * ratio_of_new_with_to_old))
        return dimension

    def scale_image(self, image, size):
        img_resized_scaled = cv2.resize(image,
                                        self.calculate_scaled_dimension(
                                            size, image),
                                        interpolation=cv2.INTER_AREA)
        return img_resized_scaled

    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        return cv2.warpAffine(image, M, (nW, nH))

    def black_box(self, image):
        original_image = image.copy()
        image = self.scale_image(image, 800)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        screen_cnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)
            if len(approx) == 4:
                screen_cnt = approx
                transformed = self.perspective_transform(image, screen_cnt)
                break
        # ROI
        cv2.drawContours(image, [screen_cnt], -1, (0, 255, 0), 1)
        (h, w) = transformed.shape[:2]
        if (h > w):
            rotated = self.rotate_image(transformed, 270)
        else:
            rotated = transformed
        return rotated

    def toText(self, image):
        return pytesseract.image_to_string(image,
                                           lang='vie',
                                           config='--psm 7 --oem 3')
