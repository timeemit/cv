import sys
import cv2
import numpy as np
import preprocess
import contour


def process(original, contours, filename):
    # moments = []
    # for c in contours:
    #     m = cv2.moments(c)
    #     moment = [int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])]
    #     moments.append(moment)
    #     original = cv2.circle(
    #             original, tuple(moment), 55, (255, 0, 255), -10)
    # moments = np.array(moments, dtype=np.int)
    # perimiter = cv2.arcLength(moments, True)
    # print(perimiter)

    points = np.zeros((0, 1, 2), dtype=np.int)
    for c in contours:
        points = np.append(points, c, axis=0)

    for point in cv2.convexHull(points, True):
        original = cv2.circle(
                original, tuple(point[0]), 35, (255, 0, 255), -10)

    cv2.imwrite("5-perimiter-{}".format(filename), original)


if __name__ == "__main__":
    filename = sys.argv[1]
    original = cv2.imread(filename)

    processed = preprocess.process(original, filename)
    contours = contour.process(original, processed, filename)
    process(original, contours, filename)
