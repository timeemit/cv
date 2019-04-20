import sys
import cv2
import numpy as np
import preprocess


def process(original, processed, filename):
    # image = cv2.cornerHarris(image, 25, 9, 0.01)
    # cv2.imwrite("corners-" + filename, image)

    image, contours, hierarchy = cv2.findContours(
            processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    original_with_contours = cv2.drawContours(
            np.copy(original), contours, -1, (0, 255, 0), 3)
    cv2.imwrite(
        "3-contours-original-{}".format(filename),
        original_with_contours
        )

    hierarchy = hierarchy[0]
    heights = np.zeros((len(hierarchy)), dtype=np.int)
    for i, (after, behind, child, parent) in enumerate(hierarchy):
        while parent != -1:
            parent = hierarchy[parent][3]
            heights[parent] += 1

    for i, height in enumerate(np.nonzero(heights)[0]):
        # Make sure to increment the height of all its siblings, too
        sibling = hierarchy[i][0]
        while sibling != -1:
            heights[sibling] += 1
            sibling = hierarchy[sibling][0]

    mean = np.mean(heights)
    std = np.std(heights)

    contours = np.array(contours)
    contours = contours[np.nonzero(heights > mean + 1 * std)[0]]

    original_with_contours = cv2.drawContours(
            np.copy(original), contours, -1, (0, 255, 0), 3)
    cv2.imwrite(
            "4-contours-filtered-{}".format(filename),
            original_with_contours
            )

    return contours


if __name__ == "__main__":
    filename = sys.argv[1]
    original = cv2.imread(filename)

    processed = preprocess.process(original, filename)
    contours = process(original, processed, filename)
