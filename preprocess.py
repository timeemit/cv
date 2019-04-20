import sys
import cv2


def process(image, filename):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("0-grey-{}".format(filename), image)

    image = cv2.GaussianBlur(image, (25, 25), 35)
    cv2.imwrite("1-blurred-{}".format(filename), image)

    adaptive = cv2.adaptiveThreshold(image, 255, cv2.THRESH_BINARY,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 25, 2)
    cv2.imwrite("2-threshold-adaptive-{}".format(filename), adaptive)
    return adaptive

# otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv2.imwrite("threshold-otsu-" + filename, otsu)


if __name__ == "__main__":
    filename = sys.argv[1]
    image = cv2.imread(filename)
    process(image)
