import cv2
import numpy as np

img = cv2.imread('Competition\\Image_Processing\\test_image.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

threshold = 125

lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)
print(lines.size)
for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite('Competition\\Image_Processing\\houghlines3.jpg', img)

minLineLength = 100
maxLineGap = 10
img = cv2.imread('Competition\\Image_Processing\\test_image.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold,
                        minLineLength, maxLineGap)
print(lines.size)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite('Competition\\Image_Processing\\houghlines5.jpg', img)
