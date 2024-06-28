import cv2

image = cv2.imread('asset\/officialKey\/H706\/LINE_ALBUM_H706_240628_28.jpg')
height, width, _ = image.shape

xmin, ymin, xmax, ymax = 0.359205,0.429925,0.576715,0.876777

x1, y1 = int(xmin * width), int(ymin * height) # 左上
x2, y2 = int(xmax * width), int(ymax * height) # 右下

cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()