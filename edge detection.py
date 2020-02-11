import cv2
import numpy as np
# 打开图片
image = cv2.imread("t15.jpg")
# 转换灰度并去噪声
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 增大边缘的对比度
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1)

gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# 继续去除噪声，图像二值化
blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

# 填充空余区域
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 刻画细节
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)


(cnts, _) = cv2.findContours(
    # 参数一：二值化图像
    closed.copy(),
    # 参数二：轮廓类型
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE
)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))
 
draw_img = cv2.drawContours(image.copy(), [box], -1, (0, 0, 255), 3)
cv2.imshow("draw_img", draw_img)

# cv2.imshow('image',gray)
# cv2.imshow('image2',gradient)
# cv2.imshow('image3',thresh)
cv2.imshow('image4',thresh)
cv2.imshow('image5',closed)
cv2.waitKey(0)
cv2.destroyAllWindows()


