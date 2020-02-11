import cv2
import numpy as np
import math


def isRayIntersectsSegment(poi,s_poi,e_poi): #[x,y] [lng,lat]
    #输入：判断点，边起点，边终点，都是[lng,lat]格式数组
    if s_poi[1]==e_poi[1]: #排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if s_poi[1]>poi[1] and e_poi[1]>poi[1]: #线段在射线上边
        return False
    if s_poi[1]<poi[1] and e_poi[1]<poi[1]: #线段在射线下边
        return False
    if s_poi[1]==poi[1] and e_poi[1]>poi[1]: #交点为下端点，对应spoint
        return False
    if e_poi[1]==poi[1] and s_poi[1]>poi[1]: #交点为下端点，对应epoint
        return False
    if s_poi[0]<poi[0] and e_poi[1]<poi[1]: #线段在射线左边
        return False

    xseg=e_poi[0]-(e_poi[0]-s_poi[0])*(e_poi[1]-poi[1])/(e_poi[1]-s_poi[1]) #求交
    if xseg<poi[0]: #交点在射线起点的左侧
        return False
    return True  #排除上述情况之后

def isPoiWithinPoly(poi,poly):
    #输入：点，多边形三维数组
    #poly=[[[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]],[[w1,t1],……[wk,tk]]] 三维数组

    #可以先判断点是否在外包矩形内 
    #if not isPoiWithinBox(poi,mbr=[[0,0],[180,90]]): return False
    #但算最小外包矩形本身需要循环边，会造成开销，本处略去
    sinsc=0 #交点个数
    for epoly in poly: #循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]
        for i in range(len(epoly)-1): #[0,len-1]
            s_poi=epoly[i]
            e_poi=epoly[i+1]
            if isRayIntersectsSegment(poi,s_poi,e_poi):
                sinsc+=1 #有交点就加1

    return True if sinsc%2==1 else  False


# 计算中心点
def center_point(box):
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    cx = (x2-x1)/2 + x1
    cy = (y2-y1)/2 + y1
    return (cx,cy)


def main(img):
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 阈值化,生成二值图
    # 平滑图像
    dst = cv2.GaussianBlur(dst,(3,3),0.5)
    # 阈值分割
    OtsuThresh = 0
    # src：表示的是图片源 thresh：表示的是阈值（起始值）maxval：表示的是最大值 type：表示的是这里划分的时候使用的是什么类型的算法
    # 调整点
    OtsuThresh,dst = cv2.threshold(dst,80,255,2)

    # 消除细小白点
    # 创建结构元
    s = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    dst = cv2.morphologyEx(dst,cv2.MORPH_OPEN,s,iterations=1)
    cv2.imshow("dst", dst)

    (cnts, _) = cv2.findContours(
        # 参数一：二值化图像
        dst.copy(),
        # 参数二：轮廓类型
        # cv2.RETR_LIST,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    #############################test
    cache = cv2.drawContours(img.copy(),cnts,-1,(0,255,0),3) 
    cv2.imshow('test2',cache)
##########################

    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

#######################
    d = img.copy()
    try:
        c1 = sorted(cnts, key=cv2.contourArea, reverse=True)[1]
        rect1 = cv2.minAreaRect(c1)
        box1 = np.int0(cv2.boxPoints(rect1))
        draw_img = cv2.drawContours(d, [box1], -1, (255, 0, 0), 3)
    except:
        pass
    # cv2.imshow('test3',draw_img)
###############################################

    cp = center_point(box)



        # print(" 第%s个顶点：X:%s,Y:%s" % (i,box[i][0],box[i][1]))
    cp = isPoiWithinPoly(cp,area)
    p0 = isPoiWithinPoly((box[0][0],box[0][1]),area)
    p1 = isPoiWithinPoly((box[1][0],box[1][1]),area)
    p2 = isPoiWithinPoly((box[2][0],box[2][1]),area)
    p3 = isPoiWithinPoly((box[3][0],box[3][1]),area)
    if (cp or p0 or p1 or p2 or p3):
        print("在区域内")

    else:
        print("在区域外")
    
    
    draw_img = cv2.drawContours(d, [box], -1, (0, 0, 255), 3)
    cv2.imshow('test',draw_img)





# 创建边界
area = [[[220,60],[560,110],[530,290],[30,120]]]


vc= cv2.VideoCapture('9.mp4')
if vc.isOpened():
    open, frame = vc.read()
else:
    open =False

while open:
    ret,frame = vc.read()
    if frame is None:
        break
    if ret == True:
        # cv2.imshow('test',main(frame))
        main(frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break
vc.release()
cv2.destroyAllWindows()