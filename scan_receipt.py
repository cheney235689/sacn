import numpy as np
# =============================================================================
# import argparse
# =============================================================================
import cv2 

# find the rectangle
def order_points(pts):

    # 4 points
    rect = np.zeros((4, 2), dtype = "float32")
    #rect[0] = left_top , rect[1] = right_top , recp[2] = right_bot , rect[3] = left_bot
    
    #step.1: Calculate the rect[0] , rect[1]  
    s = pts.sum(axis = 1)
    
    rect[0] = pts[np.argmin(s)]
    
    rect[2] = pts[np.argmax(s)]
    
    #step.2: Calculate the rect[2] , rect[3]  
    
    diff = np.diff(pts, axis = 1)
    
    rect[1] = pts[np.argmin(diff)]
    
    rect[3] = pts[np.argmax(diff)]

    return rect




# get input coordinate
def four_point_transform(image, pts):

    rect = order_points(pts)
    
    (tl, tr, br, bl) = rect
    
    #get  w, h 
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    
    maxHeight = max(int(heightA), int(heightB))

    # coord after transformation
    
    dst = np.array([
    
    [0, 0],
    
    [maxWidth - 1, 0],
    
    [maxWidth - 1, maxHeight - 1],
    
    [0, maxHeight - 1]], dtype="float32")
    

    #Calculate the transformation matrix 
    #Homogeneous coordinates: use N+1 dimensions to represent N-dimensional coordinates [kx,ky,k]
    
    M = cv2.getPerspectiveTransform(rect, dst)
    
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # transformed result
    return warped


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    
    dim = None
    
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
    
        return image
    
    if width is None:
    
        r = height / float(h)
        
        dim = (int(w * r), height)
    
    else:
    
        r = width / float(w)
        
        dim = (width, int(h * r))
    
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

image = cv2.imread('711.jpg')



ratio = image.shape[0] / 500.0

# image.shape[0], the vertical size of the image

# image.shape[1], the horizontal size of the image

# image.shape[2], number of image channels

orig = image.copy()
image = resize(orig, height=500)# scaling


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (3, 3), 0)
edged = cv2.Canny(gray, 120, 200)


print("STEP 1: edge detection  ")

cv2.imshow("Image", image)

cv2.imshow("Edged", edged)

cv2.waitKey(0)

cv2.destroyAllWindows()


# contour detection

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]#select those bigger contours


# retreval contours

for c in cnts:
# contour approximation
    peri = cv2.arcLength(c, True)

    
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)# 輪廓 , 輪廓精度 , 越小可能是多邊形 , 越大可能是矩形

    print(len(approx))
    if len(approx) == 4:
        screenCnt = approx
        print(screenCnt) 
        break


print(len(approx))
print("STEP 2: Get contours ")

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

cv2.imshow("Outline", image)

cv2.waitKey(0)

cv2.destroyAllWindows()



# Perspective 

warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# tresholding
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]

cv2.imwrite('scan.jpg', ref)

# Result

print("STEP 3: transfromation ")

cv2.imshow("Original", resize(orig, height = 650))

cv2.imshow("Scanned", resize(ref, height = 650))

cv2.waitKey(0)
cv2.destroyAllWindows()