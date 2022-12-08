import cv2
import numpy as np
import time

def sliding_window(img,stepSize=1,windowSize=1):
    half = int(windowSize/2)
    for j in range(half,img.shape[0]-half,stepSize):
        print("\rProcessing.. %d%% complete"%(j / (img.shape[0]-half) * 100), end="", flush=True)
        for i in range(half,img.shape[1]-half,stepSize):
            yield (i,j,img[j-half:j+half,i-half:i+half])

def downscale(img):
    result = cv2.resize(img,(img.shape[0]//2,img.shape[1]//2))
    return result

def upscale(img):
    result = cv2.resize(img,(img.shape[0]*2,img.shape[1]*2))
    return result

def match(window,right,i,j,method,windowSize,offset):
    disparity = 0
    half = int(windowSize/2)
    if method == "SSD":
        minSSD = 99999999
        temp = 0
        lf = i-offset if i-offset > half else half
        rt = i+offset if i+offset < right.shape[1]-half else right.shape[1]-half
        for k in range(lf,rt):
            temp = np.sum((window-right[j-half:j+half,k-half:k+half])**2)
            if temp < minSSD:
                minSSD = temp
                disparity = i - k
    elif method == "SAD":
        minSAD = 99999999
        temp = 0
        lf = i-offset if i-offset > half else half
        rt = i+offset if i+offset < right.shape[1]-half else right.shape[1]-half
        for k in range(lf,rt):
            temp = np.sum(np.abs(window-right[j-half:j+half,k-half:k+half]))
            if temp < minSAD:
                minSAD = temp
                disparity = i - k
    elif method == "NCC":
        maxZNCC = -99999
        temp = 0
        lf = i-offset if i-offset > half else half
        rt = i+offset if i+offset < right.shape[1]-half else right.shape[1]-half
        window_m = window - window.mean(axis=0)
        for k in range(lf,rt):
            right_m = right[j-half:j+half,k-half:k+half] - right[j-half:j+half,k-half:k+half].mean(axis=0)
            temp = np.sum((window_m/np.linalg.norm(window_m))*(right_m/np.linalg.norm(right_m)))
            if temp > maxZNCC:
                maxZNCC = temp
                disparity = i - k
    return disparity

def main():
    resize = True
    imgDir = "data/"
    #imgList = ["tsukuba1.jpg", "tsukuba2.jpg"]
    imgList = ["moebius1.png", "moebius2.png"]
    
    img1 = cv2.imread(imgDir + imgList[0],cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(imgDir + imgList[1],cv2.IMREAD_GRAYSCALE)

    if resize:
        for  i in range(2):
            img1 = downscale(img1)
            img2 = downscale(img2)
        
    
    img1 = img1/255
    img2 = img2/255
    
    windowSize = 6
    step = 1
    offset = 30
    
    disparities = np.zeros((img1.shape),np.uint8)
    
    start = time.time()
    
    for i,j,window in sliding_window(img1,step,windowSize):
        disparities[j,i] = abs(match(window,img2,i,j,"SAD",windowSize,offset) * (255 / offset))
    
    end = time.time()
    
    print('\n',"Time spent: ",end-start)
    # if resize:
    #     for i in range(2):
    #         disparities = upscale(disparities)
    
    cv2.imwrite("mobi.jpg",disparities)
    # cv2.imshow("img",disparities)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()