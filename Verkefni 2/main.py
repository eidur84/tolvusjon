import cv2 
import time
import numpy as np
import warnings

# Ignooring polyfit warnings
warnings.simplefilter('ignore', np.RankWarning)

# Font for puttext funcion
font = cv2.FONT_HERSHEY_PLAIN
# Global variables

maxinl = 0
nrAvg = 0
fps = 0.0
timestart = time.time()
timeend = 0.0
elapsed = 0.0
fpsAvg = np.zeros(10)

# If only one frame is going to be run for testing
test = True

## Functions ##

# Random integer generated in the range zero to number of edge pixels
def randomnr(maxnr: int, n: int):
    if maxnr == 0:
        print("Number must be higher than zero!")
        return 0
    else:
        return np.random.randint(0, maxnr, n)

def getframe(video):
    _, frame = video.read()
    
    return frame

def makegray(frame):
    grays = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return grays

def drawline(frame, pt1, pt2, colr=False):
    if colr:
        frame = cv2.line(frame,pt1,pt2,(0,0,255),2)
    else:
        frame = cv2.line(frame,pt1,pt2,(255,0,0),2)
    return frame

def randomPts(nredge, edgList):
        random = randomnr(nredge, 2)
        # Random points selected from the edge pixel list
        pt1 = (edgList[random[0]][0], edgList[random[0]][1]) 
        pt2 = (edgList[random[1]][0], edgList[random[1]][1])

        return pt1, pt2

# finding coefficient
def coeffc(pt1, pt2):
    coef = np.zeros(3)
    coef[:2] = np.polyfit(pt1, pt2, 1)
    coef[2] = np.dot(pt1,coef[:2])

    return coef

# Point lying within range of line
def pointsWhithin(edges, boundary, coef):   
    d = abs(((np.dot(edges,coef[:2]) + coef[2])) / (np.sqrt(np.square(coef[0]) + np.square(coef[1]))))
    idx = np.where(d <= boundary)

    return idx[0], idx[0].shape[0]


# Gets fps, uses average over 5 values
def getFps(start, end, nrAvg, display: bool):
    end = time.time()
    elapsed = end - start
    if display:
        print(elapsed)
    fpsAvg[nrAvg] = int(1 / elapsed)
    nrAvg +=1
    start = time.time()
    if nrAvg == 9:
        nrAvg = 0
    return np.average(fpsAvg), nrAvg, start, end

# Edge pivels extracted using canny
def edgeIdx(frame, trsL, trsH):

    # Frame scaled
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    # Image converted to grayscale value for brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Canny used to extract edges
    #edge = cv2.Canny(gray, 100, 220)
    # Webcam thresholds
    edge = cv2.Canny(gray, trsL, trsH)
    # Edges extracted
    edgList = np.nonzero(edge)
    edgH = edgList[1]
    edgW = edgList[0]
    # Index values put into two column vectors - [H W]
    edgList = np.stack((edgH, edgW), axis=1)
    # Number of edge pixels
    nredge = edgList.shape[0]

    return edgList, edge, nredge

# The ransac algorithm (sort of)
def ransac(frame, iterations):
    # Edges found using edge function. Returns two column matrix [H W], Canny edges and total number of egde pixels
    edgList, edge, nredge = edgeIdx(frame, 50, 150)
    if nredge >=2:
        maxinliers = 300
        maxpt1 = (0,0)
        maxpt2 = (0,0)
        for i in range(iterations):
            pt1, pt2 = randomPts(nredge, edgList)
            coef = coeffc(pt1,pt2)
            #edge = drawline(edge, pt1, pt2)
            #edge = cv2.circle(edge, pt1, 10, (0,0,255), thickness=5, lineType=cv2.LINE_AA)
            idx, inliers = pointsWhithin(edgList, 30, coef)
            if inliers > maxinliers:
                maxinliers = inliers
                maxpt1 = (edgList[idx][0,0], edgList[idx][0,1])
                maxpt2 = (edgList[idx][0,-1], edgList[idx][0,-1])

        return edge, idx, maxpt1, maxpt2, maxinliers

    else:
        return edge, _, (0,0), (1,1), 0

            # Lines drawn from the two points
            
# define a video capture object
#vid = cv2.VideoCapture(0)
vid = cv2.VideoCapture("rtsp://admin:eizi!kil@192.168.1.4:8554/live")
  
while(True):
    # Frame read from camera
    _, frame = vid.read()
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    # Ransac run
    edge, idx, pt1, pt2, inl = ransac(frame, 200)
    # Time and fps calculated
    fps, nrAvg, timestart, timeend = getFps(timestart, timeend, nrAvg, False)
    # Text written, numerical value is seperate from text to keep the text fixed in place
    cv2.putText(edge,f'{fps}',(425,450), font, 4,(255,0,0),2,cv2.LINE_4)
    #cv2.putText(edge,f'fps',(500,450), font, 4,(255,255,255),2,cv2.LINE_4)
    # Display the resulting frame
    #frame = drawline(frame, pt1, pt2, colr=True)
    edge = drawline(edge, pt1, pt2)
    #cv2.imshow('Rammi', frame)
    cv2.imshow('Edge', edge)

    test = False
        
    # 'q' for quit
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break


# After the loop release the cap object 
#vid.release() 
# Destroy all the windows
#cv2.destroyAllWindows() 