import cv2 
import time
import numpy as np

# Font for puttext funcion
font = cv2.FONT_HERSHEY_PLAIN
# Timer and fps variables initialized
fps = 0.0
timeend = 0.0
elapsed = 0.0

# define a video capture object 
vid = cv2.VideoCapture(0)
timestart = time.time()
  
while(True):
    # Frame read from camera
    islive, frame = vid.read()
    # Image converted to grayscale value for brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Image converted to hsv for red detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Creating mask for pixels within hsv red regions

    # Range for lower red
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Range for upper range
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)

    # Combined upper and lower red region mask
    mask1 = mask1+mask2

    # "Reddest" pixel found using mask
    minvalr, maxvalr, minlocr, maxlocr = cv2.minMaxLoc(hsv[:,:,2], mask1)
    
    # Brightest pixel found
    minval, maxval, minloc, maxloc = cv2.minMaxLoc(gray)

    # Brightest spot using for loop
    """
    maxval = 0
    for i in range(int(vid.get(4))):
        for j in range(int(vid.get(3))):
            if gray[i,j] > maxval:
                maxval = gray[i,j]
                maxloc = (j, i)

    """

    # Circle drawn around the brightest pixel
    cv2.circle(frame, maxloc, 5, (255, 0, 0), 2)
    cv2.drawMarker(frame, maxlocr, (0, 0, 255), cv2.MARKER_SQUARE, 10, 2)
    
  
    # Time and fps calculated
    timeend = time.time()
    elapsed = timeend - timestart
    fps = int(1 / elapsed)
    # Text written, numerical value is seperate from text to keep the text fixed in place
    cv2.putText(frame,f'{fps}',(350,450), font, 4,(255,255,255),2,cv2.LINE_4)
    cv2.putText(frame,f'fps',(500,450), font, 4,(255,255,255),2,cv2.LINE_4) 
    # Display the resulting frame 
    cv2.imshow('frame', frame)
    # Time started after the frame is shown
    timestart = time.time()
    
    
    # 'q' for quit
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

  
print(np.average(timers[1:]))
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 