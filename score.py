import pose_model
import cv2
import math

def norm(k):
    k = k/(math.sqrt(2)*224)
    return k


cap = cv2.VideoCapture('vid.avi')
# Find OpenCV version

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver)  < 3 :
    fps =cap.get(cv2.cv.CV_CAP_PROP_FPS)

else :
    fps = cap.get(cv2.CAP_PROP_FPS)

i = 0
kb = None
temp = []
flag = 0
c = 0
score = []
while(cap.isOpened() and (flag!=2)):
    ret, frame = cap.read()
    if(i == fps):
        try:
            k = pose_model.get_points(frame)
        except:
            flag+=1
            continue
        if(kb and k):
            #Code goes here
            try:

                lw = k[5]
                rw = k[2]
                Lw = kb[5]
                Rw = kb[2]
                if(lw!=(0,0) and rw!=(0,0) and Lw!=(0,0) and Rw!=(0,0)):
                    distR = math.sqrt(((rw[0]-Rw[0])**2)+((rw[1]-Rw[1])**2))
                    distL = math.sqrt(((lw[0]-Lw[0])**2)+((lw[1]-Lw[1])**2))
                    distR = norm(distR)
                    distL = norm(distL)
                    if(distR<0.2 or distL<0.2):
                        c+=1
                    else:
                        if(c>100):
                            score.append(0.9)
                        elif(c in range(80,100)):
                            score.append(0.8)
                        elif(c in range(40,80)):
                            score.append(0.5)
                        else:
                            score.append(0.2)
                        c = 0

            except:
                pass


        kb = k
        i = 0

    i+=1
    if  0xFF == ord('q'):
        break
# print(temp)
# print(c)
# print(fps)
print(score)
cap.release()
cv2.destroyAllWindows()
