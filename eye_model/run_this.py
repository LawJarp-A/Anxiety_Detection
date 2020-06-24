import cv2
from gaze_tracking import GazeTracking
import pandas as pd
import datetime
import math
import numpy as np

def get_data():
    l=[]
    center_left=[]
    center_right=[]
    l_x=[]
    l_y=[]
    r_x=[]
    r_y=[]
    d={'Time':[],'Left eye':[],'Right eye':[]}

    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    while True:
        # We get a new frame from the webcam
        _, frame = webcam.read()

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

        frame = gaze.annotated_frame()
        text = ""

        if gaze.is_blinking():
            text = "Blinking"
            l.append(datetime.datetime.now())
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking right"
            

        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        try:
            l_x.append(gaze.pupil_left_coords()[0])
            l_y.append(gaze.pupil_left_coords()[1])
            r_x.append(gaze.pupil_right_coords()[0])
            r_y.append(gaze.pupil_right_coords()[1])
        except:            
            l_x.append(0)
            l_y.append(0)
            r_x.append(0)
            r_y.append(0)
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        #print((left_pupil,right_pupil))
        try:
            d['Left eye'].append((left_pupil[0],left_pupil[1]))
            d['Right eye'].append((right_pupil[0],right_pupil[1]))
            d['Time'].append(datetime.datetime.now())
        except:
            d['Left eye'].append(0)
            d['Right eye'].append(0)
            d['Time'].append(datetime.datetime.now())

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == 27:
            break


    eye_coordinates=pd.DataFrame(d)
    eye_coordinates.columns=['Time','Left eye','Right eye']
    eye_blinking=pd.Series(l)
    return eye_coordinates#,eye_blinking,center_right,center_left,l_x,l_y,r_x,r_y

def removeOutliers(x, outlierConstant):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if(y>=quartileSet[0] and y<=quartileSet[1]):
            resultList.append(y)
    return resultList

def distract(l_y,c2):
    deltas=[[]]
    sec=[]
    per=[]
    status=0
    for i in range(len(l_y)):
        if(l_y[i]<=(c2-5) or l_y[i]>=(c2+5)):
            if status==0:
                deltas.append([])
                status=1
            deltas[len(deltas)-1].append(both['Time'].iloc[i])
        else:
            status=0
    for i in deltas[1:]:
        sec.append(i[len(i)-1]-i[0])
    for i in deltas[1:]:
        per.append(len(i))
    return sec,per

def score(deltas_x,deltas_y,per_x,per_y):
    s1,s2,s3,s4=0,0,0,0
    for i in deltas_x:
        if i!=0:
            if(i<70):
                s1+=((0.2*i)/70)
            elif(i>=70 and i<=100):
                s1+=(0.2+(((i-70)*0.4)/30))
            elif(i>100):
                s1+=(0.6+(((i-100)*0.4)/50))
    try:
        s1=s1/len(deltas_x)
    except:
        s1=0
    for i in deltas_y:
        if i!=0:
            if(i<70):
                s2+=((0.2*i)/70)
            elif(i>=70 and i<=100):
                s2+=(0.2+(((i-70)*0.4)/30))
            elif(i>100):
                s2+=(0.6+(((i-100)*0.4)/50))
    try:
        s2=s2/len(deltas_y)
    except:
        s2=0
    for i in per_x:
        if(i>=15 and i<=60):
            s3+=((i-15)/45)
        if i>60:
            s3+=1
    try:
        s3=s3/len(per_x)
    except:
        s3=0
    for i in per_y:
        if(i>=15 and i<=60):
            s4+=((i-15)/45)
        if i>60:
            s4+=1
    try:
        s4=s4/len(per_y)
    except:
        s4=0
    return s1,s2,s3*12,s4*12

if __name__=='__main__':
    both=get_data()
    
    left=both['Left eye']
    right=both['Right eye']
    
    l_x=[x[0] if x!=0 else 0 for x in left]
    l_y=[x[1] if x!=0 else 0 for x in left]
    r_x=[x[0] if x!=0 else 0 for x in right]
    r_y=[x[1] if x!=0 else 0 for x in right]
    
    l_c_x=[x[0] if x!=0 else 0 for x in left]
    l_c_y=[x[1] if x!=0 else 0 for x in left]
    r_c_x=[x[0] if x!=0 else 0 for x in right]
    r_c_y=[x[1] if x!=0 else 0 for x in right]
    
    l_c_x=removeOutliers(l_c_x,1.5)
    l_c_y=removeOutliers(l_c_y,1.5)
    r_c_x=removeOutliers(r_c_x,1.5)
    r_c_y=removeOutliers(r_c_y,1.5)
    
    l_x=[sum(removeOutliers(l_x,1.5))/len(removeOutliers(l_y,1.5)) if x==0 else x for x in l_x]
    l_y=[sum(removeOutliers(l_y,1.5))/len(removeOutliers(l_y,1.5)) if x==0 else x for x in l_y]
    r_x=[sum(removeOutliers(r_x,1.5))/len(removeOutliers(r_y,1.5)) if x==0 else x for x in r_x]
    r_y=[sum(removeOutliers(r_y,1.5))/len(removeOutliers(r_y,1.5)) if x==0 else x for x in r_y]
    
    l_c1=sum(l_c_x)/len(l_c_x)
    l_c2=sum(l_c_y)/len(l_c_y)
    r_c1=sum(r_c_x)/len(r_c_x)
    r_c2=sum(r_c_y)/len(r_c_y)
    
    l_deltas_x,l_per_x=distract(l_x,l_c1)
    l_deltas_y,l_per_y=distract(l_y,l_c2)
    l_deltas_x=[i.total_seconds()*60 for i in l_deltas_x]
    l_deltas_y=[i.total_seconds()*60 for i in l_deltas_y]
    r_deltas_x,r_per_x=distract(r_x,r_c1)
    r_deltas_y,r_per_y=distract(r_y,r_c2)
    r_deltas_x=[i.total_seconds()*60 for i in r_deltas_x]
    r_deltas_y=[i.total_seconds()*60 for i in r_deltas_y]
    
    l_final_score=sum(score(l_deltas_x,l_deltas_y,l_per_x,l_per_y))/4
    r_final_score=sum(score(r_deltas_x,r_deltas_y,r_per_x,r_per_y))/4
    final_score=(l_final_score+r_final_score)/2
    
    print(final_score)
    
    

    
