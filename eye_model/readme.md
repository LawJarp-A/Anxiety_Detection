# EYE-TRACKING Model:

The primary model of eye gaze is due to the courtesy of https://github.com/antoinelame/GazeTracking

Follow the following steps to run the program:

1. pip install -r requirements.txt.
NOTE:If some problem arises installing dlib, use dlib-19.19.0.tar.gz(in the ey_moedl folder) and refer https://www.youtube.com/watch?v=pHrgi8QLcKk&feature=youtu.be (in windows 10) 
or refer to https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/

2. Run the python file 'run_this.py' to get input from the webcam.For a video input , change the 0 in run_this.py to the videoname.mp4,
i.e. webcam = cv2.VideoCapture(0) to 
webcam = cv2.VideoCapture('videoname.mp4') 
where,videoname is the name of the video file

3. Press 'esc' on the keyboard to interept the input.

## Synopsis for eye model:

1. Eye frame is extracted from the face,using the particular landmarks defined using the pre-trained model shape_predictor_68_face_landmarks.dat.

2. It is converted into a grayscale image and uses the pupil detection algorithm by finding the best binarization threshold value for the person and the webcam(thanks to https://github.com/antoinelame/GazeTracking).

3. The pupil is extracted using contours and the center is found out using cenroid method of that contour i.e.
Cx=M10/M00
Cy=M01/M00
where,M is Image Moment which is a particular weighted average of image pixel intensities, with the help of which we can find some specific properties of an image, like radius, area, centroid etc

4. The video is converted into many frames and centroid of the pupil is found of both eyes of each frame.

## Synopsis for Anxiety Score:A score between 0 to 1 is assigned based on the following:-

1.The coordinates of the eye looking at the center is found by mean and the amount of distraction is found by calculating the spread of the coordinates from the mean.

2.Before finding the mean,outliers are removed using quartile method as to avoid the reduce the effect of the outliers on the mean.Even the data of certain coordinates which couldn't be recorded are filled by the mean.

3.Scores normalized between zero and 1 w.r.t avoiding eye contact (in this case, looking straight/center) for 15-60 seconds.

4.Changes in position of the eye in measured every frame. Scores are assigned as following:

               0-70 times per minute.                     0.0 to 0.2 

               70-100 times per minute.                   0.4 to 0.6     
   
               100-150 times per minute.                  0.6 to 1.0  

5.Mean of all above scores and of both eyes is the final score.
