import tf_pose
print("OK")
def get_points(k):

  li = tf_pose.infer(k)
  li = li[0][0]
  coco_ids = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
  keypoints = {}
  pt = 0
  for i in coco_ids:
    keypoints[i]=(li[pt],li[pt+1])
    pt+=3
  req = {}
  req["left_shoulder"]=keypoints[5]
  req["right_shoulder"]=keypoints[2]
  req["left_elbow"]=keypoints[6]
  req["right_elbow"]=keypoints[3]
  req["left_wrist"]=keypoints[7]
  req["right_wrist"]=keypoints[4]
  return (req)



  


