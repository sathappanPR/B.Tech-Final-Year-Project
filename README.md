# <img src="https://github.com/sathappanPR/B.Tech-Final-Year-Project/assets/84607354/5b451cb6-ba6f-4fc4-91c8-b9db8781585a" width="40"> Controlling the speed of the simulated car in the CARLA simulator using YOLOv8m
## <img src="https://github.com/sathappanPR/B.Tech-Final-Year-Project/assets/84607354/2508111d-e6b9-4fe1-826f-3f834a3c9d30" width="40"> Simulator Result
* A car has been integrated with a camera that helps capture live feed from the roadside. Once the speed limit signs (30 km/h, 60 km/h, and 90 km/h) are detected, the first frame will display the car's speed in the left corner, while the second frame will show object detection. Initially, the car is traveling at a speed of 33 km/hr.

![Screenshot 2024-04-15 100029](https://github.com/sathappanPR/B.Tech-Final-Year-Project/assets/84607354/186b8221-8de8-4efb-8841-f11558a9d9b6)

* Once the speed limit signboard is detected and crossed, the speed will increase or decrease according to the board detection.

![Screenshot 2024-04-15 100035](https://github.com/sathappanPR/B.Tech-Final-Year-Project/assets/84607354/d427a436-4594-4895-a378-f15d42b04067)

## <img src="https://github.com/sathappanPR/B.Tech-Final-Year-Project/assets/84607354/456a031f-ecf3-444e-91de-e3c26818c2db" width="20"> Web cam Results
* Initially, the YOLO model is created and its performance is tested using the webcam to check the model's accuracy.

![Screenshot 2024-04-03 103015](https://github.com/sathappanPR/B.Tech-Final-Year-Project/assets/84607354/afbd0b87-28d2-4321-83ae-f5720642bae5)

## <img src="https://github.com/sathappanPR/B.Tech-Final-Year-Project/assets/84607354/e087bfd8-19d4-4fd9-8aa2-7dc99d90815a" width="40"> YOLOv8m 
YOLOv8-M is an advanced real-time object detection model that builds upon the YOLO series. It features:

* State-of-the-art backbone and neck architectures.
* An anchor-free split Ultralytics head for improved accuracy and efficiency.
* Optimized tradeoff between accuracy and speed.
* Pre-trained variants specialized for detection, segmentation, pose estimation, classification, and oriented object detection.
  
![Paper](https://github.com/sathappanPR/B.Tech-Final-Year-Project/assets/84607354/60dc0bd0-9028-4f2c-82dd-5636ee9cb5f4)
