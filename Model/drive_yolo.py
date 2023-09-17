from ultralytics import YOLO
import matplotlib
model = YOLO(r"Model/best.pt")

model.predict(source = "Model\coin_30.jpg" , classes = [0,1,2,3,4,5] ,show = True )