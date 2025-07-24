from ultralytics import YOLO 
model = YOLO("/home/serena/Data/Urchins/OK_CV/urchin_6/weights/best.pt")    # pretrained YOLOv8n model 
results = model.track(source="/home/serena/Data/Urchins/videos/GH011277.MP4",
                        show=True,
                        conf=0.3,
                        iou=0.5,
                        #vid_stride=3,
                        device=0, 
                        save=True)#,       
                        #save_txt=True ) 

