from ultralytics import YOLO 
model = YOLO("/home/serena/Data/Fish/training/fish_all_11m/weights/best.pt")    # pretrained YOLOv8n model 
results = model.predict(source="/home/serena/Downloads/GX014902.MP4",
                        show=False,
                        vid_stride=3,
                        device=0, 
                        save=True,       
                        save_txt=True ) 

