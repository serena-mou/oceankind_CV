from ultralytics import YOLO 
model = YOLO("/home/serena/repos/oceankind_CV/training/urchin/GB_1_2/weights/best.pt")    # pretrained YOLOv8n model 
results = model.predict(source="/home/serena/Data/urchins/2025_urchin_data/160625_GaboIs/1606_GI_1/",
                        show=False,
                        classes = [0],
                        name="1606GI1",
                        conf=0.5,
                        device=0, 
                        save=True,       
                        save_txt=True )     
