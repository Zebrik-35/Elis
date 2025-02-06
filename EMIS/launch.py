from ultralytics import YOLO

model = YOLO("yolov11_custom_v1_14_25.pt")

model.predict(source = "0", show = True, save = True, conf = 0.05, line_width = 3, save_crop = False, save_txt = False, show_conf = True, show_labels = True)