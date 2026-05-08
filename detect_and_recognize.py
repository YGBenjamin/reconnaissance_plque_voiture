from ultralytics import YOLO
from easyocr import Reader
import time
import torch
import cv2
import os
import csv

CONFIDENCE_THRESHOLD = 0.4
COLOR = (0, 255, 0)

def detect_number_plates(image, model, display=False):
    start = time.time()

    detections = model.predict(image)[0].boxes.data
    if detections.shape != torch.Size([0, 6]): #Si le modèle détecte une plaque
        
        boxes = []
        confidences = []

        for detection in detections: #dans le cas où plusieurs plaques sont détectées
            confidence = detection[4] #score de confiance

            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            boxes.append(detection[:4]) #on prend les valeurs de la bounding box
            confidences.append(detection[4])
            #Pour rappel, la forme de detection : tensor([[116.6600,  57.9264, 240.0706, 116.5879,   0.8975,   0.0000]])

        print(f"{len(boxes)} plaques de voitures ont été détectées")

        number_plate_list = [] #On va enregistrer les coordonnées de chaque bounding box des plaques

        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = int(boxes[i][0]),int(boxes[i][1]),int(boxes[i][2]),int(boxes[i][3])
            number_plate_list.append([[xmin, ymin, xmax, ymax]])

            #On va colorer les bounding boxes
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR, 2)
            text = f"Number Plate : {confidences[i]*100:.2f}%"
            cv2.putText(image, text, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

            if display: #Dans ce cas on display les plaques (on les montre)
                number_plate = image[ymin:ymax, xmin:xmax]
                cv2.imshow(f"Number Plate {i}", number_plate)

        end = time.time()

        print(f"Temps utilisé pour détecter les plaques : {(end - start)*1000:.0f} ms")

        return number_plate_list

    else:
        print("No number plate have been detected")
        return []
    

def recognize_number_plate(image_or_path, reader, number_plate_list, write_to_csv=False):

    start = time.time()
    image = cv2.imread(image_or_path) if isinstance(image_or_path, str) else image_or_path

    #print(number_plate_list)
    for i, box in enumerate(number_plate_list):
        np_image = image[box[0][1]:box[0][3], box[0][0]:box[0][2]]

        detection = reader.readtext(np_image, paragraph=True)
       #print(detection)

        if len(detection) == 0:
            text = ""
        else:
            text = str(detection[0][1])

        number_plate_list[i].append(text)

    if write_to_csv:
        csv_file = open("number_plate.csv", "w")
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow(["image_path", "box", "text"])

        for box, text in number_plate_list:
            csv_writer.writerow([image_or_path, box, text])

        csv_file.close()

    end = time.time()

    print(f"Le temps pour reconnaître les plaques est de : {(end - start) *1000:.0f} ms")
    return number_plate_list

if __name__ == "__main__":
    model = YOLO("./runs/detect/train/weights/best.pt")
    video_path = "datasets/videos/traffic.mp4"
    file_path = "datasets/images/train/f2f0b6f0-77.jpeg"
    file_path=video_path
    reader = Reader(["en"],)

    _, file_extension = os.path.splitext(file_path)
    if file_extension in [".jpg", ".jpeg", ".png"]:
        print("Traitement de l'image...")
        image = cv2.imread(file_path)

        number_plate_list = detect_number_plates(image, model, display=False)
        
        #cv2.imshow("image", image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        if number_plate_list != []:
            number_plate_list = recognize_number_plate(file_path, reader, number_plate_list, write_to_csv=True)

            for box, text in number_plate_list:
                cv2.putText(image, text, (box[0], box[3]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif file_extension in [".mp4", ".mkv", ".avi", ".wmv", ".mov"]:
        print("Traitement de la vidéo...")

        video_cap = cv2.VideoCapture(file_path)
        frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

        while True:
            start = time.time()
            success, frame = video_cap.read()

            if not success:
                print("Il ne reste plus d'images à traiter"
                      "Sortie")
                break

            number_plate_list = detect_number_plates(frame, model)

            if number_plate_list != []:
                number_plate_list = recognize_number_plate(frame, reader, number_plate_list, write_to_csv=True)

                for box, text in number_plate_list:
                    cv2.putText(frame, text, (box[0], box[3]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

            end = time.time()
            fps = f"FPS: {1 / (end-start):.2f}"
            cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

            cv2.imshow("Output", frame)
            writer.write(frame)

            if cv2.waitKey(10) == ord("q"):
                break

        video_cap.release()
        writer.release()

        cv2.destroyAllWindows()
