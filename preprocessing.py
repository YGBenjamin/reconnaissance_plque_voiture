from sklearn.model_selection import train_test_split
import cv2
import os
import yaml

root_dir = "./car-number-plate/"
valid_formats = [".png", ".jpg", ".jpeg", ".txt"]

def file_paths(root, valid_formats):

    file_paths = []

    for dirpath, dirnames, filenames in os.walk(root):

        for filename in filenames:
            extension = os.path.splitext(filename)[1].lower()

            if extension in valid_formats:
                file_path = os.path.join(dirpath, filename)
                file_paths.append(file_path)
        
    return file_paths

#image_paths = file_paths(root_dir + "images", valid_formats[:3])
#label_paths = file_paths(root_dir + "labels", valid_formats[-1])

#X_train, X_valid_test, y_train, y_valid_test = train_test_split(image_paths, label_paths, test_size=0.3, random_state=42)
#X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size=0.7, random_state=42)

def write_to_file(images_path, labels_path, X):
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    for image_path in X:
        img_name = image_path.split("/")[-1].split(".")[0]
        img_ext = image_path.split("/")[-1].split(".")[1]

        image = cv2.imread(image_path)
        cv2.imwrite(f"{images_path}/{img_name}.{img_ext}", image)

        f = open(f"{labels_path}/{img_name}.txt", "w")
        label_file = open(f"{root_dir}/labels/{img_name}.txt", "r")
        f.write(label_file.read())
        f.close()
        label_file.close()

#write_to_file("datasets/images/train", "datasets/labels/train", X_train)
#write_to_file("datasets/images/valid", "datasets/labels/valid", X_valid)
#write_to_file("datasets/images/test", "datasets/labels/test", X_test)

data = {
    "path": "./datasets/",
    "train": "images/train",
    "val": "images/valid",
    "test": "images/test",

    "names": ["Number Plate"]
}

with open("number-plate.yaml", "w") as f:
    yaml.dump(data, f)