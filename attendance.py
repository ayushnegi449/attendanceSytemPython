import cv2
import numpy as np
import os
from sklearn.metrics import accuracy_score
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def distance(x, X):
    return np.sqrt(np.sum((x - X) ** 2))

def KNN(X_train, Y_train, x_test, K=5):
    m = X_train.shape[0]
    x_test = x_test.reshape((1, -1))
    val = []
    for i in range(m):
        xi = X_train[i].reshape((1, -1))
        dist = distance(x_test, xi)
        val.append((dist, Y_train[i]))
    val = sorted(val, key=lambda x: x[0])[:K]
    val = np.asarray(val)
    new_vals = np.unique(val[:, 1], return_counts=True)
    index = new_vals[1].argmax()
    output = new_vals[0][index]
    return output

cap = cv2.VideoCapture(0)
path = "C:\\Users\\ayush\\Desktop\\AttendanceProject\\haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(path)
dirpath = "C:\\Users\\ayush\\Desktop\\AttendanceProject\\data"
face_data = []
labels = []
names = {}
class_id = 0
face_section = np.zeros((100,100), dtype='uint8')
attendance = {}  # Dictionary to store attendance

for file in os.listdir(dirpath):
    if file.endswith(".npy"):
        data_item = np.load(os.path.join(dirpath, file))
        names[class_id] = file[:-4]
        face_data.append(data_item)
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)


face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape(-1, 1)


scaler = StandardScaler()
face_dataset_scaled = scaler.fit_transform(face_dataset)

pca = PCA(n_components=0.95)
face_dataset_pca = pca.fit_transform(face_dataset_scaled)
face_dataset_pca = face_dataset_pca.astype(np.float32)
face_labels = face_labels.astype(np.int32)
knn = cv2.ml.KNearest_create()
knn.train(face_dataset_pca, cv2.ml.ROW_SAMPLE, face_labels)

true_labels = []
pred_labels = []

csv_file = open("attendance.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Name", "Attendance"])

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret == False:
        continue
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3])
    for face in faces[-1:]:
        x, y, w, h = face
        face_section = gray_frame[y:y+h, x:x+w]
        face_section = cv2.resize(face_section, (100, 100))
        face_section = face_section.flatten().reshape(1, -1).astype(np.float32)
        face_section_scaled = scaler.transform(face_section)
        face_section_pca = pca.transform(face_section_scaled)
        face_section_pca = face_section_pca.astype(np.float32)  # Convert to float32

        _, result, _, _ = knn.findNearest(face_section_pca, k=5)

        pred_label = int(result[0][0])
        pred_name = names[pred_label]
        cv2.putText(frame, pred_name, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if pred_label not in attendance:  # Check if attendance has been marked already
            attendance[pred_label] = True
            csv_writer.writerow([pred_name, "Present"])
            true_labels.append(0)
            pred_labels.append(pred_label)

    cv2.imshow("camera", frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

csv_file.close()


