import os
import cv2
import numpy as np

from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

# -----------------------------
# SIMPLE TUMOR SEGMENTATION
# -----------------------------
def segment_tumor(img):
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    return thresh

# -----------------------------
# GLCM FEATURE EXTRACTION
# -----------------------------
def extract_glcm_features(img):
    glcm = graycomatrix(
        img,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    return [contrast, correlation, energy, homogeneity]

# -----------------------------
# TUMOR SIZE & LOCATION ESTIMATION
# -----------------------------
def tumor_size_location(segmented_img):
    contours, _ = cv2.findContours(
        segmented_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return 0, 0  # No tumor

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    # Size estimation
    if area < 500:
        size = 1  # Small
    elif area < 2000:
        size = 2  # Medium
    else:
        size = 3  # Large

    # Location estimation
    M = cv2.moments(c)
    if M["m00"] == 0:
        location = 0
    else:
        cx = int(M["m10"] / M["m00"])
        if cx < 42:
            location = 1  # Left
        elif cx > 86:
            location = 3  # Right
        else:
            location = 2  # Center

    return size, location

# -----------------------------
# LOAD DATASET
# -----------------------------
features = []
labels = []

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")


class_map = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
    "no_tumor": 3
}

for class_name, label in class_map.items():
    folder = os.path.join(DATASET_PATH, class_name)

    if not os.path.exists(folder):
        print(f"[ERROR] Folder not found: {folder}")
        continue

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        img = preprocess_image(img_path)
        segmented = segment_tumor(img)

        glcm_features = extract_glcm_features(img)
        size, location = tumor_size_location(segmented)

        final_features = glcm_features + [size, location]

        features.append(final_features)
        labels.append(label)

# -----------------------------
# PREPARE DATA
# -----------------------------
X = np.array(features)
y = np.array(labels)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# KNN CLASSIFIER
# -----------------------------
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric='euclidean'
)

knn.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = knn.predict(X_test)

print("CONFUSION MATRIX:\n")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Glioma", "Meningioma", "Pituitary", "No Tumor"]
))

# -----------------------------
# TEST A NEW MRI IMAGE
# -----------------------------
def predict_new_image(img_path, model, scaler):
    img = preprocess_image(img_path)
    segmented = segment_tumor(img)

    glcm_features = extract_glcm_features(img)
    size, location = tumor_size_location(segmented)

    final_features = glcm_features + [size, location]
    final_features = np.array(final_features).reshape(1, -1)

    final_features = scaler.transform(final_features)

    prediction = model.predict(final_features)[0]

    class_names = {
        0: "Glioma Tumor",
        1: "Meningioma Tumor",
        2: "Pituitary Tumor",
        3: "No Tumor Detected"
    }

    size_map = {
        0: "Not Applicable",
        1: "Small",
        2: "Medium",
        3: "Large"
    }

    location_map = {
        0: "Not Applicable",
        1: "Left",
        2: "Center",
        3: "Right"
    }

    print("\n--- Prediction Result ---")
    print("Class:", class_names[prediction])
    print("Estimated Tumor Size:", size_map[size])
    print("Estimated Tumor Location:", location_map[location])


# 🔹 Example test
test_image_path = "test_images/sample_mri.jpg"
predict_new_image(test_image_path, knn, scaler)
