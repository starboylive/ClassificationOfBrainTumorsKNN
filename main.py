import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve




def extract_ml_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.resize(img, (128, 128))
    img = cv2.GaussianBlur(img, (5, 5), 0)

    glcm = graycomatrix(
        img,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    features = []
    for prop in ["contrast", "correlation", "energy", "homogeneity"]:
        vals = graycoprops(glcm, prop)
        features.append(np.mean(vals))
        features.append(np.std(vals))

    return features


def analyze_tumor(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))

    _, thresh = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None, None

    tumor = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(tumor)

    pixels = int(area)
    if pixels < 500:
        size = f"Small ({pixels} pixels)"
    elif pixels < 2000:
        size = f"Medium ({pixels} pixels)"
    else:
        size = f"Large ({pixels} pixels)"

    h, w = img.shape
    M = cv2.moments(tumor)

    if M["m00"] == 0:
        location = "Unknown"
    else:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        if cx < w * 0.45:
            hemi = "Left Cerebral Hemisphere"
        elif cx > w * 0.55:
            hemi = "Right Cerebral Hemisphere"
        else:
            hemi = "Midline Structure"

        if cy < h * 0.30:
            lobe = "Frontal Lobe"
        elif cy < h * 0.55:
            lobe = "Parietal Lobe"
        elif cy < h * 0.75:
            lobe = "Temporal Lobe"
        else:
            lobe = "Occipital Lobe"

        if hemi == "Midline Structure":
            location = "Midline – Corpus Callosum / Ventricular Region"
        else:
            location = hemi + " – " + lobe

    grade = "High Grade" if area > 1500 else "Low Grade"

    return size, location, grade


def load_dataset(dataset_path):
    X, y = [], []

    class_map = {
        "glioma": 0,
        "meningioma": 1,
        "pituitary": 2,
        "notumor": 3
    }

    for cls, label in class_map.items():
        folder = os.path.join(dataset_path, cls)
        if not os.path.exists(folder):
            continue

        for img in os.listdir(folder):
            if img.lower().endswith((".jpg", ".png", ".jpeg")):
                feats = extract_ml_features(os.path.join(folder, img))
                if feats is not None:
                    X.append(feats)
                    y.append(label)

    return np.array(X), np.array(y)


def train_hybrid_model(dataset_path):
    X, y = load_dataset(dataset_path)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    svm = SVC(kernel="rbf", C=10, gamma=0.1, probability=True)
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")

    svm.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    svm_prob = svm.predict_proba(X_test)
    knn_prob = knn.predict_proba(X_test)

    final_prob = (svm_prob + knn_prob) / 2
    preds = np.argmax(final_prob, axis=1)

    print("\nHYBRID MODEL ACCURACY:",
          accuracy_score(y_test, preds) * 100, "%")

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        y_test,
        preds,
        target_names=["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    ))

    # =============================
    # CONFUSION MATRIX
    # =============================
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    )
    disp.plot(cmap="Blues")
    plt.title("Hybrid Model - Confusion Matrix")
    plt.show()

    # =============================
    # ROC CURVE (MULTI-CLASS)
    # =============================
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], final_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i, name in enumerate(["Glioma", "Meningioma", "Pituitary", "No Tumor"]):
        plt.plot(fpr[i], tpr[i], label=f"{name} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Hybrid SVM + KNN")
    plt.legend()
    plt.show()

    # =============================
    # TRAIN vs VALIDATION ACCURACY
    # =============================
    train_sizes, train_scores, val_scores = learning_curve(
        svm, X_train, y_train,
        cv=5, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5)
    )

    train_acc = np.mean(train_scores, axis=1)
    val_acc = np.mean(val_scores, axis=1)

    plt.plot(train_sizes, train_acc, marker="o", label="Training Accuracy")
    plt.plot(train_sizes, val_acc, marker="o", label="Validation Accuracy")
    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy (SVM)")
    plt.legend()
    plt.show()


    return svm, knn, scaler


def predict_image(img_path, svm, knn, scaler):
    feats = extract_ml_features(img_path)
    feats = scaler.transform([feats])

    prob = (svm.predict_proba(feats) +
            knn.predict_proba(feats)) / 2

    label = np.argmax(prob)
    confidence = np.max(prob) * 100

    class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

    print("\n================ FINAL REPORT ================")
    print("Tumor Type :", class_names[label])

    if label != 3:
        size, location, grade = analyze_tumor(img_path)
        print("Tumor Size :", size)
        print("Location   :", location)
        print("Tumor Grade:", grade)

    print(f"Confidence : {confidence:.2f}%")
    print("==============================================")



if __name__ == "__main__":

    DATASET_PATH = r"C:\Users\these\Downloads\archive (1)\Training"
    TEST_IMAGE = r"C:\Users\these\Downloads\archive (1)\Testing\meningioma\Te-me_0265.jpg"

    svm, knn, scaler = train_hybrid_model(DATASET_PATH)
    predict_image(TEST_IMAGE, svm, knn, scaler)
