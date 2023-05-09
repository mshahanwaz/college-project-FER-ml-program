import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt

# Path of the model
MODEL_PATH = r"FER_Model.h5"

# Load the model
model = load_model(MODEL_PATH)

validation_data_dir = r'dataset/test'
validation_datagen = ImageDataGenerator(rescale=1./255)
num_classes = 5
img_rows,img_cols = 48,48
batch_size = 32
activation_function = 'fer+'

# Generate predictions for the test data
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Get the true labels and predictions for the validation set
Y_pred = model.predict(validation_generator, validation_generator.samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
y_true = validation_generator.classes

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot the confusion matrix
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')

# Calculate total accuracy
acc = accuracy_score(y_true, y_pred)

print(f"Accuracy: {acc*100:.2f}%")

# Calculate precision, recall, and F1-score
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Precision: {precision*100:.2f}%")

print(f"Recall: {recall*100:.2f}%")

print(f"F1-score: {f1*100:.2f}%")

# Generate the classification report
cls_report = classification_report(y_true, y_pred, target_names=validation_generator.class_indices.keys())
print(cls_report)


with open('scores.txt', 'a') as f:
   f.write(f"{activation_function} Accuracy: {acc*100:.2f}% Precision: {precision*100:.2f}% Recall: {recall*100:.2f}% F1-score: {f1*100:.2f}%\n")


plt.show()

# Accuracy is the proportion of correct predictions (true positives and true negatives) out of all predictions. It is a common measure of a model's performance, but it may not be appropriate for imbalanced datasets.
# Precision represents the proportion of true positive results (correctly predicted positive samples) out of all positive predictions (total predicted positive samples). In other words, precision indicates how often the model is correct when it predicts that a sample belongs to a certain class.
# Recall represents the proportion of true positive results out of all actual positive samples. In other words, recall indicates how well the model is able to identify all positive samples.
# F1-score is the harmonic mean of precision and recall. It is a measure of a model's accuracy that considers both precision and recall. F1-score is useful when both precision and recall are important and there is a trade-off between them.
