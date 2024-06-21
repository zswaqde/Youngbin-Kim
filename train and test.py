import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
excel_path = 'D:/late fusion_2/hemorrhage_diagnosis1.xlsx'
df = pd.read_excel(excel_path)

# Function to preprocess the image
def preprocess_image(image_path, target_size):
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Parameters
IMG_SIZE = (128, 128)
brain_images = []
bone_images = []
labels = []
image_paths = []

# Load images and labels
image_dir = 'D:/late fusion_2/Patients_CT'
for index, row in df.iterrows():
    patient_number = row['PatientNumber']
    if patient_number > 100:
        continue
    
    slice_number = row['SliceNumber']
    
    brain_image_path = os.path.join(image_dir, str(patient_number), 'brain', f'{slice_number}.jpg')
    bone_image_path = os.path.join(image_dir, str(patient_number), 'bone', f'{slice_number}.jpg')
    
    if os.path.exists(brain_image_path) and os.path.exists(bone_image_path):
        if 'HGE' not in brain_image_path:  # Exclude images with 'HGE' in the filename
            brain_image = preprocess_image(brain_image_path, IMG_SIZE)
            bone_image = preprocess_image(bone_image_path, IMG_SIZE)
            if brain_image is not None and bone_image is not None:
                label = row[['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural', 'No_Hemorrhage', 'Fracture_Yes_No']].values
                
                brain_images.append(brain_image)
                bone_images.append(bone_image)
                labels.append(label)
                image_paths.append((brain_image_path, bone_image_path))

# Convert lists to numpy arrays and ensure they have the same length
min_length = min(len(brain_images), len(bone_images), len(labels))
brain_images = np.array(brain_images[:min_length])
bone_images = np.array(bone_images[:min_length])
labels = np.array(labels[:min_length])
image_paths = image_paths[:min_length]

# Normalize images
brain_images = brain_images / 255.0
bone_images = bone_images / 255.0

# Check class distribution
print("Number of brain images:", len(brain_images))
print("Number of bone images:", len(bone_images))
print("Number of labels:", len(labels))

# Convert labels to float32 numpy arrays
labels = np.array(labels, dtype=np.float32)

if len(brain_images) > 0 and len(bone_images) > 0 and len(labels) > 0:
    # Split the data into training (70%), validation (20%), and testing (10%) sets
    X_train_brain, X_temp_brain, X_train_bone, X_temp_bone, y_train, y_temp, train_paths, temp_paths = train_test_split(
        brain_images, bone_images, labels, image_paths, test_size=0.3, random_state=42)
    
    X_val_brain, X_test_brain, X_val_bone, X_test_bone, y_val, y_test, val_paths, test_paths = train_test_split(
        X_temp_brain, X_temp_bone, y_temp, temp_paths, test_size=(1/3), random_state=42)

    # Print test paths
    print("Test Paths:")
    for brain_path, bone_path in test_paths:
        print(f"Brain: {brain_path}, Bone: {bone_path}")

    # Build the CNN model for brain images
    input_brain = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x1 = Conv2D(16, (3, 3), activation='relu')(input_brain)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Conv2D(32, (3, 3), activation='relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Flatten()(x1)

    # Build the CNN model for bone images
    input_bone = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x2 = Conv2D(16, (3, 3), activation='relu')(input_bone)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Conv2D(32, (3, 3), activation='relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Flatten()(x2)

    # Combine the outputs
    combined = concatenate([x1, x2])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(7, activation='sigmoid')(combined)  # Change to sigmoid for multi-label classification

    model = Model(inputs=[input_brain, input_bone], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    # Train the model
    model.fit([X_train_brain, X_train_bone], y_train, epochs=20, batch_size=32, validation_data=([X_val_brain, X_val_bone], y_val))

    # Save the model
    model.save('intracranial_hemorrhage_model.h5')

# Load the pre-trained model
model = load_model('intracranial_hemorrhage_model.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate([X_test_brain, X_test_bone], y_test)
print(f"Test Accuracy: {test_accuracy}, Test Loss: {test_loss}")

# Predict on the test set
predicted_labels = model.predict([X_test_brain, X_test_bone])
predicted_labels = (predicted_labels > 0.5).astype(int)

# Helper function to convert numpy array to a string without dots
def array_to_string(array):
    return str(array).replace('.', '')

# Create a DataFrame to compare actual and predicted labels
comparison = pd.DataFrame({
    'PatientNumber': [path[0].split('/')[-3] for path in test_paths],
    'SliceNumber': [path[0].split('/')[-1].split('.')[0] for path in test_paths],
    'Actual': [array_to_string(label) for label in y_test],
    'Predicted': [array_to_string(label) for label in predicted_labels]
})

# Save the comparison to a CSV file
comparison.to_csv('prediction_comparison.csv', index=False)

# Save the comparison to an Excel file
comparison.to_excel('prediction_comparison.xlsx', index=False)

print(comparison.head())

# Visualization
# Fix the invalid syntax issue by adding commas in the arrays
def fix_array_syntax(array_string):
    return array_string.replace(' ', ', ')

# Apply the fix to both Actual and Predicted columns
comparison['Actual'] = comparison['Actual'].apply(fix_array_syntax).apply(eval)
comparison['Predicted'] = comparison['Predicted'].apply(fix_array_syntax).apply(eval)

# Convert the lists of lists into numpy arrays
actual_array = np.array(comparison['Actual'].tolist())
predicted_array = np.array(comparison['Predicted'].tolist())

# Define the labels
labels = ['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural', 'No_Hemorrhage', 'Fracture_Yes_No']

# Define a function to count occurrences for individual conditions
def count_individual_occurrences(actual, predicted):
    counts = np.zeros((8, 8), dtype=int)  # 7 conditions + 1 for false positives/negatives
    for a, p in zip(actual, predicted):
        for i in range(len(a)):
            counts[i, i] += (a[i] == 1 and p[i] == 1)
            counts[i, -1] += (a[i] == 1 and p[i] == 0)
            counts[-1, i] += (a[i] == 0 and p[i] == 1)
    return counts

# Count occurrences for individual conditions
occurrence_matrix = count_individual_occurrences(actual_array, predicted_array)

# Create a heatmap with actual values on the y-axis and predicted values on the x-axis
plt.figure(figsize=(10, 8))
sns.heatmap(occurrence_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels + ['False Negative'], yticklabels=labels + ['False Positive'])
plt.title('Actual vs Predicted Values Heatmap')
plt.xlabel('Predicted Condition')
plt.ylabel('Actual Condition')
plt.show()