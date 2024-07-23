import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Define paths
dataset_path = r'D:\KSU\SUMMER 2024\NN+DL\brain tumor\archive (1)'
training_path = os.path.join(dataset_path, 'Training')
testing_path = os.path.join(dataset_path, 'Testing')
categories = ['glioma', 'notumor', 'pituitary', 'meningioma']

# Function to display original and processed images
def process_and_display_images(category, num_images=5):
    category_path = os.path.join(training_path, category)
    all_images = os.listdir(category_path)
    random_images = random.sample(all_images, num_images)

    plt.figure(figsize=(15, 10))

    # Display original images
    for i, img_name in enumerate(random_images):
        img_path = os.path.join(category_path, img_name)
        img = Image.open(img_path)
        img = img.convert('RGB')  # Convert to RGB if grayscale
        plt.subplot(2, num_images, i + 1)
        plt.imshow(img)
        plt.title(f'Original: {category}')
        plt.axis('off')

    # Display processed images
    for i, img_name in enumerate(random_images):
        img_path = os.path.join(category_path, img_name)
        img = Image.open(img_path)
        img = img.convert('RGB')  # Convert to RGB if grayscale
        img = img.resize((150, 150))  # Resize image
        img_array = np.array(img) / 255.0  # Normalize image
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(img_array)
        plt.title(f'Resized/Rescaled: {category}')
        plt.axis('off')

    plt.show()

# Display images for each category
for category in categories:
    process_and_display_images(category)

# Define ImageDataGenerator for training and testing
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    training_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    testing_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important for evaluation
)

# Plot image distribution in training and test sets
def plot_image_distribution(path, title):
    category_counts = {}
    for category in categories:
        category_path = os.path.join(path, category)
        category_counts[category] = len(os.listdir(category_path))

    plt.figure(figsize=(10, 5))
    plt.bar(category_counts.keys(), category_counts.values(), color='skyblue')
    plt.xlabel('Category')
    plt.ylabel('Number of Images')
    plt.title(f'Image Distribution in {title} Set')
    plt.show()

plot_image_distribution(training_path, 'Training')
plot_image_distribution(testing_path, 'Test')

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    epochs=20,
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {test_acc:.2f}')

# Additional evaluation metrics
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

print('Classification Report')
print(classification_report(y_true, y_pred, target_names=categories))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
cm_display.plot()
plt.show()

# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


