from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras import VGG16
from tensorflow.keras import ImageDataGenerator

train_dir = "C:\Jai's work\PycharmProjects\SVM (classify cats & dogs)\train\train"
test_dir = "C:\Jai's work\PycharmProjects\SVM (classify cats & dogs)\test1\test1"

datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_generator = datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode='binary'
)

validation_generator = datagen.flow_from_directory(
    test_dir, target_size=(150, 150), batch_size=32, class_mode='binary'
)

model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
model.trainable = False

train_features = model.predict(train_generator)
test_features = model.predict(validation_generator)

train_features = train_features.reshape(-1, train_features.shape[1] * train_features.shape[2] * train_features.shape[3])
test_features = test_features.reshape(-1, test_features.shape[1] * test_features.shape[2] * test_features.shape[3])

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

train_classes = train_generator.classes
svm = SVC(kernel='linear')
svm.fit(train_features_scaled, train_classes)

print("Model trained and ready for predictions!")
