import cv2
import tensorflow as tf
import numpy as np

def conv_block(x, filters, kernel_size, padding='valid'):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
    return x

def Build_Model():
    inputs = tf.keras.Input(shape=(250, 250, 3))
    x = tf.keras.layers.Rescaling(1./255)(inputs) 

    x = conv_block(x, filters=16, kernel_size=3, padding='same')
    x = conv_block(x, filters=32, kernel_size=3)
    x = conv_block(x, filters=64, kernel_size=3)
    x = conv_block(x, filters=64, kernel_size=3)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    x = tf.keras.layers.Dense(units=8, activation='relu')(x)
    x = tf.keras.layers.Dense(units=1, activation='linear')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

model = Build_Model()
model.load_weights('face_classifier.weights.h5')

# OpenCV built in detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
window_name = 'AI Face Tracker'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(60, 60))

    for (x, y, w, h) in faces:
        offset = int(w * 0.4) 
        y1, y2 = max(0, y-offset), min(frame.shape[0], y+h+offset)
        x1, x2 = max(0, x-offset), min(frame.shape[1], x+w+offset)
        # Crop and Preprocess
        face_roi = frame[y1:y2, x1:x2]
        img = cv2.resize(face_roi, (250, 250))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        input_arr = np.expand_dims(img, axis=0)

        predictions = model.predict(input_arr, verbose=0)
        probability = tf.nn.sigmoid(predictions).numpy()[0][0]

        if probability < 0.6: 
            label = "CONFIRMED FACE"
            color = (0, 255, 0) # Green
            thickness = 2
        else:
            label = "SCANNING..."
            color = (0, 255, 255) # Yellow
            thickness = 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        cv2.putText(frame, f"{label} {100*(1-probability):.1f}%", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()