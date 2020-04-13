# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf
import cv2
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write


'''#Recording sound
fs = 44100  # Sample rate
seconds = 4  # Duration of recording

print('Recording')
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording)  # Save as WAV file 
print('Recording is finished')
'''
#Spectrum Generation
rate, data =scipy.io.wavfile.read('output.wav')
plt.plot(data)
plt.savefig('test.jpg')
plt.clf()

#loading model
new_model = tf.keras.models.load_model('GunshotDetector(model)')

new_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing import image

test_image = image.load_img('test.jpg', target_size = (64, 64)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

#predict the result
rslt = new_model.predict(test_image)

if rslt[0][0] == 1:
    print(rslt[0][0])
    prediction='others'
else:
    print(rslt[0][0])
    prediction='gunshot'

print(prediction)
