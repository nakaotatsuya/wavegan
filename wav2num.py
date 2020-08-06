from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import os

#
wav_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/drums/test/Ride_00156.wav")

sound = AudioSegment.from_file(wav_path, "wav")

wav_path2 = os.path.join(os.path.abspath(os.path.dirname(__file__)), "output_data/exapmle_0.wav")

sound2 = AudioSegment.from_file(wav_path2, "wav")

data = np.array(sound.get_array_of_samples())

data2 = np.array(sound2.get_array_of_samples())
print(data.shape)

x = data[::sound.channels]

x2 = data2[::sound2.channels]
print(x.shape)

plt.plot(x[::10])
#plt.plot(x2[::10])
plt.grid()
plt.show()
