import tensorflow as tf
#from IPython.display import display, Audio
import os
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write

# Load the graph
ngenerate = 100
ndisplay = 4

tf.reset_default_graph()
saver = tf.train.import_meta_graph(os.path.join(os.path.abspath(os.path.dirname(__file__)),"train","infer",'infer.meta'))
graph = tf.get_default_graph()
sess = tf.InteractiveSession()

#with tf.Session() as sess:
saver.restore(sess, os.path.join(os.path.abspath(os.path.dirname(__file__)),"train", 'model.ckpt-6505'))
# Create 50 random latent vectors z

_z = (np.random.rand(ngenerate, 100) * 2.) -1
print(_z.shape)

# Synthesize G(z)
z = graph.get_tensor_by_name('z:0')
G_z = graph.get_tensor_by_name('G_z:0')
#G_z_spec = graph.get_tensor_by_name("G_z_spec:0")

_G_z  = sess.run(G_z , {z: _z})
#_G_z, _G_z_spec = sess.run([G_z,G_z_spec] , {z: _z})
#sess.close()

print(_G_z.shape)
# outfile = "test.wav"
# save_wav(_G_z, outfile)
# Play audio in notebook
    
#display(Audio(_G_z[0, :, 0], rate=16000))

for i in range(ngenerate):
    #print("aaaa")
    write("./output_data/exapmle_{}.wav".format(i), 16000, _G_z[i,:,0])


_w = _z[0] + _z[1]
_w = np.reshape(_w, (1, 100))

w = graph.get_tensor_by_name("z:0")
G_w = graph.get_tensor_by_name("G_z:0")

_G_w = sess.run(G_w, {w: _w})

print(_G_w)

sess.close()

write("./output_data/example_0and1.wav", 16000, _G_w[0,:,0])
