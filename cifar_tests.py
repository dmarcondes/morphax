#Test on CIFAR-10
from jinnax import data as jd
from jinnax import arch as jar
from jinnax import morph as jmp
from jinnax import training as jtr
import jax.numpy as jnp
import numpy as np
import time
import jax
import pickle
import scipy

#Read data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = None
for i in range(5):
    a = unpickle('data/cifar-10-python/data_batch_' + str(i+1))
    image_array = a[b'data']
    nimages = image_array.shape[0]
    image_R = np.reshape(image_array[:,0:1024], (nimages,32,32,1))
    image_G = np.reshape(image_array[:,1024:2048], (nimages,32,32,1))
    image_B = np.reshape(image_array[:,2048:4096], (nimages,32,32,1))
    image = np.append(image_R,image_G,axis = 3)
    image = np.append(image,image_B,axis = 3)
    if data is None:
        data = image
    else:
        data = np.append(data,image,axis = 0)

data.shape
data = jnp.array(data[:,:,:,0],dtype = jnp.float32)/255
jd.save_images(data,['cifar_test.png'])

#Define architecture
type = ['asf','asf']
width = [1,1]
size = [3,3]
shape_x = data.shape[1:3]
index_x = jmp.index_array(shape_x)
width_str = 5 * [40]
net = jar.cmnn_iter(type,width,width_str,size,shape_x)
forward = net['forward']
params = net['params']
t0 = time.time()
forward(data,params)
time.time() - t0

####Test training function####
x = data
loss = jtr.MSE
lr = 1e-3
p = jnp.zeros((1,1,3,3)) + 0.05
pcor = list()
pcor.append(p)
p = jnp.zeros((1,1,3,3)) + 0.1
pcor.append(p)
y = jmp.asf(x,index_x,pcor[0][0,0,:,:])
y = jmp.asf(y,index_x,pcor[0][1,0,:,:])
jd.save_images(y,['y_test.png'])
par = jtr.train_morph(x,y,forward,params,loss,lr = lr,epochs = 5000,batches = 1)
par
fx = forward(x,par)
loss(fx,y)/jnp.mean(fx ** 2)
jnp.mean(jnp.abs(255*(fx - y)))
jd.save_images(y,['NN_test.png'])
255*fx[0,:,:]
255*x[0,:,:]
batches = 2

y = jnp.minimum(jnp.maximum(0.0,x + 0.1*jax.random.normal(key = jax.random.PRNGKey(34),shape = x.shape)),1.0)
lr = 0.001
b1 = 0.9
b2 = 0.999
eps = 1e-08
eps_root = 0.0
key = 0

scipy.ndimage.morphology.grey_erosion(input = np.array(255*x.shape[0,:,:],dtype = np.int8),structure = np.array(jnp.round(255*pcor[0][0,0,:,:]),dtype = np.int8))
