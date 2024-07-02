#Load libraries
from morphax import data as md
from morphax import dmnn as dmnn
import jax.numpy as jnp
import numpy as np
import pandas as pd
import time
import jax
import os
import scipy
import math

#Read Game of Life
x_files_path = ['/scratch/user/dmarcondes/morphax/data/x_' + str(i) + '.csv' for i in range(1000)]
y_files_path = ['/scratch/user/dmarcondes/morphax/data/y_' + str(i) + '.csv' for i in range(1000)]

#Read train and test images
x = md.read_data_frame(x_files_path[0]).reshape((1,32,32)).astype(jnp.int32)
y = md.read_data_frame(y_files_path[0]).reshape((1,32,32)).astype(jnp.int32)
for i in range(9):
    x = jnp.append(x,md.read_data_frame(x_files_path[i+1]).reshape((1,32,32)).astype(jnp.int32),0)
    y = jnp.append(y,md.read_data_frame(y_files_path[i+1]).reshape((1,32,32)).astype(jnp.int32),0)

xval = md.read_data_frame(x_files_path[10]).reshape((1,32,32)).astype(jnp.int32)
yval = md.read_data_frame(y_files_path[10]).reshape((1,32,32)).astype(jnp.int32)
for i in range(10,100):
    xval = jnp.append(xval,md.read_data_frame(x_files_path[i+1]).reshape((1,32,32)).astype(jnp.int32),0)
    yval = jnp.append(yval,md.read_data_frame(y_files_path[i+1]).reshape((1,32,32)).astype(jnp.int32),0)

#Architectures
net = list(range(39))
net[0] = dmnn.cdmnn(['supgen','sup'],[4,1],[3,1],shape_x = (32,32),sample = True,p1 = 0.5)
net[1] = dmnn.cdmnn(['supgen','sup'],[8,1],[3,1],shape_x = (32,32),sample = True,p1 = 0.5)
net[2] = dmnn.cdmnn(['supgen','sup'],[16,1],[3,1],shape_x = (32,32),sample = True,p1 = 0.5)
net[3] = dmnn.cdmnn(['supgen','sup'],[32,1],[3,1],shape_x = (32,32),sample = True,p1 = 0.5)
net[4] = dmnn.cdmnn(['supgen','sup'],[64,1],[3,1],shape_x = (32,32),sample = True,p1 = 0.5)
net[5] = dmnn.cdmnn(['supgen','sup'],[128,1],[3,1],shape_x = (32,32),sample = True,p1 = 0.5)
net[6] = dmnn.cdmnn(['supgen','sup'],[256,1],[3,1],shape_x = (32,32),sample = True,p1 = 0.5)
net[7] = dmnn.cdmnn(['supgen','sup'],[512,1],[3,1],shape_x = (32,32),sample = True,p1 = 0.5)
net[8] = dmnn.cdmnn(['supgen','sup'],[1024,1],[3,1],shape_x = (32,32),sample = True,p1 = 0.5)

#net[9] = dmnn.cdmnn(2 * ['supgen','sup'],2 * [2,1],2 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[10] = dmnn.cdmnn(2 * ['supgen','sup'],2 * [4,1],2 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[11] = dmnn.cdmnn(2 * ['supgen','sup'],2 * [8,1],2 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[12] = dmnn.cdmnn(2 * ['supgen','sup'],2 * [16,1],2 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[13] = dmnn.cdmnn(2 * ['supgen','sup'],2 * [32,1],2 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[14] = dmnn.cdmnn(2 * ['supgen','sup'],2 * [64,1],2 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[15] = dmnn.cdmnn(2 * ['supgen','sup'],2 * [128,1],2 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[16] = dmnn.cdmnn(2 * ['supgen','sup'],2 * [256,1],2 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[17] = dmnn.cdmnn(2 * ['supgen','sup'],2 * [512,1],2 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)

#net[18] = dmnn.cdmnn(4 * ['supgen','sup'],4 * [2,1],4 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[19] = dmnn.cdmnn(4 * ['supgen','sup'],4 * [4,1],4 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[20] = dmnn.cdmnn(4 * ['supgen','sup'],4 * [8,1],4 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[21] = dmnn.cdmnn(4 * ['supgen','sup'],4 * [16,1],4 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[22] = dmnn.cdmnn(4 * ['supgen','sup'],4 * [32,1],4 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[23] = dmnn.cdmnn(4 * ['supgen','sup'],4 * [64,1],4 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[24] = dmnn.cdmnn(4 * ['supgen','sup'],4 * [128,1],4 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[25] = dmnn.cdmnn(4 * ['supgen','sup'],4 * [256,1],4 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)

#net[26] = dmnn.cdmnn(8 * ['supgen','sup'],8 * [2,1],8 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[27] = dmnn.cdmnn(8 * ['supgen','sup'],8 * [4,1],8 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[28] = dmnn.cdmnn(8 * ['supgen','sup'],8 * [8,1],8 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[29] = dmnn.cdmnn(8 * ['supgen','sup'],8 * [16,1],8 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[30] = dmnn.cdmnn(8 * ['supgen','sup'],8 * [32,1],8 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[31] = dmnn.cdmnn(8 * ['supgen','sup'],8 * [64,1],8 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[32] = dmnn.cdmnn(8 * ['supgen','sup'],8 * [128,1],8 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)

#net[33] = dmnn.cdmnn(16 * ['supgen','sup'],16 * [2,1],16 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[34] = dmnn.cdmnn(16 * ['supgen','sup'],16 * [4,1],16 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[35] = dmnn.cdmnn(16 * ['supgen','sup'],16 * [8,1],16 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[36] = dmnn.cdmnn(16 * ['supgen','sup'],16 * [16,1],16 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[37] = dmnn.cdmnn(16 * ['supgen','sup'],16 * [32,1],16 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)
#net[38] = dmnn.cdmnn(16 * ['supgen','sup'],16 * [64,1],16 * [3,1],shape_x = (32,32),sample = True,p1 = 0.5)

#Training each one
results = list(range(len(net)))
for i in [5]:#range(len(net)):
    print(i)
    results[i] = dmnn.train_dmnn(x,y,net[i],dmnn.MSE,xval = xval,yval = yval,sample = True,neighbors = 8,epochs = 25000,batches = 2,notebook = True,epoch_print= 100,epoch_store = 10)
    tmp_table = pd.DataFrame(np.array([results[i]['trace_epoch'],results[i]['trace_time'],results[i]['trace_loss'],results[i]['trace_val_loss']]).transpose(),columns = ['epoch','time','train_loss','val_loss'])
    tmp_table.to_csv('dmnn_gol_2bacthes' + str(i) + '.csv')
    jnp.save("params_5.npy",results[i]['best_par'])
    del tmp_table
