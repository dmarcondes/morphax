python3 setup.py sdist bdist_wheel
pdoc morphax/* -o docs/
pip3 uninstall morphax;
pip3 install --break-system-packages git+ssh://git@github.com/dmarcondes/morphax
from morphax import data as md
from morphax import mnn as mnn
from morphax import morph as mp
from morphax import dmorph as dmnn
from morphax import dmorph as dmp
import jax.numpy as jnp
import numpy as np
import time
import jax
import os
import scipy
import math
import itertools

#####Stack filter######
x = (1-md.image_to_jnp(['input_reduced.png']))*255
#x = x + x*0.1*jax.random.normal(jax.random.PRNGKey(345),shape = x.shape)
y = (1-md.image_to_jnp(['output_reduced.png']))*255

type = ['supgen','sup']
width = [8,1]
size = [3,1]
net = dmnn.cdmnn(type,width,size,shape_x = (128,128),sample = False,p1 = 0.1)
res = dmnn.train_dmnn_stack_slda(x,y,net,dmnn.MSE,sample = True,neighbors = 32,epochs = 1000,epoch_print = 1,epoch_store = 1000)

pred = res['oper'](x)
md.save_images(pred/255,['pred.png'])
