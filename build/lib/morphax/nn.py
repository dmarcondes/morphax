#Functions to train NN
import jax
import jax.numpy as jnp
import optax
from alive_progress import alive_bar
import math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from morphax import morph as mp
import sys

__docformat__ = "numpy"

#MSE
@jax.jit
def MSE(true,pred):
  return jnp.mean((true - pred)**2)

#MSE self-adaptative
@jax.jit
def MSE_SA(true,pred,wheight):
  return jnp.mean(jax.nn.sigmoid(wheight) * (true - pred)**2)

#L2 error
@jax.jit
def L2error(pred,true):
  return jnp.sqrt(jnp.sum((true - pred)**2))/jnp.sqrt(jnp.sum(true ** 2))

#Croos entropy
@jax.jit
def CE(true,pred):
  return jnp.mean((- true * jnp.log(pred + 1e-5) - (1 - true) * jnp.log(1 - pred + 1e-5)))

#Croos entropy self-adaptative
@jax.jit
def CE_SA(true,pred,wheight):
  return jnp.mean(jax.nn.sigmoid(wheight) * (- true * jnp.log(pred + 1e-5) - (1 - true) * jnp.log(1 - pred + 1e-5)))

#IoU
@jax.jit
def IoU(true,pred):
  return 1 - (jnp.sum(2 * true * pred) + 1)/(jnp.sum(true + pred) + 1)

#IoU self-adaptative
@jax.jit
def IoU_SA(true,pred,wheight):
  return 1 - (jnp.sum(jax.nn.sigmoid(wheight) *2 * true * pred) + 1)/(jnp.sum(jax.nn.sigmoid(wheight) * (true + pred + 1)))

#Simple fully connected architecture. Return the function for the forward pass
def fconNN(width,activation = jax.nn.tanh,key = 0):
    #Initialize parameters with Glorot initialization
    initializer = jax.nn.initializers.glorot_normal()
    key = jax.random.split(jax.random.PRNGKey(key),len(width)-1) #Seed for initialization
    params = list()
    for key,lin,lout in zip(key,width[:-1],width[1:]):
        W = initializer(key,(lin,lout),jnp.float32)
        B = initializer(key,(1,lout),jnp.float32)
        params.append({'W':W,'B':B})

    #Define function for forward pass
    @jax.jit
    def forward(x,params):
      *hidden,output = params
      for layer in hidden:
        x = activation(x @ layer['W'] + layer['B'])
      return x @ output['W'] + output['B']

    #Return initial parameters and forward function
    return {'params': params,'forward': forward}

#Fully connected architecture for structural element
def fconNN_str(width,activation = jax.nn.tanh,key = 0):
    #Add first and last layer
    width = [2] + width + [1]

    #Initialize parameters with Glorot initialization
    initializer = jax.nn.initializers.glorot_normal()
    key = jax.random.split(jax.random.PRNGKey(key),len(width)-1) #Seed for initialization
    params = list()
    for key,lin,lout in zip(key,width[:-1],width[1:]):
        W = initializer(key,(lin,lout),jnp.float32)
        B = initializer(key,(1,lout),jnp.float32)
        params.append({'W':W,'B':B})

    #Define function for forward pass
    @jax.jit
    def forward(x,params):
      *hidden,output = params
      for layer in hidden:
        x = activation(x @ layer['W'] + layer['B'])
      return 2*jax.nn.tanh(x @ output['W'] + output['B'])

    #Return initial parameters and forward function
    return {'params': params,'forward': forward}

#Apply a morphological layer
def apply_morph_layer(x,type,params,index_x):
    #Apply each operator
    params = 2 * jax.nn.tanh(params)
    oper = mp.operator(type)
    fx = None
    for i in range(params.shape[0]):
        if fx is None:
            fx = oper(x,index_x,params[i,:,:,:]).reshape((1,x.shape[0],x.shape[1],x.shape[2]))
        else:
            fx = jnp.append(fx,oper(x,index_x,params[i,:,:,:]).reshape((1,x.shape[0],x.shape[1],x.shape[2])),0)
    return fx

#Apply a morphological layer in iterated NN
def apply_morph_layer_iter(x,type,params,index_x,w,forward_inner,d):
    #Compute structural elements
    k = None
    if type == 'supgen' or type == 'infgen':
        for i in range(int(len(params)/2)):
            tmp = forward_inner(w,params[2*i]).reshape((1,d,d))
            tmp = jnp.append(tmp,forward_inner(w,params[2*i + 1]).reshape((1,d,d)),0).reshape((1,2,d,d))
            if k is None:
                k = tmp
            else:
                k = jnp.append(k,tmp,0)
    else:
        for i in range(len(params)):
            tmp = forward_inner(w,params[i]).reshape((1,1,d,d))
            if k is None:
                k = tmp
            else:
                k = jnp.append(k,tmp,0)
    params = k

    #Apply each operator
    oper = mp.operator(type)
    fx = None
    for i in range(params.shape[0]):
        if fx is None:
            fx = oper(x,index_x,params[i,:,:,:]).reshape((1,x.shape[0],x.shape[1],x.shape[2]))
        else:
            fx = jnp.append(fx,oper(x,index_x,params[i,:,:,:]).reshape((1,x.shape[0],x.shape[1],x.shape[2])),0)
    return fx

#Canonical Morphological NN
def cmnn(x,type,width,size,shape_x,mask = 'inf',key = 0):
    key = jax.random.split(jax.random.PRNGKey(key),(len(width),max(width)))[:,:,0]

    #Index window
    index_x = mp.index_array(shape_x)

    #Initialize parameters
    params = list()
    for i in range(len(width)):
        if type[i] in ['sup','inf','complement']:
            params.append(jnp.array(0.0).reshape((1,1,1)))
        else:
            if type[i] == 'supgen' or type[i] == 'infgen':
                ll = jnp.arctanh(jnp.maximum(jnp.minimum(mp.struct_lower(x,size[i])/2,1-1e-5),-1 + 1e-5)).reshape((1,1,size[i],size[i]))
                ul = jnp.arctanh(jnp.maximum(jnp.minimum(mp.struct_upper(x,size[i])/2,1-1e-5),-1 + 1e-5)).reshape((1,1,size[i],size[i]))
                sl = jnp.std(ll)
                su = jnp.std(ul)
                p = jnp.append(ll + sl*jax.random.normal(jax.random.PRNGKey(key[i,-1]),ll.shape),ul + su*jax.random.normal(jax.random.PRNGKey(key[i,-1]),ul.shape),1)
                for j in range(width[i] - 1):
                    interval = jnp.append(ll + sl*jax.random.normal(jax.random.PRNGKey(key[i,j]),ll.shape),ul + su*jax.random.normal(jax.random.PRNGKey(key[i,j]),ul.shape),1)
                    p = jnp.append(p,interval,0)
            else:
                ll = jnp.arctanh(jnp.maximum(jnp.minimum(mp.struct_lower(x,size[i])/2,1-1e-5),-1 + 1e-5)).reshape((1,1,size[i],size[i]))
                sl = jnp.std(ll)
                p = ll + sl*jax.random.normal(jax.random.PRNGKey(key[i,-1]),ll.shape)
                for j in range(width[i] - 1):
                    interval = ll + sl*jax.random.normal(jax.random.PRNGKey(key[i,j]),ll.shape)
                    p = jnp.append(p,interval,0)
            params.append(p)

    #Initialize mask
    mask_list = list()
    for i in range(len(width)):
        if type[i] in ['sup','inf','complement']:
            mask_list.append(jnp.array(0.0))
        else:
            if type[i] == 'supgen' or type[i] == 'infgen':
                if mask == 'inf':
                    ll = jnp.array(math.floor((size[i] ** 2)/2) * [0] + [1] + math.floor((size[i] ** 2)/2) * [0]).reshape((1,1,size[i],size[i]))
                    ul = jnp.array(math.floor((size[i] ** 2)/2) * [0] + [1] + math.floor((size[i] ** 2)/2) * [0]).reshape((1,1,size[i],size[i]))
                else:
                    ll = jnp.array((size[i] ** 2) * [1]).reshape((1,1,size[i],size[i]))
                    ul = jnp.array((size[i] ** 2) * [1]).reshape((1,1,size[i],size[i]))
                m = jnp.append(ll,ul,1)
                for j in range(width[i] - 1):
                    interval = jnp.append(ll,ul,1)
                    m = jnp.append(m,interval,0)
            else:
                if mask == 'inf':
                    ll = jnp.array(math.floor((size[i] ** 2)/2) * [0] + [1] + math.floor((size[i] ** 2)/2) * [0]).reshape((1,1,size[i],size[i]))
                else:
                    ll = jnp.array((size[i] ** 2) * [1]).reshape((1,1,size[i],size[i]))
                m = ll
                for j in range(width[i] - 1):
                    m = jnp.append(m,interval,0)
            mask_list.append(m)

    #Forward pass
    @jax.jit
    def forward(x,params):
        x = x.reshape((1,x.shape[0],x.shape[1],x.shape[2]))
        for i in range(len(type)):
            #Apply sup and inf
            if type[i] == 'sup':
                x = mp.vmap_sup(x)
            elif type[i] == 'inf':
                x = mp.vmap_inf(x)
            elif type[i] == 'complement':
                x = 1 - x
            else:
                #Apply other layer
                x = apply_morph_layer(x[0,:,:,:],type[i],params[i],index_x)
        return x[0,:,:,:]

    #Return initial parameters and forward function
    return {'params': params,'forward': forward,'mask': mask_list}

#Canonical Morphological NN with iterated NN
def cmnn_iter(type,width,width_str,size,shape_x,x = None,activation = jax.nn.tanh,key = 0,init = 'identity',loss = MSE_SA,sa = True,epochs = 1000,batches = 1,lr = 0.001,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0,notebook = False):
    #Index window
    index_x = mp.index_array(shape_x)

    #Create w to apply str NN
    unique_size = set(size)
    w = {}
    for d in unique_size:
        w[str(d)] = jnp.array([[x1.tolist(),x2.tolist()] for x1 in jnp.linspace(-jnp.floor(d/2),jnp.floor(d/2),d) for x2 in jnp.linspace(jnp.floor(d/2),-jnp.floor(d/2),d)])

    #Initialize parameters
    ll = None
    ul = None
    if init == 'identity':
        #Train inner NN to generate zero and one kernel
        max_size = max(size)
        w_max = w[str(max_size)]
        l = math.floor(max_size/2)

        #Lower limit
        nn = fconNN_str(width_str,activation,key)
        forward_inner = nn['forward']
        w_y = mp.struct_lower(x,max_size).reshape((w_max.shape[0],1))
        params_ll = jtr.train_fcnn(w_max,w_y,forward_inner,nn['params'],loss,sa,epochs,batches,lr,b1,b2,eps,eps_root,key,notebook)
        ll = forward_inner(w_max,params_ll).reshape((max_size,max_size))

        #infgen does not work, fix later
        if 'supgen' in type or 'infgen' in type:
            #Upper limit
            nn = fconNN_str(width_str,activation,key)
            forward_inner = nn['forward']
            w_y = mp.struct_upper(x,max_size).reshape((w_max.shape[0],1))
            params_ul = jtr.train_fcnn(w_max,w_y,forward_inner,nn['params'],loss,sa,epochs,batches,lr,b1,b2,eps,eps_root,key,notebook)
            ul = forward_inner(w_max,params_ul).reshape((max_size,max_size))

        #Assign trained parameters
        params = list()
        for i in range(len(width)):
            params.append(list())
            for j in range(width[i]):
                if type[i] ==  'sup' or type[i] ==  'inf' or type[i] ==  'complement':
                    params[i].append(jnp.array(0.0,dtype = jnp.float32))
                else:
                    params[i].append(params_ll)
                    if type[i] == 'supgen' or type[i] == 'infgen':
                        params[i].append(params_ul)
    elif init == 'random':
        initializer = jax.nn.initializers.normal()
        k = jax.random.split(jax.random.PRNGKey(key),(len(width)*max(width))) #Seed for initialization
        c = 0
        forward_inner = fconNN_str(width_str,activation = jax.nn.tanh,key = 0)['forward']
        params = list()
        for i in range(len(width)):
            params.append(list())
            for j in range(width[i]):
                if type[i] ==  'sup' or type[i] ==  'inf' or type[i] ==  'complement':
                    params[i].append(jnp.array(0.0,dtype = jnp.float32))
                else:
                    tmp = fconNN_str(width_str,activation = jax.nn.tanh,key = k[c,0])
                    params[i].append(tmp['params'])
                    if type[i] == 'supgen' or type[i] == 'infgen':
                        tmp2 = fconNN_str(width_str,activation = jax.nn.tanh,key = k[c,1])
                        params[i].append(tmp2['params'])
                    c = c + 1

    #Forward pass
    @jax.jit
    def forward(x,params):
        x = x.reshape((1,x.shape[0],x.shape[1],x.shape[2]))
        for i in range(len(type)):
            #Apply sup and inf
            if type[i] == 'sup':
                x = mp.vmap_sup(x)
            elif type[i] == 'inf':
                x = mp.vmap_inf(x)
            elif type[i] == 'complement':
                x = 1 - x
            else:
                #Apply other layer
                x = apply_morph_layer_iter(x[0,:,:,:],type[i],params[i],index_x,w[str(size[i])],forward_inner,size[i])
        return x[0,:,:,:]

    #Compute structuring elements
    @jax.jit
    def compute_struct(params):
        #Compute for each layer
        struct = list()
        for i in range(len(width)):
            struct.append(list())
            if type[i] ==  'sup' or type[i] ==  'inf' or type[i] ==  'complement':
                struct[i].append(jnp.array(0.0,dtype = jnp.float32))
            elif type[i] in ['infgen','supgen']:
                for j in range(width[i]):
                    k0 = forward_inner(w[str(size[i])],params[i][2*j]).reshape((1,size[i],size[i]))
                    k1 = forward_inner(w[str(size[i])],params[i][2*j + 1]).reshape((1,size[i],size[i]))
                    struct[i].append(jnp.append(k0,k1,0))
            else:
                for j in range(width[i]):
                    struct[i].append(forward_inner(w[str(size[i])],params[i][j]).reshape((size[i],size[i])))
        return struct

    #Return initial parameters and forward function
    return {'params': params,'forward': forward,'ll': ll,'ul': ul,'compute_struct': compute_struct}

#Training function MNN
def train_morph(x,y,forward,params,loss,mask = None,sa = False,epochs = 1,batches = 1,lr = 0.001,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0,key = 0,notebook = False,epoch_print = 100):
    #Key
    key = jax.random.split(jax.random.PRNGKey(key),epochs)

    #Batch size
    bsize = int(math.floor(x.shape[0]/batches))

    #Self-adaptative
    if sa:
        params.append({'w': jnp.zeros((y.shape)) + 1.0})
        @jax.jit
        def lf(params,x,y):
            return jnp.mean(jax.vmap(loss,in_axes = (0,0,0))(forward(x,params[:-1]),y,params[-1]['w']))
    else:
        #Loss function
        @jax.jit
        def lf(params,x,y):
            return jnp.mean(jax.vmap(loss,in_axes = (0,0))(forward(x,params),y))

    #Optmizer NN
    optimizer = optax.adam(lr,b1,b2,eps,eps_root)
    opt_state = optimizer.init(params)

    #Training function
    grad_loss = jax.jit(jax.grad(lf,0))
    @jax.jit
    def update(opt_state,params,x,y,mask):
      grads = grad_loss(params,x,y)
      if sa:
          grads[-1]['w'] = - grads[-1]['w']
          if mask is not None:
              for i in range(len(grads) - 1):
                  grads[i] = grads[i] * mask[i]
      if mask is not None:
          for i in range(len(grads)):
              grads[i] = grads[i] * mask[i]
      updates, opt_state = optimizer.update(grads, opt_state)
      params = optax.apply_updates(params, updates)
      return opt_state,params

    #Train
    t0 = time.time()
    with alive_bar(epochs) as bar:
        for e in range(epochs):
            if not sa:
                #Permutate x
                x = jax.random.permutation(jax.random.PRNGKey(key[e,0]),x,0)
                for b in range(batches):
                    if b < batches - 1:
                        xb = jax.lax.dynamic_slice(x,(b*bsize,0,0),(bsize,x.shape[1],x.shape[2]))
                        yb = jax.lax.dynamic_slice(x,(b*bsize,0,0),(bsize,x.shape[1],x.shape[2]))
                    else:
                        xb = x[b*bsize:x.shape[0],:,:]
                        yb = y[b*bsize:y.shape[0],:,:]
                    opt_state,params = update(opt_state,params,xb,yb,mask)
            else:
                opt_state,params = update(opt_state,params,x,y,mask)
            if e % epoch_print == 0 and notebook:
                l = str(jnp.round(lf(params,x,y),10))
                if notebook:
                    print('Epoch: ' + str(e) + ' Time: ' + str(jnp.round(time.time() - t0,2)) + ' s Loss: ' + l)
                if not notebook:
                    bar.title("Loss: " + l)
            bar()

    return params


#Training function FCNN
def train_fcnn(x,y,forward,params,loss,sa = False,epochs = 1,batches = 1,lr = 0.001,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0,key = 0,notebook = False,epoch_print = 1000):
    #Key
    key = jax.random.split(jax.random.PRNGKey(key),epochs)

    #Batch size
    bsize = int(math.floor(x.shape[0]/batches))

    #Self-adaptative
    if sa:
        params.append({'w': jnp.zeros((y.shape)) + 1.0})
        @jax.jit
        def lf(params,x,y):
            return jnp.mean(jax.vmap(loss,in_axes = (0,0,0))(forward(x,params[:-1]),y,params[-1]['w']))
    else:
        #Loss function
        @jax.jit
        def lf(params,x,y):
            return jnp.mean(jax.vmap(loss,in_axes = (0,0))(forward(x,params),y))

    #Optmizer NN
    optimizer = optax.adam(lr,b1,b2,eps,eps_root)
    opt_state = optimizer.init(params)

    #Training function
    grad_loss = jax.jit(jax.grad(lf,0))
    @jax.jit
    def update(opt_state,params,x,y):
      grads = grad_loss(params,x,y)
      if sa:
          grads[-1]['w'] = - grads[-1]['w']
      updates, opt_state = optimizer.update(grads, opt_state)
      params = optax.apply_updates(params, updates)
      return opt_state,params

    #Train
    t0 = time.time()
    with alive_bar(epochs) as bar:
        for e in range(epochs):
            #Permutate x
            if not sa:
                x = jax.random.permutation(jax.random.PRNGKey(key[e,0]),x,0)
                for b in range(batches):
                    if b < batches - 1:
                        xb = jax.lax.dynamic_slice(x,(b*bsize,0),(bsize,x.shape[1]))
                        yb = jax.lax.dynamic_slice(x,(b*bsize,0),(bsize,x.shape[1]))
                    else:
                        xb = x[b*bsize:x.shape[0],:]
                        yb = y[b*bsize:y.shape[0],:]
                    opt_state,params = update(opt_state,params,xb,yb)
            else:
                opt_state,params = update(opt_state,params,x,y)
            l = str(jnp.round(lf(params,x,y),10))
            if(e % epoch_print == 0 and notebook):
                print('Epoch: ' + str(e) + ' Time: ' + str(jnp.round(time.time() - t0,2)) + ' s Loss: ' + l)
            if not notebook:
                bar.title("Loss: " + l)
                bar()

    del params[-1]
    return params

#SLDA for training DMNN
def slda(x,y,x_val,y_val,forward,params,loss,epochs_nn,epochs_slda,sample_neigh,mask = None,sa = False,batches = 1,lr = 0.001,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0,key = 0,notebook = False,epoch_print = 100):
    #Find out width,size, type and calculate probabilities
    width = []
    size = []
    prob = []
    type = []
    for i in range(len(params)):
        width = width + [params[i].shape[0]]
        size = size + [params[i].shape[2]]
        if params[i].shape[2] > 1:
            prob  = prob + [params[i].shape[0] * (params[i].shape[2] ** 2)]
        else:
            prob  = prob + [0]
        if params[i].shape[1] == 2:
            type = type + ['gen']
            prob[-1] = 2*prob[-1]
        else:
            type = type + ['other']

    prob = [x/sum(prob) for x in prob]

    #Current error
    current_error = loss(forward(x_val,params),y_val).tolist()

    #Epochs of SLDA
    for e in range(epochs_slda):
        print("Epoch SLDA " + str(e) + ' Current validation error: ' + str(jnp.round(current_error,6)))
        min_error = jnp.inf
        #Sample neighbors
        for n in range(sample_neigh):
            print("Sample neighbor " + str(n) + ' Min local validation error :' + str(jnp.round(min_error,6)))
            #Sample layer
            layer = np.random.choice(list(range(len(width))),1,p = prob)[0]
            #Sample node
            node = np.random.choice(list(range(width[layer])),1)[0]
            #Position
            pos = np.random.choice(list(range(size[layer] ** 2)),1)[0]
            pos = [math.floor(pos/size[layer]),pos - (math.floor(pos/size[layer]))*size[layer]]
            #Limit of interval
            if type[layer] == 'gen':
                limit = np.random.choice([0,1],1)[0]
            else:
                limit = 0
            #New mask
            new_mask = mask
            new_mask[layer] = new_mask[layer].at[node,limit,pos[0],pos[1]].set(jnp.abs(1 - new_mask[layer][node,limit,pos[0],pos[1]]))

            #Train
            res_neigh = train_morph(x,y,forward,params,loss,new_mask,sa,epochs_nn,batches,lr,b1,b2,eps,eps_root,key,notebook,epoch_print)

            #Val error
            error_neigh = loss(forward(x_val,res_neigh),y_val).tolist()

            #Store is best
            if error_neigh < min_error:
                min_error = error_neigh
                min_mask = new_mask
                min_params = res_neigh

        #Update
        mask = min_mask
        params = min_params
        current_error = min_error

    return {'params': params,'mask': mask}