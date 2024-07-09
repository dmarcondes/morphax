#Functions to train (continuous) Morphological Neural Networks
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
from morphax import dmorph as dmp
import sys
import itertools

__docformat__ = "numpy"

#MSE
@jax.jit
def MSE(pred,true):
    """
    Mean square error
    ----------

    Parameters
    ----------
    pred : jax.numpy.array

        A JAX numpy array with the predicted values

    true : jax.numpy.array

        A JAX numpy array with the true values

    Returns
    -------
    mean square error
    """
    return jnp.mean((true - pred)**2)

#MSE self-adaptative
@jax.jit
def MSE_SA(pred,true,wheight,c = 100,q = 2):
    """
    Selft-adaptative mean square error
    ----------

    Parameters
    ----------
    pred : jax.numpy.array

        A JAX numpy array with the predicted values

    true : jax.numpy.array

        A JAX numpy array with the true values

    wheight : jax.numpy.array

        A JAX numpy array with the weights

    c,q : float

        Hyperparameters

    Returns
    -------
    self-adaptative mean square error
    """
    return jnp.mean(c * (wheight ** q) * (true - pred)**2)

#L2 error
@jax.jit
def L2error(pred,true):
    """
    L2 error
    ----------

    Parameters
    ----------
    pred : jax.numpy.array

        A JAX numpy array with the predicted values

    true : jax.numpy.array

        A JAX numpy array with the true values

    Returns
    -------
    L2 error
    """
    return jnp.sqrt(jnp.sum((true - pred)**2))/jnp.sqrt(jnp.sum(true ** 2))

#Croos entropy
@jax.jit
def CE(pred,true):
    """
    Cross entropy error
    ----------

    Parameters
    ----------
    pred : jax.numpy.array

        A JAX numpy array with the predicted values

    true : jax.numpy.array

        A JAX numpy array with the true values

    Returns
    -------
    cross-entropy error
    """
    return jnp.mean((- true * jnp.log(pred + 1e-6) - (1 - true) * jnp.log(1 - pred + 1e-6)))

#Croos entropy self-adaptative
@jax.jit
def CE_SA(pred,true,wheight,c = 100,q = 2):
    """
    Self-adaptative cross entropy error
    ----------

    Parameters
    ----------
    pred : jax.numpy.array

        A JAX numpy array with the predicted values

    true : jax.numpy.array

        A JAX numpy array with the true values

    wheight : jax.numpy.array

        A JAX numpy array with the weights

    c,q : float

        Hyperparameters

    Returns
    -------
    delf-adaptative cross-entropy error
    """
    return jnp.mean(c * (wheight ** q) * (- true * jnp.log(pred + 1e-5) - (1 - true) * jnp.log(1 - pred + 1e-5)))

#IoU
@jax.jit
def IoU(pred,true):
    """
    Intersection over union error
    ----------

    Parameters
    ----------
    pred : jax.numpy.array

        A JAX numpy array with the predicted values

    true : jax.numpy.array

        A JAX numpy array with the true values

    Returns
    -------
    intersection over union error
    """
    return 1 - (jnp.sum(2 * true * pred) + 1)/(jnp.sum(true + pred) + 1)

#IoU self-adaptative
@jax.jit
def IoU_SA(pred,true,wheight,c = 100,q = 2):
    """
    Selft-adaptative intersection over union error
    ----------

    Parameters
    ----------
    pred : jax.numpy.array

        A JAX numpy array with the predicted values

    true : jax.numpy.array

        A JAX numpy array with the true values

    wheight : jax.numpy.array

        A JAX numpy array with the weights

    c,q : float

        Hyperparameters

    Returns
    -------
    selft adaptative intersection over union error
    """
    return 1 - (jnp.sum(c * (wheight ** q) * 2 * true * pred) + 1)/(jnp.sum(c * (wheight ** q) * (true + pred + 1)))

#Activation
def activate(x,b = 5):
    """
    Activation function for structuring elements
    ----------

    Parameters
    ----------
    x : float

        Input

    b : float

        Approximating constant

    Returns
    -------
    outuput of activation function
    """
    return jax.nn.sigmoid(2 * b * x - b) - 1

#Simple fully connected architecture. Return params and the function for the forward pass
def fconNN(width,activation = jax.nn.tanh,key = 0):
    """
    Initialize a Fully Connected Neural Network.
    ----------

    Parameters
    ----------
    width : list of int

        List with the width of each layer

    activation : function

        Activation function

    key : int

        Key for sampling

    Returns
    -------
    dictionary with the initial parameters, forward function and width
    """
    #Initialize parameters with Glorot initialization
    initializer = jax.nn.initializers.glorot_normal()
    key = jax.random.split(jax.random.PRNGKey(key),len(width) - 1) #Seed for initialization
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
    return {'params': params,'forward': forward,'width': width}

#Stochastic gradient descent
def sgd(x,y,forward,params,loss,sa = False,c = 100,q = 2,epochs = 1,batches = 1,lr = 0.001,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0,key = 0,notebook = False,epoch_print = 1000):
    """
    Stochastic gradient descent algorithm
    ----------

    Parameters
    ----------
    x,y : jax.numpy.array

        Input and output data

    forward : function

        Forward function

    params : list

        Initial parameters

    loss : function

        loss function

    sa : logical

        Whether to consider self-adaptative weights

    c,q : float

        Hyperparameters of self-adaptative weights

    epochs : int

        Number of training epochs

    batches : int

        Number of batches

    lr,b1,b2,eps,eps_root: float

        Hyperparameters of the Adam algorithm. Default lr = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-08, eps_root = 0.0

    key : int

        Seed for parameters initialization. Default 0

    notebook : logical

        Whether code is being executed in a notebook

    epoch_print : int

        Number of epochs to print training error

    Returns
    -------
    list of parameters
    """
    #Key
    key = jax.random.split(jax.random.PRNGKey(key),epochs)

    #Batch size
    bsize = int(math.floor(x.shape[0]/batches))

    #Self-adaptative
    if sa:
        params.append({'w': jnp.zeros((y.shape)) + (1/c) ** (1/q)})
        @jax.jit
        def lf(params,x,y):
            return jnp.mean(jax.vmap(lambda true,pred,weight: loss(true,pred,weight,c,q),in_axes = (0,0,0))(forward(x,params[:-1]),y,params[-1]['w']))
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
                y = jax.random.permutation(jax.random.PRNGKey(key[e,0]),y,0)
                for b in range(batches):
                    if b < batches - 1:
                        xb = jax.lax.dynamic_slice_in_dim(x,b*bsize,bsize,axis = 0)
                        yb = jax.lax.dynamic_slice_in_dim(y,b*bsize,bsize,axis = 0)
                    else:
                        xb = jax.lax.dynamic_slice_in_dim(x,b*bsize,x.shape[0] - b*bsize,axis = 0)
                        yb = jax.lax.dynamic_slice_in_dim(y,b*bsize,y.shape[0] - b*bsize,axis = 0)
                    opt_state,params = update(opt_state,params,xb,yb)
            else:
                opt_state,params = update(opt_state,params,x,y)
            l = str(jnp.round(lf(params,x,y),10))
            if(e % epoch_print == 0 and notebook):
                print('Epoch: ' + str(e) + ' Time: ' + str(jnp.round(time.time() - t0,2)) + ' s Loss: ' + l)
            if not notebook:
                bar.title("Loss: " + l)
                bar()

    if sa:
        del params[-1]
    return params


#Fully connected architecture for w-operator characteristic function
def fconNN_wop(width,d,activation = jax.nn.tanh,key = 0,epochs = 1000):
    """
    Initialize a Fully Connected Neural Network to represent the characteristic function of a W-operator.
    ----------

    Parameters
    ----------
    width : list of int

        List with the width of each layer

    d : int

        Dimension of window

    activation : function

        Activation function

    key : int

        Key for sampling

    epochs : int

        Epochs to pretrain neural network

    Returns
    -------
    dictionary with the initial parameters, forward function and width
    """
    #Add first and last layer
    width = [d ** 2] + width + [1]

    #Forward and params
    net = fconNN(width,activation,key)
    forward = lambda x,params: jax.nn.sigmoid(net['forward'](x,params))
    params = net['params']

    #Train
    input = jnp.array(list(itertools.product([0, 1], repeat = d ** 2)))
    output = jnp.where(input[:,math.ceil((d ** 2)/2)] == 1,1,0).reshape((input.shape[0],1))
    params = sgd(input,output,forward,params,MSE_SA,sa = True,epochs = epochs,epoch_print = 100)

    #Return initial parameters and forward function
    return {'params': params,'forward': forward,'width': width}

#Apply a morphological layer
def apply_morph_layer(x,type,params,index_x,forward_wop = None,d = None):
    """
    Apply a morphological layer.
    ----------

    Parameters
    ----------
    x : jax.numpy.array

        A JAX array with images

    type : str

        Names of the operator to apply

    params : jax.numpy.array

        Parameters of the operator

    index_x : jax.numpy.array

        Array with the indexes of f

    forward_wop : function

        Forward function of the W-operator

    d : int

        Dimension of the window

    Returns
    -------
    jax.numpy.array with output of layer
    """
    #Apply each operator
    if type != "wop":
        oper = mp.operator(type)
        fx = None
        for i in range(params.shape[0]):
            if fx is None:
                fx = oper(x,index_x,activate(params[i,:,:,:])).reshape((1,x.shape[0],x.shape[1],x.shape[2]))
            else:
                fx = jnp.append(fx,oper(x,index_x,activate(params[i,:,:,:])).reshape((1,x.shape[0],x.shape[1],x.shape[2])),0)
    else:
        fx = mp.w_operator_nn(x,index_x,forward_wop,params,d).reshape((1,x.shape[0],x.shape[1],x.shape[2]))
    return fx

#Canonical Morphological NN
def cmnn(type,width,size,shape_x,sample = False,p1 = 0.1,key = 0,width_wop = None,activation = jax.nn.sigmoid):
    """
    Initialize a Discrete Morphological Neural Network as the identity operator.
    ----------

    Parameters
    ----------
    type : list of str

        List with the names of the operators to be applied by each layer

    width : list of int

        List with the width of each layer

    size : list of int

        List with the size of the structuring element of each layer

    shape_x : list

        Shape of the input images

    sample : logical

        Whether to sample initial parameters

    p1 : float

        Expected proportion of ones in sampled parameters

    key : int

        Key for sampling

    width_wop : list of int

        List with the width of each layer of the W-operator neural network

    activation : function

        Activation function for W-operator neural network

    Returns
    -------
    dictionary with the initial parameters, forward function, width, size, type and forward funtion of the W-operator
    """
    key = jax.random.split(jax.random.PRNGKey(key),(2*len(width),max(width)))
    k = 0
    forward_wop = None

    #Index window
    index_x = dmp.index_array(shape_x)

    #Initialize parameters
    params = list()
    for i in range(len(width)):
        if type[i] in ['sup','inf','complement']:
            params.append(jnp.array(0.0).reshape((1,1,1)))
        elif type[i] == "wop":
            net = fconNN_wop(width_wop,size[i],activation,key[k,0,0])
            k = k + 1
            forward_wop = net['forward']
            params.append(net['params'])
        else:
            if sample:
                if type[i] == 'supgen' or type[i] == 'infgen':
                    ll = np.zeros((1,1,size[i],size[i]),dtype = float)
                    ll[0,0,int(np.round(size[i]/2 - 0.1)),int(np.round(size[i]/2 - 0.1))] = 1.0
                    ll = jnp.array(ll)
                    ul = 1.0 + jnp.zeros((1,1,size[i],size[i]),dtype = float)
                    for j in range(width[i]):
                        s = jax.random.choice(jax.random.PRNGKey(key[k,0,0]),2,p = jnp.array([1 - p1,p1]),shape = (1,1,size[i],size[i]))
                        k = k + 1
                        s = jnp.maximum(ll,s)
                        if j == 0:
                            p = jnp.append(s,ul,1)
                        else:
                            interval = jnp.append(s,ul,1)
                            p = jnp.append(p,interval,0)
                else:
                    ll = np.zeros((1,1,size[i],size[i]),dtype = int)
                    ll[0,0,int(np.round(size[i]/2 - 0.1)),int(np.round(size[i]/2 - 0.1))] = 1
                    ll = jnp.array(ll)
                    for j in range(width[i]):
                        s = jax.random.choice(jax.random.PRNGKey(key[k,0,0]),2,p = jnp.array([1 - p1,p1]),shape = (1,1,size[i],size[i]))
                        k = k + 1
                        s = jnp.maximum(ll,s)
                        if j == 0:
                            p = jnp.array(s)
                        else:
                            p = jnp.append(p,s,0)
            else:
                if type[i] == 'supgen' or type[i] == 'infgen':
                    ll = np.zeros((1,1,size[i],size[i]),dtype = int)
                    ll[0,0,int(np.round(size[i]/2 - 0.1)),int(np.round(size[i]/2 - 0.1))] = 1
                    ll = jnp.array(ll)
                    ul = 1 + jnp.zeros((1,1,size[i],size[i]),dtype = int)
                    p = jnp.append(ll,ul,1)
                    for j in range(width[i] - 1):
                        interval = jnp.append(ll,ul,1)
                        p = jnp.append(p,interval,0)
                else:
                    ll = np.zeros((1,1,size[i],size[i]),dtype = int)
                    ll[0,0,int(np.round(size[i]/2 - 0.1)),int(np.round(size[i]/2 - 0.1))] = 1
                    ll = jnp.array(ll)
                    p = ll
                    for j in range(width[i] - 1):
                        interval = ll
                        p = jnp.append(p,interval,0)
            params.append(p.astype(jnp.float32) + 0.01*jax.random.normal(key[k,0],p.shape,jnp.float32))
            k = k + 1

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
                x = apply_morph_layer(x[0,:,:,:],type[i],params[i],index_x,forward_wop,size[i])
        return x[0,:,:,:]

    #Return initial parameters and forward function
    return {'params': params,'forward': forward,'forward_wop': forward_wop,'type': type,'width': width,'size': size}


#Canonical Morphological NN with iterated NN
def cmnn_iter(type,width,width_str,size,shape_x,width_wop = None,activation = jax.nn.sigmoid,identity = True,sample = False,p1 = 0.1,key = 0,loss = MSE_SA,sa = True,c = 100,q = 2,epochs = 5000,batches = 1,lr = 0.001,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0,notebook = False,epochs_print = 500):
    """
    Initialize a Morphological Neural Network with FCNN representing the structuring elements.
    ----------

    Parameters
    ----------
    type : list of str

        List with the names of the operators to be applied by each layer

    width : list of int

        List with the width of each layer

    width : list of int

        List with the width of each layer of the FCNN for structuring elements

    size : list of int

        List with the size of the structuring element of each layer

    shape_x : list

        Shape of the input images

    width_wop : list of int

        List with the width of each layer of the W-operator neural network

    activation : function

        Activation function for W-operator neural network

    identity : logical

        Whether to initiate the parameters so the operator is a random perturbation of the identity operator

    sample : logical

        Whether to sample initial parameters

    p1 : float

        Expected proportion of ones in sampled parameters

    key : int

        Key for sampling

    loss : function

        loss function

    sa : logical

        Whether to consider self-adaptative weights

    c,q : float

        Hyperparameters of self-adaptative weights

    epochs : int

        Number of training epochs

    batches : int

        Number of batches

    lr,b1,b2,eps,eps_root: float

        Hyperparameters of the Adam algorithm. Default lr = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-08, eps_root = 0.0

    key : int

        Seed for parameters initialization. Default 0

    notebook : logical

        Whether code is being executed in a notebook

    epoch_print : int

        Number of epochs to print training error

    Returns
    -------
    dictionary with the initial parameters, forward function, function to comput structuring elements, width, size, type, forward funtion of the W-operator and forward function of the inner FCNN
    """
    #Index window
    index_x = dmp.index_array(shape_x)

    #Create w to apply str NN
    unique_size = set(size)
    w = {}
    for d in unique_size:
        w[str(d)] = jnp.array([[x1.tolist(),x2.tolist()] for x1 in jnp.linspace(-jnp.floor(d/2),jnp.floor(d/2),d) for x2 in jnp.linspace(jnp.floor(d/2),-jnp.floor(d/2),d)])

    #Init params
    init_net = cmnn(type,width,size,shape_x,sample = sample,p1 = p1,key = key,width_wop = width_wop,activation = activation)
    init_params = init_net['params']
    forward_wop = init_net['forward_wop']

    #Initialize parameters NN
    params = list()
    if identity:
        for i in range(len(width)):
            if type[i] in ['sup','inf','complement','wop']:
                params.append(init_params[i])
            else:
                if type[i] in ['supgen','infgen']:
                    par_layer = list()
                    for j in range(width[i]):
                        #Lower
                        nn = fconNN([2] + width_str + [1],activation,key)
                        forward_inner = lambda x,params: jax.nn.sigmoid(nn['forward'](x,params))
                        w_input = w[str(size[i])]
                        w_output = init_params[i][j,0,:,:].transpose().reshape((w_input.shape[0],1))
                        params_lower = sgd(w_input,w_output,forward_inner,nn['params'],loss,sa,c,q,epochs,batches,lr,b1,b2,eps,eps_root,key,notebook)
                        #Upper
                        nn = fconNN([2] + width_str + [1],activation,key)
                        forward_inner = lambda x,params: jax.nn.sigmoid(nn['forward'](x,params))
                        w_input = w[str(size[i])]
                        w_output = init_params[i][j,1,:,:].transpose().reshape((w_input.shape[0],1))
                        params_upper = sgd(w_input,w_output,forward_inner,nn['params'],loss,sa,c,q,epochs,batches,lr,b1,b2,eps,eps_root,key,notebook)
                        par_layer.append([params_lower,params_upper])
                    params.append(par_layer)
                else:
                    par_layer = list()
                    for j in range(width[i]):
                        nn = fconNN([2] + width_str + [1],activation,key)
                        forward_inner = lambda x,params: jax.nn.sigmoid(nn['forward'](x,params))
                        w_input = w[str(size[i])]
                        w_output = init_params[i][j,:,:,:].transpose().reshape((w_input.shape[0],1))
                        params_str = sgd(w_input,w_output,forward_inner,nn['params'],loss,sa,c,q,epochs,batches,lr,b1,b2,eps,eps_root,key,notebook)
                        par_layer.append(params_str)
                    params.append(par_layer)
    else:
        key = jax.random.split(jax.random.PRNGKey(key),(2*len(width)*max(width)))
        k = 1
        for i in range(len(width)):
            if type[i] in ['sup','inf','complement','wop']:
                params.append(init_params[i])
            else:
                if type[i] in ['supgen','infgen']:
                    par_layer = list()
                    for j in range(width[i]):
                        #Lower
                        nn_lower = fconNN([2] + width_str + [1],activation,key[k,0])
                        k = k + 1
                        forward_inner = lambda x,params: jax.nn.sigmoid(nn_lower['forward'](x,params))
                        params_lower = nn_lower['params']
                        #Upper
                        nn_upper = fconNN([2] + width_str + [1],activation,key[k,0])
                        k = k + 1
                        forward_inner = lambda x,params: jax.nn.sigmoid(nn_upper['forward'](x,params))
                        params_upper = nn_upper['params']
                        par_layer.append([params_lower,params_upper])
                    params.append(par_layer)
                else:
                    par_layer = list()
                    for j in range(width[i]):
                        nn_str = fconNN([2] + width_str + [1],activation,key[k,0])
                        k = k + 1
                        forward_inner = lambda x,params: jax.nn.sigmoid(nn_str['forward'](x,params))
                        params_str = nn_lower['params']
                        params_str = nn_str['params']
                        par_layer.append(params_str)
                    params.append(par_layer)

    #Compute structuring elements
    @jax.jit
    def compute_struct(params):
        #Compute for each layer
        struct = list()
        for i in range(len(type)):
            if type[i] in ['sup','inf','complement','wop']:
                struct.append(params[i])
            elif type[i] in ['infgen','supgen']:
                par = forward_inner(w[str(size[i])],params[i][0][0]).reshape((size[i],size[i])).transpose().reshape((1,size[i],size[i]))
                par = jnp.append(par,forward_inner(w[str(size[i])],params[i][0][1]).reshape((size[i],size[i])).transpose().reshape((1,size[i],size[i])),0).reshape((1,2,size[i],size[i]))
                for j in range(width[i] - 1):
                    tmp = forward_inner(w[str(size[i])],params[i][j + 1][0]).reshape((size[i],size[i])).transpose().reshape((1,size[i],size[i]))
                    tmp = jnp.append(tmp,forward_inner(w[str(size[i])],params[i][j + 1][1]).reshape((size[i],size[i])).transpose().reshape((1,size[i],size[i])),0).reshape((1,2,size[i],size[i]))
                    par = jnp.append(par,tmp,0)
                struct.append(par)
            else:
                par = forward_inner(w[str(size[i])],params[i][0]).reshape((size[i],size[i])).transpose().reshape((1,1,size[i],size[i]))
                for j in range(width[i] - 1):
                    par = jnp.append(par,forward_inner(w[str(size[i])],params[i][j + 1]).reshape((size[i],size[i])).transpose().reshape((1,size[i],size[i])),0)
                struct.append(par)
        return struct

    #Forward pass
    @jax.jit
    def forward(x,params):
        params_array = compute_struct(params)
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
                x = apply_morph_layer(x[0,:,:,:],type[i],params_array[i],index_x,forward_wop,size[i])
        return x[0,:,:,:]

    #Return initial parameters and forward function
    return {'params': params,'forward': forward,'compute_struct': compute_struct,'forward_wop': forward_wop,'forward_inner': forward_inner,'width': width,'size': size,'type': type}
