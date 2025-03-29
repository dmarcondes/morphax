#Discrete morphology module using CV2
import math
import numpy as np
import cv2 as cv
import random
import time
from alive_progress import alive_bar
from joblib import Parallel, delayed
import os

#Transpose a structuring element
def transpose_se(k):
    """
    Transpose a structuring element.
    -------

    Parameters
    ----------
    k : numpy.array

        A 2D structuring element

    Returns
    -------

    a numpy array

    """
    d = k.shape[0]
    kt = k
    for i in range(d):
        for j in range(d):
            kt[i,j] = k[d - 1 - i,d - 1 - j]
    return kt

#Erosion in batches
def erosion(f,k):
    """
    Erosion of batches of images.
    -------

    Parameters
    ----------
    f : numpy.array

        A 3D array with the binary images

    k : numpy.array

        Structuring element

    Returns
    -------

    a numpy array

    """
    l = math.floor(k.shape[0]/2)
    f = np.pad(f, [(0,0), (l,l), (l,l)], mode='constant')
    eb = cv.erode(f,k,iterations = 1)[:,l:-l,l:-l]
    return eb

#Dilation in batches
def dilation(f,k):
    """
    Dilation of batches of images.
    -------

    Parameters
    ----------
    f : numpy.array

        A 3D array with the binary images

    k : numpy.array

        Structuring element

    Returns
    -------

    a numpy array

    """
    l = math.floor(k.shape[0]/2)
    f = np.pad(f, [(0,0), (l,l), (l,l)], mode='constant')
    db = cv.dilate(f,k,iterations = 1)[:,l:-l,l:-l]
    return db

#Opening of f by k
def opening(f,k):
    """
    Opening of batches of images.
    -------

    Parameters
    ----------
    f : numpy.array

        A 3D array with the binary images

    k : numpy.array

        Structuring element

    Returns
    -------

    a numpy array

    """
    l = math.floor(k.shape[0]/2)
    f = np.pad(f, [(0,0), (l,l), (l,l)], mode='constant')
    op = cv.morphologyEx(f, cv.MORPH_OPEN, k)[:,l:-l,l:-l]
    return op

#Colosing of f by k
def closing(f,k):
    """
    Closing of batches of images.
    -------

    Parameters
    ----------
    f : numpy.array

        A 3D array with the binary images

    k : numpy.array

        Structuring element

    Returns
    -------

    a numpy array

    """
    l = math.floor(k.shape[0]/2)
    f = np.pad(f, [(0,0), (l,l), (l,l)], mode='constant')
    cl = cv.morphologyEx(f, cv.MORPH_CLOSE, k)[:,l:-l,l:-l]
    return cl

#Alternate-sequential filter of f by k
def asf(f,k):
    """
    Alternate-sequential filter applied to batches of images.
    -------

    Parameters
    ----------
    f : numpy.array

        A 3D array with the binary images

    k : numpy.array

        Structuring element

    Returns
    -------

    a numpy array

    """
    return closing(opening(f,k),k)

#Complement
def complement(f):
    """
    Complement batches of images.
    -------

    Parameters
    ----------
    f : numpy.array

        A 3D array with the binary images

    Returns
    -------

    a numpy array

    """
    return 1 - f

#Sup-generating with interval [k1,k2]
def supgen(f,k1,k2):
    """
    Sup-generating operator applied to batches of images.
    -------

    Parameters
    ----------
    f : numpy.array

        A 3D array with the binary images

    k1,k2 : numpy.array

        Limits of interval [k1,k2]

    Returns
    -------

    a numpy array

    """
    return np.minimum(erosion(f,k1),complement(dilation(f,complement(transpose_se(k2)))))

#Inf-generating with interval [k1,k2]
def infgen(f,k1,k2):
    """
    Inf-generating operator applied to batches of images.
    -------

    Parameters
    ----------
    f : numpy.array

        A 3D array with the binary images

    k1,k2 : numpy.array

        Limits of interval [k1,k2]

    Returns
    -------

    a numpy array

    """
    return np.maximum(dilation(f,k1),complement(erosion(f,complement(transpose_se(k2)))))

#Sup of array of images
def sup(f):
    """
    Supremum of batches of images.
    -------

    Parameters
    ----------
    f : numpy.array

        A 3D array with the binary images

    Returns
    -------

    a numpy array

    """
    return np.apply_along_axis(np.max,0,f)

#Inf of array of images
def inf(f):
    """
    Infimum of batches of images.
    -------

    Parameters
    ----------
    f : numpy.array

        A 3D array with the binary images

    Returns
    -------

    a numpy array

    """
    return np.apply_along_axis(np.min,0,f)

#Return operator by name
def operator(type):
    """
    Get a morphological operator by name.
    -------

    Parameters
    ----------
    type : str

        Name of operator

    Returns
    -------

    a function that applies a morphological operator

    """
    if type == 'erosion':
        oper = lambda x,k: erosion(x,k[0,:,:].reshape((k.shape[1],k.shape[2])))
    elif type == 'dilation':
        oper = lambda x,k: dilation(x,k[0,:,:].reshape((k.shape[1],k.shape[2])))
    elif type == 'opening':
        oper = lambda x,k: opening(x,k[0,:,:].reshape((k.shape[1],k.shape[2])))
    elif type == 'closing':
        oper = lambda x,k: closing(x,k[0,:,:].reshape((k.shape[1],k.shape[2])))
    elif type == 'asf':
        oper = lambda x,k: asf(x,k[0,:,:].reshape((k.shape[1],k.shape[2])))
    elif type == 'supgen':
        oper = lambda x,k: supgen(x,k[0,:,:].reshape((k.shape[1],k.shape[2])),k[1,:,:].reshape((k.shape[1],k.shape[2])))
    elif type == 'infgen':
        oper = lambda x,k: infgen(x,k[0,:,:].reshape((k.shape[1],k.shape[2])),k[1,:,:].reshape((k.shape[1],k.shape[2])))
    else:
        print('Type of layer ' + type + 'is wrong!')
        return 1
    return oper

####Discrete Morphological Neural Networks####
#MSE
def MSE(pred,true):
    """
    Mean square error
    ----------

    Parameters
    ----------
    pred : numpy.array

        A numpy array with the predicted values

    true : numpy.array

        A numpy array with the true values

    Returns
    -------
    mean square error
    """
    return np.sum((1/(pred.shape[0]*pred.shape[1]*pred.shape[2]))*((true - pred) ** 2))

#L2 error
def L2error(pred,true):
    """
    L2 error
    ----------

    Parameters
    ----------
    pred : numpy.array

        A numpy array with the predicted values

    true : numpy.array

        A numpy array with the true values

    Returns
    -------
    L2 error
    """
    e = 0
    for i in range(true.shape[0]):
        e = e + np.sqrt(np.sum((true[i,:,:] - pred[i,:,:])**2))/np.sqrt(np.sum(true[i,:,:] ** 2))
    return e/true.shape[0]

#Croos entropy
def CE(pred,true):
    """
    Cross entropy error
    ----------

    Parameters
    ----------
    pred : numpy.array

        A numpy array with the predicted values

    true : numpy.array

        A numpy array with the true values

    Returns
    -------
    cross-entropy error
    """
    e = 0
    for i in range(true.shape[0]):
        e = e + np.mean((- true[i,:,:] * np.log(pred[i,:,:] + 1e-6) - (1 - true[i,:,:]) * np.log(1 - pred[i,:,:] + 1e-6)))
    return e/true.shape[0]

#IoU
def IoU(pred,true):
    """
    Intersection over union error
    ----------

    Parameters
    ----------
    pred : numpy.array

        A numpy array with the predicted values

    true : numpy.array

        A numpy array with the true values

    Returns
    -------
    intersection over union error
    """
    e = 0
    for i in range(true.shape[0]):
        e = e + 1 - (np.sum(np.minimum(pred[i,:,:],true[i,:,:]))/np.sum(np.maximum(pred[i,:,:],true[i,:,:])))
    return e/true.shape[0]

#Apply a morphological layer
def apply_morph_layer(x,type,params):
    """
    Apply a morphological layer.
    ----------

    Parameters
    ----------
    x : numpy.array

        A numpy array with binary images

    type : str

        Names of the operator to apply

    params : numpy.array

        Parameters of the operators

    Returns
    -------
    numpy.array with output of layer
    """
    #Which operator
    oper = operator(type)
    fx = None
    #For each node
    for i in range(params.shape[0]):
        if fx is None:
            fx = oper(x,params[i,:,:,:]).reshape((1,x.shape[0],x.shape[1],x.shape[2]))
        else:
            fx = np.append(fx,oper(x,params[i,:,:,:]).reshape((1,x.shape[0],x.shape[1],x.shape[2])),0)
    return fx

#Initiliaze Canonical DMNN
def cdmnn(type,width,size,sample = False,p1 = 0.5,key = 0):
    """
    Initialize a Discrete Morphological Neural Network as the identity operator or randomly.
    ----------

    Parameters
    ----------
    type : list of str

        List with the names of the operators to be applied by each layer

    width : list of int

        List with the width of each layer

    size : list of int

        List with the size of the structuring element of each layer

    sample : logical

        Whether to sample initial parameters

    p1 : float

        Expected proportion of ones in sampled parameters

    key : int

        Key for sampling

    Returns
    -------
    dictionary with the initial parameters, forward function, width, size and type
    """
    #Seed
    np.random.seed(key)
    #Initialize parameters
    params = list()
    for i in range(len(width)):
        if type[i] not in ['sup','inf','complement']:
            if sample:
                if type[i] == 'supgen' or type[i] == 'infgen':
                    ll = np.zeros((1,1,size[i],size[i]),dtype = int)
                    ll[0,0,int(np.round(size[i]/2 - 0.1)),int(np.round(size[i]/2 - 0.1))] = 1
                    ll = np.array(ll)
                    ul = 1 + np.zeros((1,1,size[i],size[i]),dtype = int)
                    for j in range(width[i]):
                        s = np.random.choice(2,p = np.array([1 - p1,p1]),size = (1,1,size[i],size[i]))
                        k = k + 1
                        s = np.maximum(ll,s)
                        if j == 0:
                            p = np.append(s,ul,1)
                        else:
                            interval = np.append(s,ul,1)
                            p = np.append(p,interval,0)
                else:
                    ll = np.zeros((1,1,size[i],size[i]),dtype = int)
                    ll[0,0,int(np.round(size[i]/2 - 0.1)),int(np.round(size[i]/2 - 0.1))] = 1
                    ll = np.array(ll)
                    for j in range(width[i]):
                        s = np.random.choice(2,p = np.array([1 - p1,p1]),size = (1,1,size[i],size[i]))
                        k = k + 1
                        s = np.maximum(ll,s)
                        if j == 0:
                            p = np.array(s)
                        else:
                            p = np.append(p,s,0)
            else:
                if type[i] == 'supgen' or type[i] == 'infgen':
                    ll = np.zeros((1,1,size[i],size[i]),dtype = int)
                    ll[0,0,int(np.round(size[i]/2 - 0.1)),int(np.round(size[i]/2 - 0.1))] = 1
                    ll = np.array(ll)
                    ul = 1 + np.zeros((1,1,size[i],size[i]),dtype = int)
                    p = np.append(ll,ul,1)
                    for j in range(width[i] - 1):
                        interval = np.append(ll,ul,1)
                        p = np.append(p,interval,0)
                else:
                    ll = np.zeros((1,1,size[i],size[i]),dtype = int)
                    ll[0,0,int(np.round(size[i]/2 - 0.1)),int(np.round(size[i]/2 - 0.1))] = 1
                    ll = np.array(ll)
                    p = ll
                    for j in range(width[i] - 1):
                        interval = ll
                        p = np.append(p,interval,0)
            params.append(p)
    #Params as jax array
    max_width = max(width)
    max_size = max(size)
    for i in range(len(params)):
        params[i] = np.pad(params[i],((0,max_width - params[i].shape[0]),(0,2 - params[i].shape[1]),(0,max_size - params[i].shape[2]),(0,max_size - params[i].shape[3])),constant_values = -2)
    params = np.array(params)
    #Numeric type
    type_num = []
    for i in range(len(type)):
        if type[i] in ['sup','inf','complement']:
            type_num = type_num + [0]
        elif type[i] == 'supgen' or type[i] == 'infgen':
            type_num = type_num + [2]
        else:
            type_num = type_num + [1]
    #Forward pass
    def forward(x,params):
        x = x.reshape((1,x.shape[0],x.shape[1],x.shape[2]))
        j = 0
        for i in range(len(type)):
            #Apply sup and inf
            if type[i] == 'sup':
                x = sup(x).reshape((1,x.shape[1],x.shape[2],x.shape[3]))
            elif type[i] == 'inf':
                x =  inf(x).reshape((1,x.shape[1],x.shape[2],x.shape[3]))
            elif type[i] == 'complement':
                x = 1 - x
            elif type[i] == "supgen" or type[i] == "infgen":
                #Apply other layer
                x = apply_morph_layer(x[0,:,:,:],type[i],params[j,0:width[i],:,0:size[i],0:size[i]].reshape((width[i],2,size[i],size[i])))
                j = j + 1
            else:
                #Apply other layer
                x = apply_morph_layer(x[0,:,:,:],type[i],params[j,0:width[i],0,0:size[i],0:size[i]].reshape((width[i],1,size[i],size[i])))
                j = j + 1
        return x[0,:,:,:]
    #Return initial parameters and forward function
    return {'params': params.astype(np.uint8),'forward': forward,'width': width,'size': size,'type': type_num}

#Visit a nighboor
def visit_neighbor(h,params,x,y,lf):
    """
    Compute the loss at a neighbor.
    ----------

    Parameters
    ----------
    h : numpy.array

        Index of the neighbor to visit

    params : numpy.array

        Array of current parameters

    x,y : numpy.array

        Input and output images

    lf : function

        Loss function

    Returns
    -------
    float
    """
    #Compute error
    params[h[0],h[1],h[2],h[3],h[4]] = 1 - params[h[0],h[1],h[2],h[3],h[4]]
    return lf(params,x,y)

#Step SLDA
def step_slda(params,x,y,forward,lf,type,width,size,neighbors = 8,key = 0):
    """
    Step of Stochastic Lattice Descent Algorithm updating the parameters.
    ----------

    Parameters
    ----------
    params : numpy.array

        Current parameters

    x,y : numpy.array

        Input and output images

    forward : function

        Forward function

    lf : function

        Loss function

    type : list of str

        Type of operator applied in each layer

    width : list of int

        List with the width of each layer

    size : list of int

        List with the size of the structuring element of each layer

    neighbors : int

        Number of neighbors to sample

    key : int

        Key for sampling

    Returns
    -------
    list of numpy.array
    """
    #Key
    np.random.seed(key)
    #Store neighbors to consider: layer + node + lim + row + col
    #Calculate probabilities of each layer
    prob = np.array([])
    j = 0
    for i in range(len(type)):
        if type[i] != 0:
            #If is not supgen/infgen
            if type[i] == 1:
                prob  = np.append(prob,np.array(width[i] * (size[i] ** 2)))
            #If supgen/infgen
            else:
                count = np.apply_along_axis(np.sum,1,params[j,0:width[i],:,0:size[i],0:size[i]])
                prob = np.append(prob,np.sum(np.where((count == 0) | (count == 2),1,2)))
                del count
            j = j + 1
    #Arrays
    max_width = max(width)
    max_size = max(size)
    #Sample layers
    prob = prob.astype(np.float32)
    prob = prob/np.sum(prob)
    prob = np.append(prob[:-1],np.array(1 - np.sum(prob[:-1])))
    layers = np.random.choice(len(prob),size = (neighbors,),p = prob)
    hood = None
    #For each layer sample a change
    for i in range(neighbors):
        l = layers[i]
        #Sample a node
        par_l = params[l,:,:,:,:]
        count_sum = np.apply_along_axis(np.sum,1,par_l)
        count = np.where(count_sum == 1,2,1)
        count = np.where(count_sum == -4,0,count)
        tmp_prob = np.sum(count.reshape((count.shape[0],count.shape[1]*count.shape[2])).transpose(),0).astype(np.float32)
        tmp_prob = tmp_prob/np.sum(tmp_prob)
        tmp_prob = np.append(tmp_prob[:-1],np.array(1 - np.sum(tmp_prob[:-1])))
        tmp_prob = tmp_prob/np.sum(tmp_prob)
        node = np.random.choice(max_width,size = (1,),p = tmp_prob)
        #Sample row and collumn
        tmp_prob = count[node,:,:].reshape((max_size ** 2)).astype(np.float32)
        tmp_prob = tmp_prob/np.sum(tmp_prob)
        tmp_prob = np.append(tmp_prob[:-1],np.array(1 - np.sum(tmp_prob[:-1])))
        tmp_prob = tmp_prob/np.sum(tmp_prob)
        tmp_random = np.random.choice(max_size ** 2,size = (1,),p = tmp_prob)
        rc = np.array([np.floor(tmp_random/max_size),tmp_random % max_size]).reshape((1,2)).astype(np.int32)
        #Sample limit
        lim = par_l[node,:,rc[0,0],rc[0,1]]
        if lim[0,0] == 1 and lim[0,1] == 1:
            lim = 0
        elif lim[0,0] == 0 and lim[0,1] == 0:
            lim = 1
        else:
            lim = np.random.choice(2)
        #Neighbor
        nei = np.append(np.append(np.append(np.array([l]),node),lim),rc)
        if hood is None:
            hood = nei.reshape((1,nei.shape[0]))
        else:
            hood = np.append(hood,nei.reshape((1,nei.shape[0])),0)
        del count, tmp_prob, tmp_random, par_l, node, rc, lim
    #Compute error in parallel
    def compute_error(i):
        return  visit_neighbor(hood[i,:],params,x,y,lf)
    error = np.array(Parallel(n_jobs = min(os.cpu_count()-1,hood.shape[0]))(delayed(compute_error)(i) for i in range(hood.shape[0])))
    return hood,error

#Threshold gray scale image
def threshold(X,t):
    return np.where(X >= t,1.0,0.0)

#Training function DMNN via SLDA
def train_dmnn_slda(x,y,net,loss,xval = None,yval = None,stack = False,neighbors = 8,epochs = 1,batches = 1,K = 255,notebook = False,epoch_print = 100,epoch_store = 1,key = 0,store_jumps = False,error_type = 'mean'):
    """
    Stochastic Lattice Descent Algorithm to train Discrete Morphological Neural Networks for stack operators.
    ----------

    Parameters
    ----------
    x,y : numpy.array

        Input and output images

    net : dict

        Dictionary returned by the function cdmnn

    loss : function

        Loss function

    xval,yval : numpy.array

        Input and output validation images

    stack : logical

        Whether to train a stack operator

    neighbors : int

        Number of neighbors to sample

    epochs : int

        Number of epochs

    epochs : int

        Number of bacthes

    K : int

        Length of gray scale for stack operators

    notebook : logical

        Wheter the code is being run in a notebook. Has an effect on tracing the Algorithm

    epoch_print : int

        Number of epochs to print the partial result

    epoch_store : int

        Number of epochs to store partial results

    key : int

        Key for sampling

    store_jumps : logical

        Whether to store jumps

    error_type : str

        Type of error to consider: 'mean' of loss or 'max' of loss

    Returns
    -------
    list of numpy.array
    """
    print('----- Initializing parameters -----')
    #Parameters
    params = net['params']
    forward = net['forward']
    type = net['type']
    width = net['width']
    size = net['size']
    stacks = 1 + np.arange(K)
    #Key
    np.random.seed(key)
    key = np.arange(epochs)
    #Batch size
    bsize = int(math.floor(x.shape[0]/batches))
    #Loss function
    if stack:
        x = np.apply_along_axis(lambda t: threshold(x,t),0,stacks.reshape((1,K)))
        def pred(params,x):
            return np.sum(np.apply_along_axis(lambda x: forward(x,params),3,x),0)
        if error_type == 'mean':
            def lf(params,x,y):
                return np.mean(loss(pred(params,x),y))
        else:
            def lf(params,x,y):
                return np.max(loss(pred(params,x),y))
    else:
        if error_type == 'mean':
            def lf(params,x,y):
                return np.mean(loss(forward(x,params),y))
        else:
            def lf(params,x,y):
                return np.max(loss(forward(x,params),y))
    #Training function
    def update(params,x,y,key,jumps):
        hood,error = step_slda(params,x,y,forward,lf,type,width,size,neighbors,key)
        hood = hood[np.argsort(error),:]
        error = error[np.argsort(error)]
        hood = hood[0,:]
        train_loss = error[0]
        if store_jumps:
          jumps = np.append(jumps,hood,0)
        params[hood[0],hood[1],hood[2],hood[3],hood[4]] = 1 - params[hood[0],hood[1],hood[2],hood[3],hood[4]]
        return params,jumps,train_loss
    #Trace
    best_par = params.copy()
    min_loss = lf(params,x,y)
    trace_time = [0]
    trace_loss = [min_loss]
    trace_epoch = [0]
    if xval is not None:
        xval = np.apply_along_axis(lambda t: threshold(xval,t),0,stacks.reshape((1,K)))
        min_val_loss = lf(params,xval,yval)
        trace_val_loss = [min_val_loss]
    else:
        min_val_loss = np.inf
        trace_val_loss = []
    params_init = params.copy()
    jumps = np.array([])
    #Step epoch
    def step_epoch(key,params,best_par,min_loss,jumps,x,y,xval,yval):
        if batches > 1:
            per = np.random.permutation(np.arange(x.shape[2]))
            x = x[:,per,:,:]
            y = y[per,:,:]
        for b in range(batches):
            if batches > 1:
                if b < batches - 1:
                    xb = x[:,b*bsize:(b+1)*bsize,:,:]
                    yb = y[:,b*bsize:(b+1)*bsize,:,:]
                else:
                    xb = x[:,b*bsize:x.shape[1],:,:]
                    yb = y[b*bsize:y.shape[0],:,:]
                #Search neighbors
                params,jumps,train_loss = update(params,x,y,key,jumps)
            else:
                #Search neighbors
                params,jumps,train_loss = update(params,x,y,key,jumps)
        #Compute loss and store at the end of epoch
        if batches > 1:
            train_loss = lf(params,x,y)
        #Update best
        best_par = (train_loss < min_loss) * params + (train_loss >= min_loss) * best_par
        min_loss = np.minimum(train_loss,min_loss)
        if xval is not None:
            min_val_loss = lf(best_par,xval,yval)
        return params,best_par,min_loss,jumps,train_loss
    #Train
    print('----- Begin epochs -----')
    t0 = time.time()
    with alive_bar(epochs) as bar:
        bar.title('Epoch: ' + str(0) + " Loss: " + str(np.round(min_loss,5)) + ' Best: ' + str(np.round(min_loss,5)) + ' Val: ' + str(np.round(min_val_loss,5)))
        for e in range(epochs):
            #Take step
            params,best_par,min_loss,jumps,train_loss = step_epoch(key[e],params,best_par,min_loss,jumps,x,y,xval,yval)
            #Store
            if (e + 1) % epoch_store == 0:
                trace_epoch = trace_epoch + [e + 1]
                trace_time = trace_time + [time.time() - t0]
                trace_loss = trace_loss + [train_loss]
                if xval is not None:
                    val_loss = lf(params,xval,yval)
                    trace_val_loss = trace_val_loss + [val_loss]
            bar.title('Epoch: ' + str(e) + " Time: " + str(np.round(time.time() - t0)) + " s Loss: " + str(np.round(train_loss,5)) + ' Best: ' + str(np.round(min_loss,5)) + ' Val: ' + str(np.round(min_val_loss,5)))
            if e % epoch_print == 0:
                if notebook:
                    print('Epoch: ' + str(e) + " Time: " + str(np.round(time.time() - t0)) + " s Loss: " + str(np.round(train_loss,5)) + ' Best: ' + str(np.round(min_loss,5)) + ' Val: ' + str(np.round(min_val_loss,5)))
            bar()
    return {'best_par': best_par,'jumps': jumps,'trace_epoch': trace_epoch,'trace_time': trace_time,'trace_loss': trace_loss,'trace_val_loss': trace_val_loss,'epochs': epochs,'oper': lambda x: np.sum(jax.vmap(lambda x: forward(x,params))(jax.vmap(lambda t: threshold(x,t))(stacks)),0),'forward': forward}
