#Discrete morphology module
import jax
import jax.numpy as jnp
import math
import numpy as np
import random
import time
from alive_progress import alive_bar

#Create an index array for an array
def index_array(shape):
    """
    Create a 2D JAX array with the indexes of an array with given 2D shape.
    -------

    Parameters
    ----------
    shape : list

        List with shape of 2D array

    Returns
    -------

    a JAX numpy array

    """
    return jnp.array([[x,y] for x in range(shape[0]) for y in range(shape[1])])

#Transpose a structuring element
@jax.jit
def transpose_se(k):
    """
    Transpose a structuring element.
    -------

    Parameters
    ----------
    k : jax.numpy.array

        A 2D structuring element

    Returns
    -------

    a JAX numpy array

    """
    d = k.shape[0]
    kt = k
    for i in range(d):
        for j in range(d):
            kt = kt.at[i,j].set(k[d - 1 - i,d - 1 - j])
    return kt

#Local erosion of f by k for pixel (i,j)
def local_erosion(f,k,l):
    """
    Define function for local erosion.
    -------

    Parameters
    ----------
    f : jax.numpy.array

        A binary image

    k : jax.numpy.array

        A structuring element

    l : int

        Length of structruring element. If k has shape d x d, then l is the greatest integer such that l <= d/2

    Returns
    -------

    a function that receives an index and returns the local erosion of f by k at this index

    """
    def jit_local_erosion(index):
        fw = jax.lax.dynamic_slice(f, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
        return jnp.min(jnp.where(k == 1,fw,1))
    return jit_local_erosion

#Erosion of f by k
@jax.jit
def erosion_2D(f,index_f,k):
    """
    Erosion of 2D image.
    -------

    Parameters
    ----------
    f : jax.numpy.array

        A padded binary image

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    Returns
    -------

    a JAX numpy array

    """
    l = math.floor(k.shape[0]/2)
    jit_local_erosion = local_erosion(f,k,l)
    return jnp.apply_along_axis(jit_local_erosion,1,l + index_f).reshape((f.shape[0] - 2*l,f.shape[1] - 2*l))

#Erosion in batches
@jax.jit
def erosion(f,index_f,k):
    """
    Erosion of batches of images.
    -------

    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the binary images

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    Returns
    -------

    a JAX numpy array

    """
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    eb = jax.vmap(lambda f: erosion_2D(f,index_f,k),in_axes = (0),out_axes = 0)(f)
    return eb

#Local dilation of f by k for pixel (i,j)
def local_dilation(f,kt,l):
    """
    Define function for local dilation.
    -------

    Parameters
    ----------
    f : jax.numpy.array

        A binary image

    kt : jax.numpy.array

        The transpose of the structuring element

    l : int

        Length of structruring element. If k has shape d x d, then l is the greatest integer such that l <= d/2

    Returns
    -------

    a function that receives an index and returns the local dilation of f by k at this index

    """
    def jit_local_dilation(index):
        fw = jax.lax.dynamic_slice(f, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
        return jnp.max(jnp.where(kt == 1, fw, 0))
    return jit_local_dilation

#Dilation of f by k
@jax.jit
def dilation_2D(f,index_f,kt):
    """
    Dilation of 2D image.
    -------

    Parameters
    ----------
    f : jax.numpy.array

        A padded binary image

    index_f : jax.numpy.array

        Array with the indexes of f

    kt : jax.numpy.array

        The transpose of the structuring element

    Returns
    -------

    a JAX numpy array

    """
    l = math.floor(kt.shape[0]/2)
    jit_local_dilation = local_dilation(f,kt,l)
    return jnp.apply_along_axis(jit_local_dilation,1,l + index_f).reshape((f.shape[0] - 2*l,f.shape[1] - 2*l))

#Dilation in batches
@jax.jit
def dilation(f,index_f,k):
    """
    Dilation of batches of images.
    -------

    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the binary images

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    Returns
    -------

    a JAX numpy array

    """
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    k = transpose_se(k)
    db = jax.vmap(lambda f: dilation_2D(f,index_f,k),in_axes = (0),out_axes = 0)(f)
    return db

#Opening of f by k
@jax.jit
def opening(f,index_f,k):
    """
    Opening of batches of images.
    -------

    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the binary images

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    Returns
    -------

    a JAX numpy array

    """
    return dilation(erosion(f,index_f,k),index_f,k)

#Colosing of f by k
@jax.jit
def closing(f,index_f,k):
    """
    Closing of batches of images.
    -------

    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the binary images

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    Returns
    -------

    a JAX numpy array

    """
    return erosion(dilation(f,index_f,k),index_f,k)

#Alternate-sequential filter of f by k
@jax.jit
def asf(f,index_f,k):
    """
    Alternate-sequential filter applied to batches of images.
    -------

    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the binary images

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    Returns
    -------

    a JAX numpy array

    """
    return closing(opening(f,index_f,k),index_f,k)

#Complement
@jax.jit
def complement(f):
    """
    Complement batches of images.
    -------

    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the binary images

    Returns
    -------

    a JAX numpy array

    """
    return 1 - f

#Sup-generating with interval [k1,k2]
@jax.jit
def supgen(f,index_f,k1,k2):
    """
    Sup-generating operator applied to batches of images.
    -------

    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the binary images

    index_f : jax.numpy.array

        Array with the indexes of f

    k1,k2 : jax.numpy.array

        Limits of interval [k1,k2]

    Returns
    -------

    a JAX numpy array

    """
    return jnp.minimum(erosion(f,index_f,k1),complement(dilation(f,index_f,complement(transpose_se(k2)))))

#Inf-generating with interval [k1,k2]
@jax.jit
def infgen(f,index_f,k1,k2):
    """
    Inf-generating operator applied to batches of images.
    -------

    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the binary images

    index_f : jax.numpy.array

        Array with the indexes of f

    k1,k2 : jax.numpy.array

        Limits of interval [k1,k2]

    Returns
    -------

    a JAX numpy array

    """
    return jnp.maximum(dilation(f,index_f,k1),complement(erosion(f,index_f,complement(transpose_se(k2)))))

#Sup of array of images
@jax.jit
def sup(f):
    """
    Supremum of batches of images.
    -------

    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the binary images

    Returns
    -------

    a JAX numpy array

    """
    fs = jnp.apply_along_axis(jnp.max,0,f)
    return fs.reshape((1,f.shape[1],f.shape[2]))

#Sup vmap for arch
vmap_sup = lambda f: jax.jit(jax.vmap(lambda f: sup(f),in_axes = (1),out_axes = 1))(f)

#Inf of array of images
@jax.jit
def inf(f):
    """
    Infimum of batches of images.
    -------

    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the binary images

    Returns
    -------

    a JAX numpy array

    """
    fi = jnp.apply_along_axis(jnp.min,0,f)
    return fi.reshape((1,f.shape[1],f.shape[2]))

#Inf vmap for arch
vmap_inf = lambda f: jax.jit(jax.vmap(lambda f: inf(f),in_axes = (1),out_axes = 1))(f)

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
        oper = lambda x,index_x,k: erosion(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])))
    elif type == 'dilation':
        oper = lambda x,index_x,k: dilation(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])))
    elif type == 'opening':
        oper = lambda x,index_x,k: opening(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])))
    elif type == 'closing':
        oper = lambda x,index_x,k: closing(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])))
    elif type == 'asf':
        oper = lambda x,index_x,k: asf(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])))
    elif type == 'supgen':
        oper = lambda x,index_x,k: supgen(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),jax.lax.slice_in_dim(k,1,2).reshape((k.shape[1],k.shape[2])))
    elif type == 'infgen':
        oper = lambda x,index_x,k: infgen(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),jax.lax.slice_in_dim(k,1,2).reshape((k.shape[1],k.shape[2])))
    else:
        print('Type of layer ' + type + 'is wrong!')
        return 1
    return jax.jit(oper)

####Discrete Morphological Neural Networks####
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
    return jnp.mean((true - pred) ** 2)

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
    return 1 - (jnp.sum(jnp.minimum(pred,true))/jnp.sum(jnp.maximum(pred,true)))

#Apply a morphological layer
def apply_morph_layer(x,type,params,index_x):
    """
    Apply a morphological layer.
    ----------

    Parameters
    ----------
    x : jax.numpy.array

        A JAX array with binary images

    type : str

        Names of the operator to apply

    params : jax.numpy.array

        Parameters of the operator

    index_x : jax.numpy.array

        Array with the indexes of f

    Returns
    -------
    jax.numpy.array with output of layer
    """
    #Which operator
    oper = operator(type)
    fx = None
    #For each node
    for i in range(params.shape[0]):
        if fx is None:
            fx = oper(x,index_x,params[i,:,:,:]).reshape((1,x.shape[0],x.shape[1],x.shape[2]))
        else:
            fx = jnp.append(fx,oper(x,index_x,params[i,:,:,:]).reshape((1,x.shape[0],x.shape[1],x.shape[2])),0)
    return fx

#Initiliaze Canonical DMNN
def cdmnn(type,width,size,shape_x,sample = False,p1 = 0.5,key = 0):
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

    Returns
    -------
    dictionary with the initial parameters, forward function, width, size and type
    """
    #Indexes of input images
    index_x = index_array(shape_x)

    #Key
    key = jax.random.split(jax.random.PRNGKey(key),3*max(width)*max(size))
    k = 0

    #Initialize parameters
    params = list()
    for i in range(len(width)):
        if type[i] not in ['sup','inf','complement']:
            if sample:
                if type[i] == 'supgen' or type[i] == 'infgen':
                    ll = np.zeros((1,1,size[i],size[i]),dtype = int)
                    ll[0,0,int(np.round(size[i]/2 - 0.1)),int(np.round(size[i]/2 - 0.1))] = 1
                    ll = jnp.array(ll)
                    ul = 1 + jnp.zeros((1,1,size[i],size[i]),dtype = int)
                    for j in range(width[i]):
                        s = jax.random.choice(jax.random.PRNGKey(key[k,0]),2,p = jnp.array([1 - p1,p1]),shape = (1,1,size[i],size[i]))
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
                        s = jax.random.choice(jax.random.PRNGKey(key[k,0]),2,p = jnp.array([1 - p1,p1]),shape = (1,1,size[i],size[i]))
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
            params.append(p)

    #Params as jax array
    max_width = max(width)
    max_size = max(size)
    for i in range(len(params)):
        params[i] = jnp.pad(params[i],((0,max_width - params[i].shape[0]),(0,2 - params[i].shape[1]),(0,max_size - params[i].shape[2]),(0,max_size - params[i].shape[3])),constant_values = -2)
    params = jnp.array(params,dtype = jnp.float32)

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
    @jax.jit
    def forward(x,params):
        x = x.reshape((1,x.shape[0],x.shape[1],x.shape[2]))
        j = 0
        for i in range(len(type)):
            #Apply sup and inf
            if type[i] == 'sup':
                x = vmap_sup(x)
            elif type[i] == 'inf':
                x = vmap_inf(x)
            elif type[i] == 'complement':
                x = 1 - x
            elif type[i] == "supgen" or type[i] == "infgen":
                #Apply other layer
                x = apply_morph_layer(x[0,:,:,:],type[i],params[j,0:width[i],:,0:size[i],0:size[i]].reshape((width[i],2,size[i],size[i])),index_x)
                j = j + 1
            else:
                #Apply other layer
                x = apply_morph_layer(x[0,:,:,:],type[i],params[j,0:width[i],0,0:size[i],0:size[i]].reshape((width[i],1,size[i],size[i])),index_x)
                j = j + 1
        return x[0,:,:,:]

    #Return initial parameters and forward function
    return {'params': params,'forward': forward,'width': width,'size': size,'type': type_num}

#Sample limit
@jax.jit
def get_lim(h,key = 0):
    ch = jnp.zeros((1,))
    ch = jnp.where(h == 0,1,ch)
    ch = jnp.where((h == 2) | (h < 0),0,ch)
    ch = jnp.where((h != 0) & (h != 2) & (h >= 0),jax.random.choice(jax.random.PRNGKey(key),2,shape = (1,)),ch)
    return ch.astype(jnp.int32)

#Visit a nighboor
def visit_neighbor(h,params,x,y,lf):
    """
    Compute the loss at a nighbor.
    ----------

    Parameters
    ----------
    h : jax.numpy.array

        Index of the neighbor to visit

    params : jax.numpy.array

        Array of current parameters

    x,y : jax.numpy.array

        Input and output images

    lf : function

        Loss function

    Returns
    -------
    float
    """
    #Compute error
    return lf(params.at[h[0],h[1],h[2],h[3],h[4]].set(1 - params[h[0],h[1],h[2],h[3],h[4]]),x,y)

#Step SLDA
def step_slda(params,x,y,forward,lf,type,width,size,sample = True,neighbors = 8,key = 0):
    """
    Step of Stochastic Lattice Descent Algorithm updating the parameters.
    ----------

    Parameters
    ----------
    params : jax.numpy.array

        Current parameters

    x,y : jax.numpy.array

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

    sample : logical

        Whether to sample neighbors

    neighbors : int

        Number of neighbors to sample

    key : int

        Key for sampling

    Returns
    -------
    list of jax.numpy.array
    """
    #Key
    key = jax.random.split(jax.random.PRNGKey(key),3*max(width)*max(size)*max(size))

    #Store neighbors to consider: layer + node + lim + row + col
    k = 0
    if sample:
        #Calculate probabilities
        prob = jnp.array([])
        j = 0
        for i in range(len(type)):
            if type[i] != 0:
                #If is not supgen/infgen
                if type[i] == 1:
                    prob  = jnp.append(prob,jnp.array(width[i] * (size[i] ** 2)))
                #If supgen/infgen
                else:
                    count = jnp.apply_along_axis(jnp.sum,1,params[j,0:width[i],:,0:size[i],0:size[i]])
                    prob = jnp.append(prob,jnp.sum(jnp.where((count == 0) | (count == 2),1,2)))
                    del count
                j = j + 1
            #Arrays
            max_width = max(width)
            max_size = max(size)

            #Sample layers
            prob = jnp.array(prob).reshape((len(prob),))
            prob = prob/jnp.sum(prob)
            layers = jax.random.choice(jax.random.PRNGKey(key[k,0]),len(prob),shape = (neighbors,),p = prob)
            k = k + 1
            hood = None
            #For each layer sample a change
            for i in range(neighbors):
                l = layers[i]
                #Sample a node
                par_l = params[l,:,:,:,:]
                count_sum = jnp.apply_along_axis(jnp.sum,1,par_l)
                count = jnp.where(count_sum == 1,2,1)
                count = jnp.where(count_sum == -4,0,count)
                tmp_prob = jax.vmap(jnp.sum)(count)
                tmp_prob = tmp_prob/jnp.sum(tmp_prob)
                node = jax.random.choice(jax.random.PRNGKey(key[k,0]),max_width,shape = (1,1),p = tmp_prob)
                k = k + 1
                #Sample row and collumn
                tmp_prob = count[node[0,0],:,:].reshape((max_size ** 2))
                tmp_prob = tmp_prob/jnp.sum(tmp_prob)
                tmp_random = jax.random.choice(jax.random.PRNGKey(key[k,0]),max_size ** 2,shape = (1,),p = tmp_prob)
                k = k + 1
                rc = jnp.array([jnp.floor(tmp_random/max_size),tmp_random % max_size]).reshape((1,2)).astype(jnp.int32)
                #Sample limit
                lim = get_lim(jnp.sum(par_l[node,:,rc[0,0],rc[0,1]]).reshape((1,1)),key[k,0])
                k = k + 1
                #Neighbor
                nei = jnp.append(jnp.append(jnp.append(l.reshape((1,1)),node,1),lim,1),rc,1).astype(jnp.int32)
                error_tmp = visit_neighbor(nei,params,x,y,lf)
                if hood is None:
                    hood = nei
                    error = error_tmp
                else:
                    hood = jnp.append(hood,nei,0)
                    error = jnp.append(error,error_tmp)
                del count, tmp_prob, tmp_random, par_l, node, rc, lim, error_tmp
    else:
        hood = None
        j = 0
        for i in range(len(type)):
            if type[i] > 0:
                if type[i] == 2:
                    par = params[j,0:width[i],:,0:size[i],0:size[i]].reshape((width[i],2,size[i],size[i]))
                else:
                    par = params[j,0:width[i],0,0:size[i],0:size[i]].reshape((width[i],1,size[i],size[i]))
                dim = par.shape
                if dim[1] == 1:
                    tmp = jnp.array([[j,node,0,row,col] for node in range(dim[0]) for row in range(dim[2]) for col in range(dim[3])])
                else:
                    tmp = jnp.array([[j,node,lim,row,col] for node in range(dim[0]) for lim in [0,1] for row in range(dim[2]) for col in range(dim[3])])
                    for i in range(tmp.shape[0]):
                        obs = jnp.sum(par[tmp[i,1],:,tmp[i,3],tmp[i,4]])
                        v = tmp[i,2]
                        v = jnp.where(obs == 0,1,v)
                        v = jnp.where(obs == 2,0,v)
                        tmp = tmp.at[i,2].set(v)
                    del v
                error_tmp = visit_neighbor(tmp,params,x,y,lf)
                if hood is None:
                    hood = tmp
                    error = error_tmp
                else:
                    hood = jnp.append(hood,tmp,0)
                    error = jnp.append(error,error_tmp)
                j = j + 1
                del tmp, error_tmp

    #Shuffle hood
    hood = jax.random.permutation(jax.random.PRNGKey(key[k,0]),hood,0)
    k = k + 1

    #Compute error for each point in the hood
    #res = jax.vmap(lambda h: visit_neighbor(h,params,x,y,lf))(hood).reshape((hood.shape[0],1))

    return hood,error #jnp.append(hood,error.reshape((hood.shape[0],1)),1)

#Threshold gray scale image
@jax.jit
def threshold(X,t):
    return jnp.where(X >= t,1.0,0.0)

#Training function DMNN via SLDA
def train_dmnn_slda(x,y,net,loss,xval = None,yval = None,stack = False,sample = False,neighbors = 8,epochs = 1,batches = 1,K = 255,notebook = False,epoch_print = 100,epoch_store = 1,key = 0,store_jumps = False,error_type = 'mean'):
    """
    Stochastic Lattice Descent Algorithm to train Discrete Morphological Neural Networks for stack operators.
    ----------

    Parameters
    ----------
    x,y : jax.numpy.array

        Input and output images

    net : dict

        Dictionary returned by the function cdmnn

    loss : function

        Loss function

    xval,yval : jax.numpy.array

        Input and output validation images

    stack : logical

        Whether to train a stack operator

    sample : logical

        Whether to sample neighbors

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
    list of jax.numpy.array
    """
    #Parameters
    params = net['params']
    forward = net['forward']
    type = net['type']
    width = net['width']
    size = net['size']
    stacks = 1 + jnp.arange(K)

    #Stack data
    if stack:
        x = jax.vmap(lambda t: threshold(x,t))(stacks)
        forward = jax.jit(lambda x,params: jnp.sum(jax.vmap(lambda x: forward(x,params))(x),0))

    #Key
    key = jax.random.split(jax.random.PRNGKey(key),epochs)

    #Batch size
    bsize = int(math.floor(x.shape[0]/batches))

    #Loss function
    if error_type == 'mean':
        @jax.jit
        def lf(params,x,y):
            return jnp.mean(jax.vmap(loss)(forward(x,params),y))
    else:
        @jax.jit
        def lf(params,x,y):
            return jnp.max(jax.vmap(loss)(forward(x,params),y))

    #Training function
    @jax.jit
    def update(params,x,y,key,jumps):
        hood,error = step_slda(params,x,y,forward,lf,type,width,size,sample,neighbors,key)
        hood = hood[jnp.argsort(error),:]
        error = error[jnp.argsort(error)]
        train_loss = error[0]
        if store_jumps:
          jumps = jnp.append(jumps,hood,0)
        new_params = params.at[hood[0],hood[1],hood[2],hood[3],hood[4]].set(1 - params[hood[0],hood[1],hood[2],hood[3],hood[4]])
        return new_params,jumps,train_loss

    #Trace
    best_par = params.copy()
    min_loss = lf(params,x,y)
    trace_time = [0]
    trace_loss = [min_loss]
    trace_epoch = [0]
    if xval is not None:
        xval = jax.vmap(lambda t: threshold(xval,t))(stacks)
        min_val_loss = lf(params,xval,yval)
        trace_val_loss = [min_val_loss]
    else:
        min_val_loss = jnp.inf
        trace_val_loss = []
    params_init = params.copy()
    jumps = jnp.array([])

    #Step epoch
    @jax.jit
    def step_epoch(e,params,best_par,min_loss,jumps,x,y,xval,yval):
        if batches > 1:
            per = jax.random.permutation(jax.random.PRNGKey(key[e,0]),jnp.arange(x.shape[2]))
            x = x[:,per,:,:]
            y = y[per,:,:]
        for b in range(batches):
            if batches > 1:
                if b < batches - 1:
                    xb = jax.lax.dynamic_slice(x,(0,b*bsize,0,0),(x.shape[0],bsize,x.shape[2],x.shape[3]))
                    yb = jax.lax.dynamic_slice(y,(b*bsize,0,0),(bsize,y.shape[1],y.shape[2]))
                else:
                    xb = x[:,b*bsize:x.shape[1],:,:]
                    yb = y[b*bsize:y.shape[0],:,:]
                #Search neighbors
                params,jumps,train_loss = update(params,x,y,key[e,1],jumps)
            else:
                #Search neighbors
                params,jumps,train_loss = update(params,x,y,key[e,1],jumps)
        #Compute loss and store at the end of epoch
        if batches > 1:
            train_loss = lf(params,x,y)
        #Update best
        best_par = (train_loss < min_loss) * params + (train_loss >= min_loss) * best_par
        min_loss = jnp.minimum(train_loss,min_loss)
        if xval is not None:
            min_val_loss = lf(best_par,xval,yval)
        return params,best_par,min_loss,jumps,train_loss

    #Train
    t0 = time.time()
    with alive_bar(epochs) as bar:
        bar.title('Epoch: ' + str(0) + " Loss: " + str(jnp.round(min_loss,5)) + ' Best: ' + str(jnp.round(min_loss,5)) + ' Val: ' + str(jnp.round(min_val_loss,5)))
        for e in range(epochs):
            #Take step
            params,best_par,min_loss,jumps,train_loss = step_epoch(key[e,1],params,best_par,min_loss,jumps,x,y,xval,yval)
            #Store
            if (e + 1) % epoch_store == 0:
                trace_epoch = trace_epoch + [e + 1]
                trace_time = trace_time + [time.time() - t0]
                trace_loss = trace_loss + [train_loss]
                if xval is not None:
                    val_loss = lf(params,xval,yval)
                    trace_val_loss = trace_val_loss + [val_loss]
            bar.title('Epoch: ' + str(e) + " Time: " + str(jnp.round(time.time() - t0)) + " s Loss: " + str(jnp.round(train_loss,5)) + ' Best: ' + str(jnp.round(min_loss,5)) + ' Val: ' + str(jnp.round(min_val_loss,5)))
            if e % epoch_print == 0:
                if notebook:
                    print('Epoch: ' + str(e) + " Time: " + str(jnp.round(time.time() - t0)) + " s Loss: " + str(jnp.round(train_loss,5)) + ' Best: ' + str(jnp.round(min_loss,5)) + ' Val: ' + str(jnp.round(min_val_loss,5)))
            bar()

    return {'best_par': best_par,'jumps': jumps,'trace_epoch': trace_epoch,'trace_time': trace_time,'trace_loss': trace_loss,'trace_val_loss': trace_val_loss,'epochs': epochs,'oper': lambda x: jnp.sum(jax.vmap(lambda x: forward(x,params))(jax.vmap(lambda t: threshold(x,t))(stacks)),0),'forward': forward}
