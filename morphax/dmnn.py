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
    f = jax.lax.pad(f,0,((0,0,0),(l,l,0),(l,l,0)))
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

    k : jax.numpy.array

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
    f = jax.lax.pad(f,0,((0,0,0),(l,l,0),(l,l,0)))
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
    mean square error
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
    mean square error
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
def cdmnn(type,width,size,shape_x,sample = False,p1 = 0.5):
    """
    Initialize a Discrete Morphological Neural Network as the identity operator.
    ----------

    Parameters
    ----------
    types : list of str

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

    Returns
    -------
    dictionary with the initial parameters, forward function, width, size and type
    """
    #Indexes of input images
    index_x = index_array(shape_x)

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
                        s = np.random.choice([0,1],p = [1 - p1,p1],size = (1,1,size[i],size[i]))
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
                        s = np.random.choice([0,1],p = [1 - p1,p1],size = (1,1,size[i],size[i]))
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
        params[i] = jnp.pad(params[i],((0,max_width - params[i].shape[0]),(0,2 - params[i].shape[1]),(0,max_size - params[i].shape[2]),(0,max_size - params[i].shape[3])))
    params = jnp.array(params)

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
    return {'params': params,'forward': forward,'width': width,'size': size,'type': type}

#Visit a nighboor
@jax.jit
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
def step_slda(params,x,y,forward,lf,type,width,size,sample = True,neighbors = 8):
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

    Returns
    -------
    list of jax.numpy.array
    """
    #Store neighbors to consider: layer + node + lim + row + col
    if sample:
        #Calculate probabilities
        prob = []
        type_j = []
        width_j = []
        size_j = []
        j = 0
        for i in range(len(type)):
            if type[i] not in ['sup','inf','complement']:
                type_j = type_j + [type[i]]
                width_j = width_j + [width[i]]
                size_j = size_j + [size[i]]
                #If is not supgen/infgen
                if type[i] not in ['supgen','infgen'] == 1:
                    prob  = prob + [width[i] * (size[i] ** 2)]
                #If supgen/infgen
                else:
                    count = jnp.apply_along_axis(jnp.sum,1,params[j,0:width[i],:,0:size[i],0:size[i]])
                    prob = prob + [jnp.sum(jnp.where((count == 0) | (count == 2),1,2))]
                j = j + 1
        #Sample layers
        prob = jnp.array(prob).reshape((len(prob),))
        prob = prob/jnp.sum(prob)
        r = jax.random.choice(jax.random.PRNGKey(np.random.choice(range(1000000))),len(prob),shape = (neighbors,),p = prob)
        hood = np.array([[l,0,0,0,0] for l in r]).reshape((neighbors,5))
        #For each layer sample a change
        for i in range(hood.shape[0]):
            l = hood[i,0]
            #If is not supgen/infgen
            if type_j[l] not in ['supgen','infgen']:
                #Sample a node
                hood[i,1] = np.random.choice(width_j[l])
                #Sample row and collumn
                hood[i,3:5] = np.random.choice(size_j[l],size = 2)
            else:
                #Sample a node
                par_l = params[l,0:width_j[l],:,0:size_j[l],0:size_j[l]]
                count = jnp.apply_along_axis(jnp.sum,1,par_l)
                count = jnp.where((count == 0) | (count == 2),1,2)
                tmp_prob = jax.vmap(jnp.sum)(count)
                tmp_prob = [x/sum(tmp_prob) for x in tmp_prob]
                hood[i,1] = np.random.choice(width_j[l],p = tmp_prob)
                #Sample row and collumn
                tmp_prob = count[hood[i,1],:,:].reshape((size_j[l] ** 2))
                tmp_prob = [x/sum(tmp_prob) for x in tmp_prob]
                tmp_random = np.random.choice(size_j[l] ** 2,p = tmp_prob)
                hood[i,3:5] = [int(np.floor(tmp_random/size_j[l])),tmp_random % size_j[l]]
                #Sample limit
                obs = par_l[hood[i,1],:,hood[i,3],hood[i,4]]
                if jnp.sum(obs) == 0:
                    hood[i,2] = 1
                elif jnp.sum(obs) == 2:
                    hood[i,2] = 0
                else:
                    hood[i,2] = np.random.choice(2,p = [0.5,0.5])
                del count, tmp_prob, tmp_random, obs, par_l
        hood = jnp.array(hood)
    else:
        hood = None
        j = 0
        for i in range(len(type)):
            if type[i] not in ['sup','inf','complement']:
                if type[i] in ['infgen','supgen']:
                    par = params[j,0:width[i],:,0:size[i],0:size[i]].reshape((width[i],2,size[i],size[i]))
                else:
                    par = params[j,0:width[i],0,0:size[i],0:size[i]].reshape((width[i],1,size[i],size[i]))
                dim = par.shape
                if dim[1] == 1:
                    tmp = np.array([[j,node,0,row,col] for node in range(dim[0]) for row in range(dim[2]) for col in range(dim[3])])
                else:
                    tmp = np.array([[j,node,lim,row,col] for node in range(dim[0]) for lim in [0,1] for row in range(dim[2]) for col in range(dim[3])])
                    for i in range(tmp.shape[0]):
                        obs = par[tmp[i,1],:,tmp[i,3],tmp[i,4]]
                        if jnp.sum(obs) == 0:
                            tmp[i,2] = 1
                        elif jnp.sum(obs) == 2:
                            tmp[i,2] = 0
                    tmp = jnp.unique(tmp,axis = 0)
                if hood is None:
                    hood = tmp
                else:
                    hood = jnp.append(hood,tmp,0)
                j = j + 1
                del tmp

    #Shuffle hood
    hood = jax.random.permutation(jax.random.PRNGKey(np.random.choice(range(1000000))),hood,0)

    #Compute error for each point in the hood
    res = jax.vmap(lambda h: visit_neighbor(h,params,x,y,lf))(hood)

    #Minimum
    hood = hood[res == jnp.min(res),:][0,:]

    #Return
    return params.at[hood[0],hood[1],hood[2],hood[3],hood[4]].set(1 - params[hood[0],hood[1],hood[2],hood[3],hood[4]])

#Training function MNN
def train_dmnn(x,y,net,loss,sample = False,neighbors = 8,epochs = 1,batches = 1,notebook = False,epoch_print = 100):
    """
    Stochastic Lattice Descent Algorithm to train Discrete Morphological Neural Networks.
    ----------

    Parameters
    ----------
    x,y : jax.numpy.array

        Input and output images

    net : dict

        Dictionary returned by the function cdmnn

    loss : function

        Loss function

    sample : logical

        Whether to sample neighbors

    neighbors : int

        Number of neighbors to sample

    epochs : int

        Number of bacthes

    notebook : logical

        Wheter the code is being run in a notebook. Has an effect on tracing the Algorithm

    epoch_print : int

        Number of epochs to print the partial result

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

    #Key
    key = jax.random.split(jax.random.PRNGKey(0),epochs)

    #Batch size
    bsize = int(math.floor(x.shape[0]/batches))

    #Loss function
    @jax.jit
    def lf(params,x,y):
        return jnp.mean(jax.vmap(loss)(forward(x,params),y))

    #Training function
    @jax.jit
    def update(params,x,y):
      params = step_slda(params,x,y,forward,lf,type,width,size,sample,neighbors)
      return params

    #Train
    min_error = jnp.inf
    xy = jnp.append(x.reshape((1,x.shape[0],x.shape[1],x.shape[2])),y.reshape((1,x.shape[0],x.shape[1],x.shape[2])),0)
    t0 = time.time()
    with alive_bar(epochs) as bar:
        bar.title("Loss: 1.00000 Best: 1.00000")
        for e in range(epochs):
            #Permutate xy
            xy = jax.random.permutation(jax.random.PRNGKey(key[e,0]),xy,1)
            for b in range(batches):
                if b < batches - 1:
                    xb = jax.lax.dynamic_slice(xy[0,:,:,:],(b*bsize,0,0),(bsize,x.shape[1],x.shape[2]))
                    yb = jax.lax.dynamic_slice(xy[1,:,:,:],(b*bsize,0,0),(bsize,x.shape[1],x.shape[2]))
                else:
                    xb = xy[0,b*bsize:x.shape[0],:,:]
                    yb = xy[1,b*bsize:y.shape[0],:,:]
                params = update(params,xb,yb)
            l = lf(params,x,y)
            bar.title("Loss: " + str(jnp.round(l,5)) + ' Best: ' + str(jnp.round(min_error,5)))
            if l < min_error:
                min_error = l
                best_par = params.copy()
            if e % epoch_print == 0:
                if notebook:
                    print('Epoch: ' + str(e) + ' Time: ' + str(jnp.round(time.time() - t0,2)) + ' s Loss: ' + l)
            if not notebook:
                bar()

    return best_par


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
