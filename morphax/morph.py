#Continuous morphology module
import jax
import jax.numpy as jnp
import math
import sys
from morphax import dmorph_jax as dmnn

#Structuring element from function
def struct_function(k,d):
    """
    Structuring element from two-dimensional function.
    -------
    Parameters
    ----------
    k : function

        A function that receives a coordinate in {-d,...,d}^2 and returns the value of the structuring element in it

    d : int

        Dimesion of window

    Returns
    -------
    a JAX numpy array
    """
    w = jnp.array([[x1.tolist(),x2.tolist()] for x1 in jnp.linspace(-jnp.floor(d/2),jnp.floor(d/2),d) for x2 in jnp.linspace(jnp.floor(d/2),-jnp.floor(d/2),d)])
    k = jnp.array(k(w))
    return jnp.transpose(k.reshape((d,d)))

#Create function to apply W-operator with characteristic function f at pixel (i,j)
def local_w_operator(x,f,l):
    """
    Define function for applying W-operator.
    -------
    Parameters
    ----------
    x : jax.numpy.array

        An image

    f : function

        The characteristic function of the W-operator

    l : int

        Length of structruring element. If k has shape d x d, then l is the greatest integer such that l <= d/2

    Returns
    -------
    a function that receives an index and returns the local value of f at this index
    """
    def jit_w_operator(index):
        return f(jax.lax.dynamic_slice(x, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1)))
    return jit_w_operator

#Apply W-operator with characteristic function f
def w_operator_2D(x,index_x,f,d):
    """
    Apply a W-operator to 2D image.
    -------
    Parameters
    ----------
    x : jax.numpy.array

        A padded image

    index_x : jax.numpy.array

        Array with the indexes of x

    f : function

        The characteristic function of the W-operator

    d : int

        Dimesion of window

    Returns
    -------
    jax.numpy.array
    """
    l = math.floor(d/2)
    jit_w_operator = local_w_operator(x,f,l)
    return jnp.apply_along_axis(jit_w_operator,1,l + index_x).reshape((x.shape[0] - 2*l,x.shape[1] - 2*l))

#Apply W-operator in batches
def w_operator(x,index_x,f,d):
    """
    Apply a W-operator to a batch of images.
    -------
    Parameters
    ----------
    x : jax.numpy.array

        A 3D array with the images

    index_x : jax.numpy.array

        Array with the indexes of x

    f : function

        The characteristic function of the W-operator

    d : int

        Dimesion of window

    Returns
    -------
    jax.numpy.array
    """
    l = math.floor(d/2)
    x = jax.lax.pad(x,0.0,((0,0,0),(l,l,0),(l,l,0)))
    wop = jax.vmap(lambda x: w_operator_2D(x,index_x,f,d),in_axes = (0),out_axes = 0)(x)
    return wop

#Apply W-operator with characteristic function f given by nn at pixel (i,j)
def local_w_operator_nn(x,forward,params,l):
    """
    Define function for applying W-operator defined from a neural network.
    -------
    Parameters
    ----------
    x : jax.numpy.array

        An image

    forward : function

        The forward function of a neural network

    params : jax.numpy.array

        Parameters of a neural network

    l : int

        Length of structruring element. If k has shape d x d, then l is the greatest integer such that l <= d/2

    Returns
    -------
    a function that receives an index and returns the local value of the W-operator at this index
    """
    def jit_w_operator(index):
        return forward(jax.lax.dynamic_slice(x, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1)).reshape((1,(2*l + 1) ** 2)),params)
    return jit_w_operator

#Apply W-operator with characteristic function f given by nn
def w_operator_2D_nn(x,index_x,forward,params,d):
    """
    Apply W-operator defined from a neural network to 2D image.
    -------
    Parameters
    ----------
    x : jax.numpy.array

        A padded image

    index_x : jax.numpy.array

        Array with the indexes of x

    forward : function

        The forward function of a neural network

    params : jax.numpy.array

        Parameters of a neural network

    d : int

        Dimesion of window

    Returns
    -------
    jax.numpy.array
    """
    l = math.floor(d/2)
    jit_w_operator = local_w_operator_nn(x,forward,params,l)
    return jnp.apply_along_axis(jit_w_operator,1,l + index_x).reshape((x.shape[0] - 2*l,x.shape[1] - 2*l))

#Apply W-operator in batches (nn)
def w_operator_nn(x,index_x,forward,params,d):
    """
    Apply a W-operator defined from a neural network to a batch of images.
    -------

    Parameters
    ----------
    x : jax.numpy.array

        A 3D array with the images

    index_x : jax.numpy.array

        Array with the indexes of x

    forward : function

        The cforward function of a neural network

    params : jax.numpy.array

        Parameters of a neural network

    d : int

        Dimesion of window

    Returns
    -------
    jax.numpy.array
    """
    l = math.floor(d/2)
    x = jax.lax.pad(x,0.0,((0,0,0),(l,l,0),(l,l,0)))
    wop = jax.vmap(lambda x: w_operator_2D_nn(x,index_x,forward,params,d),in_axes = (0),out_axes = 0)(x)
    return wop

def truncate(f,m,M):
    """
    Truncate an image to the interval [m,M]
    -------
    Parameters
    ----------
    f : jax.numpy.array

        An image

    m,M : float

        Minimum and maximum value of the interval to truncate

    Returns
    -------
    jax.numpy.array
    """
    return jnp.maximum(jnp.minimum(f,m),M)

#Local erosion of f by k for pixel (i,j)
def local_erosion(f,k,l):
    """
    Define function for local erosion.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        An image

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
        return jnp.min(fw - k)
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

        A padded image

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    Returns
    -------
    jax.numpy.array
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

        A 3D array with the images

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    Returns
    -------
    jax.numpy.array
    """
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    eb = jax.vmap(lambda f: erosion_2D(f,index_f,k),in_axes = (0),out_axes = 0)(f)
    return eb

#Local dilation of f by k for pixel (i,j) assuming k already transposed
def local_dilation(f,kt,l):
    """
    Define function for local dilation.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        An image

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
        return jnp.max(fw + kt)
    return jit_local_dilation

#Dilation of f by k assuming k already transposed
@jax.jit
def dilation_2D(f,index_f,kt):
    """
    Dilation of 2D image.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        A padded image

    index_f : jax.numpy.array

        Array with the indexes of f

    kt : jax.numpy.array

        The transpose of the structuring element

    Returns
    -------
    jax.numpy.array
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

        A 3D array with the images

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    Returns
    -------
    jax.numpy.array
    """
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    k = dmnn.transpose_se(k)
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

        A 3D array with the images

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    Returns
    -------
    jax.numpy.array
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

        A 3D array with the images

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    Returns
    -------
    jax.numpy.array
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

        A 3D array with the images

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    Returns
    -------
    jax.numpy.array
    """
    return closing(opening(f,index_f,k),index_f,k)

#Complement
@jax.jit
def complement(f,m = 1):
    """
    Complement batches of images.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the images

    m : int

        Maximum value

    Returns
    -------
    jax.numpy.array
    """
    return m - f

############NEED REVIEW##############
#Sup-generating with interval [k1,k2]
@jax.jit
def supgen(f,index_f,k1,k2,m = 1):
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

    m : int

        Maximum value

    Returns
    -------

    a JAX numpy array

    """
    return  jnp.minimum(erosion(f,index_f,k1),complement(dilation(f,index_f,complement(dmnn.transpose_se(k2 + m),m) - m),m))

#Inf-generating with interval [k1,k2]
def infgen(f,index_f,k1,k2,m = 1):
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

    m : int

        Maximum value

    Returns
    -------

    a JAX numpy array

    """
    return jnp.maximum(dilation(f,index_f,k1),complement(erosion(f,index_f,complement(dmnn.transpose_se(k2 + m),m) - m),m))

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
        oper = lambda x,index_x,k: ap_supgen(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),jax.lax.slice_in_dim(k,1,2).reshape((k.shape[1],k.shape[2])))
    elif type == 'infgen':
        oper = lambda x,index_x,k: infgen(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),jax.lax.slice_in_dim(k,1,2).reshape((k.shape[1],k.shape[2])))
    else:
        print('Type of layer ' + type + 'is wrong!')
        return 1
    return jax.jit(oper)

#Approximate supgen
#@jax.jit
#def actsupgen(x):
#    return jnp.where(x < 0,jax.nn.relu(x + 1),1.0)

#def local_supgen(f,k1,k2,l):
#    def jit_local_supgen(index):
#        fw = jax.lax.dynamic_slice(f, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
#        return actsupgen(jnp.min(jnp.minimum(fw - k1,k2 - fw)))
#    return jit_local_supgen

#@jax.jit
#def supgen_2D(f,index_f,k1,k2):
#    l = math.floor(k1.shape[0]/2)
#    jit_local_supgen = local_supgen(f,k1,k2,l)
#    return jnp.apply_along_axis(jit_local_supgen,1,l + index_f).reshape((f.shape[0] - 2*l,f.shape[1] - 2*l))

#@jax.jit
#def ap_supgen(f,index_f,k1,k2):
#    l = math.floor(k1.shape[0]/2)
#    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
#    k2 = k1 + k2 ** 2
#    k2 = jnp.where(k2 > 0,0,k2)
#    sg = jax.vmap(lambda f: supgen_2D(f,index_f,k1 + 1,k2 + 1),in_axes = (0),out_axes = 0)(f)
#    return sg


#Structuring element of the approximate identity operator in a sample
#def struct_lower(x,d):
#    #Function to apply to each index
#    l = math.floor(d/2)
#    x = jax.lax.pad(x,0.0,((0,0,0),(l,l,0),(l,l,0)))
#    index_x = index_array((x.shape[1],x.shape[2]))
#    def struct_lower(index,x):
#        fw = jax.lax.dynamic_slice(x, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
#        return fw - x[index[0],index[1]]
#    k = jax.vmap(lambda x: jnp.apply_along_axis(lambda index: struct_lower(index,x),1,index_x))(x).reshape((x.shape[0],x.shape[1],x.shape[2],d,d))
#    k = k.reshape((k.shape[0]*k.shape[1]*k.shape[2],d,d))
#    k = jnp.apply_along_axis(lambda k: jnp.min(k),0,k)
#    return k

#Structuring element of upper limit of interval of supgen approximating identity operator
#def struct_upper(x,d):
#    #Function to apply to each index
#    l = math.floor(d/2)
#    x = jax.lax.pad(x,0.0,((0,0,0),(l,l,0),(l,l,0)))
#    index_x = index_array((x.shape[1],x.shape[2]))
#    def struct_upper(index,x):
#        fw = jax.lax.dynamic_slice(x, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
#        return fw + x[index[0],index[1]]
#    k = jax.vmap(lambda x: jnp.apply_along_axis(lambda index: struct_upper(index,x),1,index_x))(x).reshape((x.shape[0],x.shape[1],x.shape[2],d,d))
#    k = k.reshape((k.shape[0]*k.shape[1]*k.shape[2],d,d))
#    k = jnp.apply_along_axis(lambda k: jnp.max(k),0,k)
#    return k

#Approximate maximum
#def max(x,h = 1/5):
#    return h * jnp.log(jnp.sum(jnp.exp(x/h)))

#def maximum(x,y,h = 1/50):
#    if len(x.shape) == 2:
#        x = x.reshape((1,x.shape[0],x.shape[1]))
#        y = y.reshape((1,y.shape[0],y.shape[1]))
#    return jax.vmap(jax.vmap(jax.vmap(lambda x,y: h * jnp.log(jnp.sum(jnp.exp(jnp.append(x,y)/h))))))(x,y)

#def maximum_array_number(arr,x,h = 1/5):
#    return h * jnp.log(jnp.exp(arr/h) + jnp.exp(x/h))

#Approximate minimum
#def min(x,h = 1/5):
#    return max(x,-h)

#def minimum(x,y,h = 1/5):
#    return maximum(x,y,-h)

#def minimum_array_number(arr,x,h = 1/5):
#    return maximum_array_number(arr,x,-h)

#def struct_function_w(k,w,d):
#    k = jnp.array(k(w))
#    return jnp.transpose(k.reshape((d,d)))
