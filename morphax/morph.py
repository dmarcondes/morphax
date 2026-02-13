#Continuous morphology module
import jax
import jax.numpy as jnp
import math
import sys
from morphax import dmorph_jax as dmnn

#Smooth max/min
def smoothExt(x,alpha):
    """
    Smooth minimum (alpha < 0) and maximum (alpha > 0)
    -------
    Parameters
    ----------
    x : jax.numpy.array

        Array to take the smooth minimum or maximum

    alpha : float

        Smoothing parameter

    Returns
    -------
    float
    """
    return jnp.sum(x * jnp.exp(alpha * x))/jnp.sum(jnp.exp(alpha * x))

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

#Smooth local erosion of f by k for pixel (i,j)
def Slocal_erosion(f,k,l,alpha = 5):
    """
    Define function for smooth local erosion.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        An image

    k : jax.numpy.array

        A structuring element

    l : int

        Length of structruring element. If k has shape d x d, then l is the greatest integer such that l <= d/2

    alpha : float

        Smoothing parameter

    Returns
    -------
    a function that receives an index and returns the smooth local erosion of f by k at this index
    """
    def jit_local_erosion(index):
        fw = jax.lax.dynamic_slice(f, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
        return smoothExt(fw - k,(-1)*alpha)
    return jit_local_erosion

#Smooth erosion of f by k
@jax.jit
def Serosion_2D(f,index_f,k,alpha = 5):
    """
    Smooth erosion of 2D image.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        A padded image

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    alpha : float

        Smoothing parameter

    Returns
    -------
    jax.numpy.array
    """
    l = math.floor(k.shape[0]/2)
    jit_local_erosion = Slocal_erosion(f,k,l,alpha)
    return jnp.apply_along_axis(jit_local_erosion,1,l + index_f).reshape((f.shape[0] - 2*l,f.shape[1] - 2*l))

#Smooth erosion in batches
@jax.jit
def Serosion(f,index_f,k,alpha = 5):
    """
    Smooth erosion of batches of images.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the images

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    alpha : float

        Smoothing parameter

    Returns
    -------
    jax.numpy.array
    """
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    eb = jax.vmap(lambda f: Serosion_2D(f,index_f,k,alpha),in_axes = (0),out_axes = 0)(f)
    return eb

#Local anti-dilation of f by k for pixel (i,j)
def local_anti_dilation(f,k,l):
    """
    Define function for local anti-dilation.
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
    a function that receives an index and returns the local anti-dilation of f by k at this index
    """
    def jit_local_anti_dilation(index):
        fw = jax.lax.dynamic_slice(f, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
        return jnp.min(k - fw)
    return jit_local_anti_dilation

#Anti-dilation of f by k
@jax.jit
def anti_dilation_2D(f,index_f,k):
    """
    Anti-dilation of 2D image.
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
    jit_local_anti_dilation = local_anti_dilation(f,k,l)
    return jnp.apply_along_axis(jit_local_anti_dilation,1,l + index_f).reshape((f.shape[0] - 2*l,f.shape[1] - 2*l))

#Anti-dilation in batches
@jax.jit
def anti_dilation(f,index_f,k):
    """
    Anti-dilation of batches of images.
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
    ad = jax.vmap(lambda f: anti_dilation_2D(f,index_f,k),in_axes = (0),out_axes = 0)(f)
    return ad

#Local smoth anti-dilation of f by k for pixel (i,j)
def Slocal_anti_dilation(f,k,l,alpha = 5):
    """
    Define function for smooth local anti_dilation.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        An image

    k : jax.numpy.array

        A structuring element

    l : int

        Length of structruring element. If k has shape d x d, then l is the greatest integer such that l <= d/2

    alpha : float

        Smoothing parameter

    Returns
    -------
    a function that receives an index and returns the smooth local anti-dilation of f by k at this index
    """
    def jit_local_anti_dilation(index):
        fw = jax.lax.dynamic_slice(f, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
        return smoothExt(k - fw,(-1)*alpha)
    return jit_local_anti_dilation

#Smooth anti-dilation of f by k
@jax.jit
def Santi_dilation_2D(f,index_f,k,alpha = 5):
    """
    Smooth anti-dilation of 2D image.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        A padded image

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    alpha : float

        Smoothing parameter

    Returns
    -------
    jax.numpy.array
    """
    l = math.floor(k.shape[0]/2)
    jit_local_anti_dilation = Slocal_anti_dilation(f,k,l,alpha)
    return jnp.apply_along_axis(jit_local_anti_dilation,1,l + index_f).reshape((f.shape[0] - 2*l,f.shape[1] - 2*l))

#Smooth anti-dilation in batches
@jax.jit
def Santi_dilation(f,index_f,k,alpha = 5):
    """
    Smooth anti-dilation of batches of images.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the images

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    alpha : float

        Smoothing parameter

    Returns
    -------
    jax.numpy.array
    """
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    ad = jax.vmap(lambda f: Santi_dilation_2D(f,index_f,k,alpha),in_axes = (0),out_axes = 0)(f)
    return ad

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

#Local smooth dilation of f by k for pixel (i,j) assuming k already transposed
def Slocal_dilation(f,kt,l,alpha = 5):
    """
    Define function for smooth local dilation.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        An image

    kt : jax.numpy.array

        The transpose of the structuring element

    l : int

        Length of structruring element. If k has shape d x d, then l is the greatest integer such that l <= d/2

    alpha : float

        Smoothing parameter

    Returns
    -------
    a function that receives an index and returns the smooth local dilation of f by k at this index
    """
    def jit_local_dilation(index):
        fw = jax.lax.dynamic_slice(f, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
        return smoothExt(fw + kt,alpha)
    return jit_local_dilation

#Smooth dilation of f by k assuming k already transposed
@jax.jit
def Sdilation_2D(f,index_f,kt,alpha = 5):
    """
    Smoothing dilation of 2D image.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        A padded image

    index_f : jax.numpy.array

        Array with the indexes of f

    kt : jax.numpy.array

        The transpose of the structuring element

    alpha : float

        Smoothing parameter

    Returns
    -------
    jax.numpy.array
    """
    l = math.floor(kt.shape[0]/2)
    jit_local_dilation = Slocal_dilation(f,kt,l,alpha)
    return jnp.apply_along_axis(jit_local_dilation,1,l + index_f).reshape((f.shape[0] - 2*l,f.shape[1] - 2*l))

#Smooth dilation in batches
@jax.jit
def Sdilation(f,index_f,k,alpha = 5):
    """
    Smoothing dilation of batches of images.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the images

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    alpha : float

        Smoothing parameter

    Returns
    -------
    jax.numpy.array
    """
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    k = dmnn.transpose_se(k)
    db = jax.vmap(lambda f: Sdilation_2D(f,index_f,k,alpha),in_axes = (0),out_axes = 0)(f)
    return db

#Local anti-erosion of f by k for pixel (i,j) assuming k already transposed
def local_anti_erosion(f,kt,l):
    """
    Define function for local anti-erosion.
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
    a function that receives an index and returns the local anti-erosion of f by k at this index
    """
    def jit_local_anti_erosion(index):
        fw = jax.lax.dynamic_slice(f, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
        return jnp.max(kt - fw)
    return jit_local_anti_erosion

#Anti-erosion of f by k assuming k already transposed
@jax.jit
def anti_erosion_2D(f,index_f,kt):
    """
    Anti-erosion of 2D image.
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
    jit_local_anti_erosion = local_anti_erosion(f,kt,l)
    return jnp.apply_along_axis(jit_local_anti_erosion,1,l + index_f).reshape((f.shape[0] - 2*l,f.shape[1] - 2*l))

#Anti-erosion in batches
@jax.jit
def anti_erosion(f,index_f,k):
    """
    Anti-erosion of batches of images.
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
    ae = jax.vmap(lambda f: anti_erosion_2D(f,index_f,k),in_axes = (0),out_axes = 0)(f)
    return ae

#Local smooth anti-erosion of f by k for pixel (i,j) assuming k already transposed
def Slocal_anti_erosion(f,kt,l,alpha = 5):
    """
    Define function for smooth local anti-erosion.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        An image

    kt : jax.numpy.array

        The transpose of the structuring element

    l : int

        Length of structruring element. If k has shape d x d, then l is the greatest integer such that l <= d/2

    alpha : float

        Smoothing parameter

    Returns
    -------
    a function that receives an index and returns the smooth local anti-erosion of f by k at this index
    """
    def jit_local_anti_erosion(index):
        fw = jax.lax.dynamic_slice(f, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
        return smoothExt(kt - fw,alpha)
    return jit_local_anti_erosion

#Smooth anti-erosion of f by k assuming k already transposed
@jax.jit
def Santi_erosion_2D(f,index_f,kt,alpha = 5):
    """
    Smoothing anti-erosion of 2D image.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        A padded image

    index_f : jax.numpy.array

        Array with the indexes of f

    kt : jax.numpy.array

        The transpose of the structuring element

    alpha : float

        Smoothing parameter

    Returns
    -------
    jax.numpy.array
    """
    l = math.floor(kt.shape[0]/2)
    jit_local_anti_erosion = Slocal_anti_erosion(f,kt,l,alpha)
    return jnp.apply_along_axis(jit_local_anti_erosion,1,l + index_f).reshape((f.shape[0] - 2*l,f.shape[1] - 2*l))

#Smooth anti-erosion in batches
@jax.jit
def Santi_erosion(f,index_f,k,alpha = 5):
    """
    Smoothing anti-erosion of batches of images.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the images

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    alpha : float

        Smoothing parameter

    Returns
    -------
    jax.numpy.array
    """
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    k = dmnn.transpose_se(k)
    ae = jax.vmap(lambda f: Santi_erosion_2D(f,index_f,k,alpha),in_axes = (0),out_axes = 0)(f)
    return ae


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

#Opening of f by k
@jax.jit
def Sopening(f,index_f,k,alpha = 5):
    """
    Smooth opening of batches of images.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the images

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    alpha : float

        Smoothing parameter

    Returns
    -------
    jax.numpy.array
    """
    return Sdilation(Serosion(f,index_f,k,alpha),index_f,k,alpha)

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

#Colosing of f by k
@jax.jit
def Sclosing(f,index_f,k,alpha = 5):
    """
    Smooth closing of batches of images.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the images

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    alpha : float

        Smoothing parameter

    Returns
    -------
    jax.numpy.array
    """
    return Serosion(Sdilation(f,index_f,k,alpha),index_f,k,alpha)

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

#Alternate-sequential filter of f by k
@jax.jit
def Sasf(f,index_f,k,alpha = 5):
    """
    Smooth alternate-sequential filter applied to batches of images.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the images

    index_f : jax.numpy.array

        Array with the indexes of f

    k : jax.numpy.array

        Structuring element

    alpha : float

        Smoothing parameter

    Returns
    -------
    jax.numpy.array
    """
    return Sclosing(Sopening(f,index_f,k,alpha),index_f,k,alpha)

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
    jax.numpy.array
    """
    return  jnp.minimum(erosion(f,index_f,k1),anti_dilation(f,index_f,k2))

#Smooth sup-generating with interval [k1,k2]
@jax.jit
def Ssupgen(f,index_f,k1,k2,alpha = 5):
    """
    Smooth sup-generating operator applied to batches of images.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the binary images

    index_f : jax.numpy.array

        Array with the indexes of f

    k1,k2 : jax.numpy.array

        Limits of interval [k1,k2]

    alpha : float

        Smoothing parameter

    Returns
    -------
    jax.numpy.array
    """
    return  jnp.minimum(Serosion(f,index_f,k1,alpha),Santi_dilation(f,index_f,k2,alpha))

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
    jax.numpy.array
    """
    return jnp.maximum(anti_erosion(f,index_f,dmnn.transpose_se(k1)),dilation(f,index_f,(-1)*dmnn.transpose_se(k2)))

#Smooth inf-generating with interval [k1,k2]
def Sinfgen(f,index_f,k1,k2):
    """
    Smooth inf-generating operator applied to batches of images.
    -------
    Parameters
    ----------
    f : jax.numpy.array

        A 3D array with the binary images

    index_f : jax.numpy.array

        Array with the indexes of f

    k1,k2 : jax.numpy.array

        Limits of interval [k1,k2]

    alpha : float

        Smoothing parameter

    Returns
    -------
    jax.numpy.array
    """
    return jnp.maximum(Santi_erosion(f,index_f,k1,alpha),Sdilation(f,index_f,(-1)*k2,alpha))

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
def operator(type,smooth = False,alpha = 5):
    """
    Get a morphological operator by name.
    -------

    Parameters
    ----------
    type : str

        Name of operator

    smooth : logical

        Whether to consider smooth operators

    alpha : float

        Smoothing parameter

    Returns
    -------

    a function that applies a morphological operator

    """
    if not smooth:
        if type == 'erosion':
            oper = lambda x,index_x,k: erosion(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])))
        elif type == 'dilation':
            oper = lambda x,index_x,k: dilation(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])))
        elif type == 'anti-erosion':
            oper = lambda x,index_x,k: anti_erosion(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])))
        elif type == 'anti-dilation':
            oper = lambda x,index_x,k: anti_dilation(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])))
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
    else:
        if type == 'erosion':
            oper = lambda x,index_x,k: Serosion(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),alpha)
        elif type == 'dilation':
            oper = lambda x,index_x,k: Sdilation(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),alpha)
        elif type == 'anti-erosion':
            oper = lambda x,index_x,k: Santi_erosion(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),alpha)
        elif type == 'anti-dilation':
            oper = lambda x,index_x,k: Santi_dilation(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),alpha)
        elif type == 'opening':
            oper = lambda x,index_x,k: Sopening(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),alpha)
        elif type == 'closing':
            oper = lambda x,index_x,k: Sclosing(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),alpha)
        elif type == 'asf':
            oper = lambda x,index_x,k: Sasf(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),alpha)
        elif type == 'supgen':
            oper = lambda x,index_x,k: Ssupgen(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),jax.lax.slice_in_dim(k,1,2).reshape((k.shape[1],k.shape[2])),alpha)
        elif type == 'infgen':
            oper = lambda x,index_x,k: Sinfgen(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),jax.lax.slice_in_dim(k,1,2).reshape((k.shape[1],k.shape[2])),alpha)
        else:
            print('Type of layer ' + type + 'is wrong!')
            return 1
    return jax.jit(oper)

#Compute tight structuring element for indentity erosion
def tight_se_identity(data,d):
    #Data parameters
    l = math.floor(d/2)
    shape_data = (data.shape[1] - l,data.shape[2] - l)
    index_data = l + mnn.index_array(shape_data)
    #Get neighbourhood for fixed image and pixel
    def jit_local_value(x,index):
        fw = jax.lax.dynamic_slice(x, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
        return fw - fw[l,l]
    # Compute tight se
    neigh = jax.vmap(lambda x: jax.vmap(lambda index: jit_local_value(x,index))(index_data))(data)
    return jnp.min(neigh,(0,1))
