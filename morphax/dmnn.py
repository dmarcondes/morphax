#Discrete morphology module
import jax
import jax.numpy as jnp
import math
import numpy as np

#Create an index array for an array
def index_array(shape):
    return jnp.array([[x,y] for x in range(shape[0]) for y in range(shape[1])])

#Transpose a structuring element
def transpose_se(k):
    d = k.shape[0]
    kt = np.zeros((d,d),dtype = int)
    for i in range(d):
        for j in range(d):
            kt[i,j] = kt[i,j] + k[d - 1 - i,d - 1 - j]
    return jnp.array(kt)

#Local erosion of f by k for pixel (i,j)
def local_erosion(f,k,l):
    def jit_local_erosion(index):
        fw = jax.lax.dynamic_slice(f, (index[0], index[1]), (2*l + 1, 2*l + 1))
        return jnp.min(fw[k == 1])
    return jit_local_erosion

#Erosion of f by k
@jax.jit
def erosion_2D(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    jit_local_erosion = local_erosion(f,k,l)
    return jnp.apply_along_axis(jit_local_erosion,1,index_f).reshape((f.shape[0] - 2*l,f.shape[1] - 2*l))

#Erosion in batches
@jax.jit
def erosion(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    eb = jax.vmap(lambda f: erosion_2D(f,index_f,k),in_axes = (0),out_axes = 0)(f)
    return eb

#Local dilation of f by k for pixel (i,j)
def local_dilation(f,k,l):
    def jit_local_dilation(index):
        fw = jax.lax.dynamic_slice(f, (index[0], index[1]), (2*l + 1, 2*l + 1))
        return jnp.max(fw[k == 1])
    return jit_local_dilation

#Dilation of f by k
@jax.jit
def dilation_2D(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    jit_local_dilation = local_dilation(f,k,l)
    return jnp.apply_along_axis(jit_local_dilation,1,index_f).reshape((f.shape[0] - 2*l,f.shape[1] - 2*l))

#Dilation in batches
@jax.jit
def dilation(f,index_f,k,h = 1/5):
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    k = transpose_se(k)
    db = jax.vmap(lambda f: dilation_2D(f,index_f,k),in_axes = (0),out_axes = 0)(f)
    return db

#Opening of f by k
def opening(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    eb = jax.vmap(lambda f: erosion_2D(f,index_f,k),in_axes = (0),out_axes = 0)
    db = jax.vmap(lambda f: dilation_2D(f,index_f,transpose_se(k)),in_axes = (0),out_axes = 0)
    f = eb(f)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    return db(f)

#Colosing of f by k
def closing(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    eb = jax.vmap(lambda f: erosion_2D(f,index_f,k),in_axes = (0),out_axes = 0)
    db = jax.vmap(lambda f: dilation_2D(f,index_f,transpose_se(k)),in_axes = (0),out_axes = 0)
    f = db(f)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    return eb(f)

#Alternate-sequential filter of f by k
def asf(f,index_f,k):
    fo = opening(f,index_f,k)
    return closing(fo,index_f,k)

#Complement
@jax.jit
def complement(f):
    return 1 - f

#Sup-generating with interval [k1,k2]
def supgen(f,index_f,k1,k2):
    return jnp.minimum(erosion(f,index_f,k1),complement(dilation(f,index_f,complement(transpose_se(k2)))))

#Inf-generating with interval [k1,k2]
def infgen(f,index_f,k1,k2):
    return maximum(dilation(f,index_f,k1),complement(erosion(f,index_f,complement(transpose_se(k2)))))

#Sup of array of images
@jax.jit
def sup(f):
    f = jnp.apply_along_axis(jnp.maximum,0,f)
    return f.reshape((1,f.shape[1],f.shape[2]))

#Sup vmap for arch
vmap_sup = lambda f,h: jax.jit(jax.vmap(lambda f: sup(f),in_axes = (1),out_axes = 1))(f)

#Inf of array of images
@jax.jit
def inf(f):
    f = jnp.apply_along_axis(jnp.minimum,0,f)
    return f.reshape((1,f.shape[1],f.shape[2]))

#Inf vmap for arch
vmap_inf = lambda f,h: jax.jit(jax.vmap(lambda f: inf(f),in_axes = (1),out_axes = 1))(f)

#Return operator by name
def operator(type,h):
    if type == 'erosion':
        oper = lambda x,index_x,k: erosion(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),h)
    elif type == 'dilation':
        oper = lambda x,index_x,k: dilation(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),h)
    elif type == 'opening':
        oper = lambda x,index_x,k: opening(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),h)
    elif type == 'closing':
        oper = lambda x,index_x,k: closing(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),h)
    elif type == 'asf':
        oper = lambda x,index_x,k: asf(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),h)
    elif type == 'supgen':
        oper = lambda x,index_x,k: supgen(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),jax.lax.slice_in_dim(k,1,2).reshape((k.shape[1],k.shape[2])),h)
    elif type == 'infgen':
        oper = lambda x,index_x,k: infgen(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),jax.lax.slice_in_dim(k,1,2).reshape((k.shape[1],k.shape[2])),h)
    else:
        print('Type of layer ' + type + 'is wrong!')
        return 1
    return oper

#Structuring element of the approximate identity operator in a sample
def struct_lower(x,d):
    #Function to apply to each index
    l = math.floor(d/2)
    x = jax.lax.pad(x,0.0,((0,0,0),(l,l,0),(l,l,0)))
    index_x = index_array((x.shape[1],x.shape[2]))
    def struct_lower(index,x):
        fw = jax.lax.dynamic_slice(x, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
        return fw - x[index[0],index[1]]
    k = jax.vmap(lambda x: jnp.apply_along_axis(lambda index: struct_lower(index,x),1,index_x))(x).reshape((x.shape[0],x.shape[1],x.shape[2],d,d))
    k = k.reshape((k.shape[0]*k.shape[1]*k.shape[2],d,d))
    k = jnp.apply_along_axis(lambda k: jnp.min(k),0,k)
    return k

#Structuring element of upper limit of interval of supgen approximating identity operator
def struct_upper(x,d):
    #Function to apply to each index
    l = math.floor(d/2)
    x = jax.lax.pad(x,0.0,((0,0,0),(l,l,0),(l,l,0)))
    index_x = index_array((x.shape[1],x.shape[2]))
    def struct_upper(index,x):
        fw = jax.lax.dynamic_slice(x, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
        return 1 - fw - x[index[0],index[1]]
    k = jax.vmap(lambda x: jnp.apply_along_axis(lambda index: struct_upper(index,x),1,index_x))(x).reshape((x.shape[0],x.shape[1],x.shape[2],d,d))
    k = k.reshape((k.shape[0]*k.shape[1]*k.shape[2],d,d))
    k = jnp.apply_along_axis(lambda k: jnp.min(k),0,k)
    return k
