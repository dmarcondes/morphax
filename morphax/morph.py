#Morphology module
import jax
import jax.numpy as jnp
import math

#Approximate maximum
def max(x,h = 1/5):
    return h * jnp.log(jnp.sum(jnp.exp(x/h)))

def maximum(x,y,h = 1/5):
    if len(x.shape) == 2:
        x = x.reshape((1,x.shape[0],x.shape[1]))
        y = y.reshape((1,y.shape[0],y.shape[1]))
    return jax.vmap(jax.vmap(jax.vmap(lambda x,y: h * jnp.log(jnp.sum(jnp.exp(jnp.append(x,y)/h))))))(x,y)

def maximum_array_number(arr,x,h = 1/5):
    return h * jnp.log(jnp.exp(arr/h) + jnp.exp(x/h))

#Approximate minimum
def min(x,h = 1/5):
    return max(x,-h)

def minimum(x,y,h = 1/5):
    return maximum(x,y,-h)

def minimum_array_number(arr,x,h = 1/5):
    return maximum_array_number(arr,x,-h)

#Transpose a structuring element
@jax.jit
def transpose_se(k):
    d = k.shape[0]
    kt = k
    for i in range(d):
        for j in range(d):
            kt = kt.at[i,j].set(k[d - 1 - i,d - 1 - j])
    return kt

#Structuring element from function
def struct_function(k,d):
    w = jnp.array([[x1.tolist(),x2.tolist()] for x1 in jnp.linspace(-jnp.floor(d/2),jnp.floor(d/2),d) for x2 in jnp.linspace(jnp.floor(d/2),-jnp.floor(d/2),d)])
    k = jnp.array(k(w))
    return jnp.transpose(k.reshape((d,d)))

def struct_function_w(k,w,d):
    k = jnp.array(k(w))
    return jnp.transpose(k.reshape((d,d)))

#Create an index array for an array
def index_array(shape):
    return jnp.array([[x,y] for x in range(shape[0]) for y in range(shape[1])])

#Local erosion of f by k for pixel (i,j)
def local_erosion(f,k,l,h = 1/5):
    def jit_local_erosion(index):
        fw = jax.lax.dynamic_slice(f, (index[0], index[1]), (2*l + 1, 2*l + 1))
        return jnp.minimum(jnp.maximum(min(fw - k,h),0.0),1.0)
    return jit_local_erosion

#Erosion of f by k
@jax.jit
def erosion_2D(f,index_f,k,h = 1/5):
    l = math.floor(k.shape[0]/2)
    jit_local_erosion = local_erosion(f,k,l,h)
    return jnp.apply_along_axis(jit_local_erosion,1,index_f).reshape((f.shape[0] - 2*l,f.shape[1] - 2*l))

#Erosion in batches
@jax.jit
def erosion(f,index_f,k,h = 1/5):
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    eb = jax.vmap(lambda f: erosion_2D(f,index_f,k,h),in_axes = (0),out_axes = 0)(f)
    return eb

#Local dilation of f by k for pixel (i,j) assuming k already transposed
def local_dilation(f,k,l,h = 1/5):
    def jit_local_dilation(index):
        fw = jax.lax.dynamic_slice(f, (index[0], index[1]), (2*l + 1, 2*l + 1))
        return jnp.minimum(jnp.maximum(max(fw + k,h),0.0),1.0)
    return jit_local_dilation

#Dilation of f by k assuming k already transposed
@jax.jit
def dilation_2D(f,index_f,k,h = 1/5):
    l = math.floor(k.shape[0]/2)
    jit_local_dilation = local_dilation(f,k,l,h)
    return jnp.apply_along_axis(jit_local_dilation,1,index_f).reshape((f.shape[0] - 2*l,f.shape[1] - 2*l))

#Dilation in batches
@jax.jit
def dilation(f,index_f,k,h = 1/5):
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    k = transpose_se(k)
    db = jax.vmap(lambda f: dilation_2D(f,index_f,k,h),in_axes = (0),out_axes = 0)(f)
    return db

#Opening of f by k
@jax.jit
def opening(f,index_f,k,h = 1/5):
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    eb = jax.vmap(lambda f: erosion_2D(f,index_f,k,h),in_axes = (0),out_axes = 0)
    db = jax.vmap(lambda f: dilation_2D(f,index_f,transpose_se(k),h),in_axes = (0),out_axes = 0)
    f = eb(f)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    return db(f)

#Colosing of f by k
@jax.jit
def closing(f,index_f,k,h = 1/5):
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    eb = jax.vmap(lambda f: erosion_2D(f,index_f,k,h),in_axes = (0),out_axes = 0)
    db = jax.vmap(lambda f: dilation_2D(f,index_f,transpose_se(k),h),in_axes = (0),out_axes = 0)
    f = db(f)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    return eb(f)

#Alternate-sequential filter of f by k
@jax.jit
def asf(f,index_f,k,h = 1/5):
    fo = opening(f,index_f,k,h)
    return closing(fo,index_f,k,h)

#Complement
@jax.jit
def complement(f,m = 1):
    return m - f

#Sup-generating with interval [k1,k2]
@jax.jit
def supgen(f,index_f,k1,k2,h = 1/5,m = 1):
    K1 = minimum(k1,k2,h)
    K2 = maximum(k1,k2,h)
    return minimum(erosion(f,index_f,K1,h),complement(dilation(f,index_f,K2,h),m),h)

#Inf-generating with interval [k1,k2]
@jax.jit
def infgen(f,index_f,k1,k2,h = 1/5,m = 1):
    K1 = minimum(k1,k2,h)
    K2 = maximum(k1,k2,h)
    return maximum(dilation(f,index_f,K1,h),complement(erosion(f,index_f,K2,h),m),h)

#Sup of array of images
@jax.jit
def sup(f,h = 1/5):
    fs = jnp.exp(f/h)
    fs = h * jnp.log(jnp.apply_along_axis(jnp.sum,0,fs))
    return fs.reshape((1,f.shape[1],f.shape[2]))

#Sup vmap for arch
vmap_sup = lambda f,h: jax.jit(jax.vmap(lambda f: sup(f,h),in_axes = (1),out_axes = 1))(f)

#Inf of array of images
@jax.jit
def inf(f,h = 1/5):
    return sup(f,-h)

#Inf vmap for arch
vmap_inf = lambda f,h: jax.jit(jax.vmap(lambda f: inf(f,h),in_axes = (1),out_axes = 1))(f)

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
