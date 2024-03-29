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
    return jnp.array([[x,y] for x in range(shape[0]) for y in range(shape[1])])

#Transpose a structuring element
@jax.jit
def transpose_se(k):
    d = k.shape[0]
    kt = k
    for i in range(d):
        for j in range(d):
            kt = kt.at[i,j].set(k[d - 1 - i,d - 1 - j])
    return kt

#Local erosion of f by k for pixel (i,j)
def local_erosion(f,k,l):
    def jit_local_erosion(index):
        fw = jax.lax.dynamic_slice(f, (index[0], index[1]), (2*l + 1, 2*l + 1))
        return jnp.min(jnp.where(k == 1, fw, 1))
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
    f = jax.lax.pad(f,0,((0,0,0),(l,l,0),(l,l,0)))
    eb = jax.vmap(lambda f: erosion_2D(f,index_f,k),in_axes = (0),out_axes = 0)(f)
    return eb

#Local dilation of f by k for pixel (i,j)
def local_dilation(f,k,l):
    def jit_local_dilation(index):
        fw = jax.lax.dynamic_slice(f, (index[0], index[1]), (2*l + 1, 2*l + 1))
        return jnp.max(jnp.where(k == 1, fw, 0))
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
    f = jax.lax.pad(f,0,((0,0,0),(l,l,0),(l,l,0)))
    k = transpose_se(k)
    db = jax.vmap(lambda f: dilation_2D(f,index_f,k),in_axes = (0),out_axes = 0)(f)
    return db

#Opening of f by k
@jax.jit
def opening(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0,((0,0,0),(l,l,0),(l,l,0)))
    eb = jax.vmap(lambda f: erosion_2D(f,index_f,k),in_axes = (0),out_axes = 0)
    db = jax.vmap(lambda f: dilation_2D(f,index_f,transpose_se(k)),in_axes = (0),out_axes = 0)
    f = eb(f)
    f = jax.lax.pad(f,0,((0,0,0),(l,l,0),(l,l,0)))
    return db(f)

#Colosing of f by k
@jax.jit
def closing(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0,((0,0,0),(l,l,0),(l,l,0)))
    eb = jax.vmap(lambda f: erosion_2D(f,index_f,k),in_axes = (0),out_axes = 0)
    db = jax.vmap(lambda f: dilation_2D(f,index_f,transpose_se(k)),in_axes = (0),out_axes = 0)
    f = db(f)
    f = jax.lax.pad(f,0,((0,0,0),(l,l,0),(l,l,0)))
    return eb(f)

#Alternate-sequential filter of f by k
@jax.jit
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
    return jnp.maximum(dilation(f,index_f,k1),complement(erosion(f,index_f,complement(transpose_se(k2)))))

#Sup of array of images
@jax.jit
def sup(f):
    fs = jnp.apply_along_axis(jnp.max,0,f)
    return fs.reshape((1,f.shape[1],f.shape[2]))

#Sup vmap for arch
vmap_sup = lambda f: jax.jit(jax.vmap(lambda f: sup(f),in_axes = (1),out_axes = 1))(f)

#Inf of array of images
@jax.jit
def inf(f):
    fi = jnp.apply_along_axis(jnp.min,0,f)
    return fi.reshape((1,f.shape[1],f.shape[2]))

#Inf vmap for arch
vmap_inf = lambda f: jax.jit(jax.vmap(lambda f: inf(f),in_axes = (1),out_axes = 1))(f)

#Return operator by name
def operator(type):
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
    return oper

####Discrete Morphological Neural Networks####

#MSE
@jax.jit
def MSE(true,pred):
  return jnp.mean((true - pred)**2)

#L2 error
@jax.jit
def L2error(pred,true):
  return jnp.sqrt(jnp.sum((true - pred)**2))/jnp.sqrt(jnp.sum(true ** 2))

#Croos entropy
@jax.jit
def CE(true,pred):
  return jnp.mean((- true * jnp.log(pred + 1e-5) - (1 - true) * jnp.log(1 - pred + 1e-5)))

#IoU
@jax.jit
def IoU(true,pred):
  return 1 - (jnp.sum(2 * true * pred) + 1)/(jnp.sum(true + pred) + 1)

#Apply a morphological layer
def apply_morph_layer(x,type,params,index_x):
    #Apply each operator
    oper = operator(type)
    fx = None
    for i in range(params.shape[0]):
        if fx is None:
            fx = oper(x,index_x,params[i,:,:,:]).reshape((1,x.shape[0],x.shape[1],x.shape[2]))
        else:
            fx = jnp.append(fx,oper(x,index_x,params[i,:,:,:]).reshape((1,x.shape[0],x.shape[1],x.shape[2])),0)
    return fx

#Canonical Discrete Morphological NN
def cdmnn(type,width,size,shape_x,key = 0):
    key = jax.random.split(jax.random.PRNGKey(key),(len(width),max(width)))

    #Index window
    index_x = index_array(shape_x)

    #Initialize parameters
    params = list()
    for i in range(len(width)):
        if type[i] in ['sup','inf','complement']:
            params.append(jnp.array(0.0).reshape((1,1,1)))
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

    #Forward pass
    @jax.jit
    def forward(x,params):
        x = x.reshape((1,x.shape[0],x.shape[1],x.shape[2]))
        for i in range(len(type)):
            #Apply sup and inf
            if type[i] == 'sup':
                x = vmap_sup(x)
            elif type[i] == 'inf':
                x = vmap_inf(x)
            elif type[i] == 'complement':
                x = 1 - x
            else:
                #Apply other layer
                x = apply_morph_layer(x[0,:,:,:],type[i],params[i],index_x)
        return x[0,:,:,:]

    #Return initial parameters and forward function
    return {'params': params,'forward': forward,'width': width,'size': size,'type': type}

#Step SLDA
def step_slda(params,x,y,forward,lf,type,sample = False,neighbors = None):
    #Current error
    error = lf(params,x,y)

    #Sample
    if sample:
        #Calculate probabilities
        prob = []
        for i in range(len(params)):
            if params[i].shape[2] > 1:
                prob  = prob + [params[i].shape[0] * (params[i].shape[2] ** 2)]
            else:
                prob  = prob + [0]
            if params[i].shape[1] == 2:
                prob[-1] = 2*prob[-1]

        prob = [x/sum(prob) for x in prob]
        # TBD
    else:
        new_par = params.copy()
        range_layers = list(range(len(params)))
        random.shuffle(range_layers)
        for l in range_layers:
            if type[l] != 'inf' and type[l] != 'sup' and type[l] != 'complement':
                range_nodes = list(range(params[l].shape[0]))
                random.shuffle(range_nodes)
                for n in range_nodes:
                    range_lim = list(range(params[l].shape[1]))
                    random.shuffle(range_lim)
                    for lm in range_lim:
                        range_row = list(range(params[l].shape[2]))
                        random.shuffle(range_row)
                        range_col = list(range(params[l].shape[3]))
                        random.shuffle(range_col)
                        for i in range_row:
                            for j in range_col:
                                if (lm == 0 and params[l][n,l,i,j] == 1) or (lm == 0 and params[l][n,1,i,j] == 1) or (lm == 1 and params[l][n,l,i,j] == 0) or (lm == 1 and params[l][n,1,i,j] == 0):
                                    test_par = params.copy()
                                    test_par[l] = params[l].at[n,lm,i,j].set(1 - params[l][n,lm,i,j])
                                    test_error = lf(test_par,x,y)
                                    new_par,error = jax.lax.cond(test_error <= error, lambda x = 0: (test_par.copy(),test_error), lambda x = 0: (new_par,error))
                                    del test_par, test_error
    return new_par

#Training function MNN
def train_dmnn(x,y,forward,params,loss,type,sample = False,neighbors = None,epochs = 1,batches = 1,key = 0,notebook = False,epoch_print = 100):
    #Key
    key = jax.random.split(jax.random.PRNGKey(key),epochs)

    #Batch size
    bsize = int(math.floor(x.shape[0]/batches))

    #Loss function
    @jax.jit
    def lf(params,x,y):
        return jnp.mean(jax.vmap(loss,in_axes = (0,0))(forward(x,params),y))

    #Training function
    #@jax.jit
    def update(params,x,y):
      params = step_slda(params,x,y,forward,lf,type,sample,neighbors)
      return params

    #Train
    t0 = time.time()
    with alive_bar(epochs) as bar:
        for e in range(epochs):
            #Permutate x
            x = jax.random.permutation(jax.random.PRNGKey(key[e,0]),x,0)
            for b in range(batches):
                if b < batches - 1:
                    xb = jax.lax.dynamic_slice(x,(b*bsize,0,0),(bsize,x.shape[1],x.shape[2]))
                    yb = jax.lax.dynamic_slice(x,(b*bsize,0,0),(bsize,x.shape[1],x.shape[2]))
                else:
                    xb = x[b*bsize:x.shape[0],:,:]
                    yb = y[b*bsize:y.shape[0],:,:]
                params = update(params,xb,yb)
            if e % epoch_print == 0:
                l = str(jnp.round(lf(params,x,y),10))
                if notebook:
                    print('Epoch: ' + str(e) + ' Time: ' + str(jnp.round(time.time() - t0,2)) + ' s Loss: ' + l)
                if not notebook:
                    bar.title("Loss: " + l)
            bar()

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
