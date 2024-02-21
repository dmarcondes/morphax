#Install from GitHub
#python3 setup.py sdist bdist_wheel
pydoc morphax/* -o docs/
pip3 uninstall jinnax; pip3 install --upgrade git+https://github.com/dmarcondes/JINNAX
pip3 uninstall --break-system-packages jinnax; pip3 install --break-system-packages git+https://github.com/dmarcondes/JINNAX
apt install python3-git+https://github.com/dmarcondes/JINNAX
from jinnax import data as jd
from jinnax import arch as jar
from jinnax import training as jtr
import jax.numpy as jnp
import time
import jax

####Test open and save images###
bw_images = jd.image_to_jnp(['Phi_comp_'+ str(i+1) + '.jpg' for i in range(5)])
bw_images.shape
color_images = jd.image_to_jnp(['small_dl.png'])
color_images.shape

jd.save_images(bw_images,['saved_Phi_comp_'+ str(i+1) + '.jpg' for i in range(5)])
jd.save_images(bw_images[0,:,:].reshape((1,585,570)),['saved_Phi_comp_'+ str(i+1) + '.jpg' for i in range(1)])
jd.save_images(color_images,['save_dl.jpeg','save_color_2.png'])
im = color_images[:,:,:,1]

####Test morphology functions####

#Structuring element from function
def fk(w):
    res = [0.05 for i in range(w.shape[0])]
    return res

def fk1(w):
    res = [0.75 for i in range(w.shape[0])]
    return res

k = jmp.struct_function(fk,3)
k1 = jmp.struct_function(fk1,3)

#Create index matrix
index_f = jmp.index_array((160,160))
index_x = jmp.index_array((160,160))

#Erosion/dilation
d = jmp.dilation
e = jmp.erosion
fd = d(im,index_f,k)
fe = e(im,index_f,k)
jd.save_images(fe,['erosion_test.png'])
jd.save_images(fd,['dilation_test.png'])

#Opening/closing
o = jmp.opening
c = jmp.closing
fo = o(im,index_f,k)
fc = c(im,index_f,k)
jd.save_images(fo,['opening_test.png'])
jd.save_images(fc,['closing_test.png'])

#asf
asf = jmp.asf
fa = asf(im,index_f,k1)
jd.save_images(fa,['asf_test.png'])

#Complement
fc = jmp.complement(im)
jd.save_images(fc,['complement_test.png'])

#Sup/ing-gen
k1 = struct_upper(x,3)
k0 = struct_lower(x,3)
sg = jmp.supgen
ig = jmp.infgen
fs = sg(im,index_f,k0,k1)
fi = ig(im,index_f,k1,k)
jd.save_images(fs,['sgen_test.png'])
jd.save_images(fi,['sinf_test.png'])
round(jmp.erosion(im,index_f,k0) - im,3)

#Sup/inf
fs = jmp.sup(im)
fi = jmp.inf(im)

####Learn the target under correct specification#####

#Learn erosion
k = jnp.array([0,-0.1,0,-0.05,0,-0.05,0,-0.1,0]).reshape((3,3))
k1 = jnp.array([-0.1,0,-0.1,-0.085,0,-0.01,0.01,-0.1,0.1]).reshape((3,3))
x = im
y = jmp.erosion(x,index_x,k).reshape((1,x.shape[1],x.shape[2]))
y = jmp.erosion(y,index_x,k1).reshape((1,x.shape[1],x.shape[2]))
jd.save_images(y,['output.png'])

type = ['erosion','erosion']
width = [1,1]
size = [3,1]
shape_x = x.shape[1:3]
index_x = jmp.index_array(shape_x)
width_str = 5 * [16]
net_iter = jar.cmnn_iter(type,width,width_str,size,shape_x,x = x,activation = jax.nn.tanh,sa = True,lr = 1e-3,epochs = 1000)
net = jar.cmnn(x,type,width,size,shape_x)

loss = jtr.MSE
lr = 1e-3
par_iter = jtr.train_morph(x,y,net_iter['forward'],net_iter['params'],loss,sa = False,lr = lr,epochs = 5000,batches = 1)
jnp.round(net_iter['compute_struct'](par_iter)[0][0],3) - k
jd.save_images(net_iter['forward'](x,net_iter['params']),['pred.png'])
par = jtr.train_morph(x,y,net['forward'],net['params'],loss,lr = lr,epochs = 5000,batches = 1)
jnp.round(2*jax.nn.tanh(par[0]),3)
2*jax.nn.tanh(net['params'][0])
jnp.std(net['params'][0])


#Generate PINN data
u = lambda x,t: x[1] + x[0]*t
xlo = 0
xup = 1
tup = 1
Ns = None
Nt = None
Nb = None
Ntb = None
Ni = None
Nc = None
Ntc = None
train = True
tlo = 0
d = 3
poss = 'grid'
post = 'grid'
posi = 'grid'
posb = 'grid'
postb = 'grid'
posc = 'grid'
postc = 'random'
sigmas = 0
sigmab = 0
sigmai = 0
data = jd.generate_PINNdata(u,xlo,xup,tlo,tup,Ns,Nt,Nb,Ntb,Ni,Nc,Ntc,train,d,poss,post,posi,posb,postb,posc,postc,sigmas,sigmab,sigmai)
data['boundary'].shape[0]/Ntb
(Nb + 2) ** d - Nb ** d
