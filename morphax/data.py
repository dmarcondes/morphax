#Functions to process data for training
import pandas
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import random
import sys
from PIL import Image
from IPython.display import display

__docformat__ = "numpy"

#Read and organize a data.frame
def read_data_frame(file,sep = None,header = 'infer',sheet = 0):
    """
    Read a data file and convert to JAX array.
    -------

    Parameters
    ----------
    file : str

        File name with extension .csv, .txt, .xls or .xlsx

    sep : str

        Separation character for .csv and .txt files. Default ',' for .csv and ' ' for .txt

    header : int, Sequence of int, ‘infer’ or None

        See pandas.read_csv documentation. Default 'infer'

    sheet : int

        Sheet number for .xls and .xlsx files. Default 0

    Returns
    -------

    a JAX numpy array

    """

    #Find out data extension
    ext = file.split('.')[1]

    #Read data frame
    if ext == 'csv':
        if sep is None:
            sep = ','
        dat = pandas.read_csv(file,sep = sep,header = header)
    elif ext == 'txt':
        if sep is None:
            sep = ' '
        dat = pandas.read_table(file,sep = sep,header = header)
    elif ext == 'xls' or ext == 'xlsx':
        dat = pandas.read_excel(file,header = header,sheet_name = sheet)

    #Convert to JAX data structure
    dat = jnp.array(dat,dtype = jnp.float32)

    return dat

#Read images into an array
def image_to_jnp(files_path):
    """
    Read an image file and convert to JAX array.
    -------

    Parameters
    ----------
    files_path : list of str

        List with the paths of the images to read

    Returns
    -------

    a JAX numpy array

    """
    dat = None
    for f in files_path:
        img = Image.open(f)
        img = jnp.array(img,dtype = jnp.float32)/255
        if len(img.shape) == 3:
            img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
        else:
            img = img.reshape((1,img.shape[0],img.shape[1]))
        if dat is None:
            dat = img
        else:
            dat = jnp.append(dat,img,0)
    return dat.astype(jnp.float32)

#Save images
def save_images(images,files_path):
    """
    Save images in a JAX array to file.
    -------

    Parameters
    ----------
    images : JAX array

    Array of images

    files_path : list of str

        List with the paths of the images to write
    """
    if len(files_path) > 1:
        for i in range(len(files_path)):
            if len(images.shape) == 4:
                tmp = Image.fromarray(np.uint8(jnp.round(255*images[i,:,:,:]))).convert('RGB')
            else:
                tmp = Image.fromarray(np.uint8(jnp.round(255*images[i,:,:])))
            tmp.save(files_path[i])
    else:
        if len(images.shape) == 4:
            tmp = Image.fromarray(np.uint8(jnp.round(255*images[0,:,:,:]))).convert('RGB')
        else:
            tmp = Image.fromarray(np.uint8(jnp.round(255*images[0,:,:])))
        tmp.save(files_path[0])

#Print images
def print_images(images):
    """
    Print images in a JAX array.
    -------

    Parameters
    ----------
    images : JAX array

    Array of images
    """
    for i in range(images.shape[0]):
        if len(images.shape) == 4:
            tmp = Image.fromarray(np.uint8(jnp.round(255*images[i,:,:,:]))).convert('RGB')
        else:
            tmp = Image.fromarray(np.uint8(jnp.round(255*images[i,:,:]))).convert('RGB')
        display(tmp)
