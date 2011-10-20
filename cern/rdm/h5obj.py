"""
Serialize and deserialize arbitrary nested python objects, dictionaries, numpy
arrays in a HDF5 file using h5py.

The correspondence is:
  scalar -> H5Attr
  numpy arrays -> H5Dataset
  dict -> H5Group
  instance -> H5Group with attribute __class__
  representable -> H5Attr with name ending with __pyrepr
  other -> H5Attr with name ending with __pyrepr

When loading an instance, if its class is found in the global namespace, an
object is created using the static method 'fromdict' if available or
bypassing the __init__ method and updating the dictionary with data, otherwise
a dictionary is created.
"""

import cPickle as pickle
import inspect

import h5py
import numpy as np


def _get_serial_class(obj):
  """
  Return serialization method, data to serialize, metadata

  scalar -> H5Attr
  numpy arrays -> H5Dataset
  dict -> H5Group
  obj  -> H5Group with attr __class__
  representable -> H5Attr with name ending with __pyrepr
  other -> H5Attr with name ending with __pyrepr
  """
  if np.isscalar(obj):
    return 'scalar',obj, None
  if hasattr(obj,'dtype'):
    return 'dataset',obj, None
  if hasattr(obj,'keys'):
    return 'group',obj,'dict'
  if hasattr(obj,'__dict__'):
    return 'group',obj.__dict__,obj.__class__.__name__
  try:
    t=repr(obj)
    eval(t,{},{})
    return 'scalar',t,'pyrepr'
  except:
    return 'scalar',cPickle.dumps(obj),'pickle'


def dump(obj,h5g,compression=5,shuffle=True):
  """
  Dump a python object in an H5File, or H5Group.

  Usage:
    h5g: H5File, H5Group or filename

  """
  if type(h5g) is str:
    h5g=h5py.File(h5g,'w-')
  method,data,pytype=_get_serial_class(obj)
  assert method=='group'
  if pytype is not 'dict':
    h5g.attrs.create('__class__',pytype)
  for k in data.keys():
    aobj=data[k]
    method,adata,pytype=_get_serial_class(aobj)
    if method=='group':
      g=h5g.create_group(k)
      dump(aobj,g)
    elif method=='dataset':
      h5g.create_dataset(k,data=aobj,compression=compression,shuffle=shuffle)
    elif method=='scalar':
      if pytype is None:
        h5g.attrs.create(k,adata)
      else:
        h5g.attrs.create("%s__%s"%(k,pytype),adata)
  if hasattr(h5g,'flush'):
    h5g.flush()


def _findname(namespace,pytype):
  cls=None
  if namespace is None:
    f=inspect.currentframe()
    while(f):
      if pytype in f.f_globals:
        cls=f.f_globals[pytype]
        break
      f=f.f_back
  else:
    cls=namespace.get(pytype)
  return cls


def _objcreate(data,namespace=None):
  if namespace is None:
    mydict=dict
  else:
    mydict=namespace.get('dict',dict)
  if '__class__' in data:
    pytype=data['__class__']
    cls=_findname(namespace,pytype)
    if cls is None:
      msg='Warning h5obj: `%s` not found, returing a %s'
      print msg%(pytype,mydict)
      obj=mydict(data)
    else:
      del data['__class__']
      if hasattr(cls,'fromdict'):
        obj=cls.fromdict(data)
      else:
        obj=cls.__new__(cls)
        obj.__dict__.update(data)
  else:
    obj=mydict(data)
  return obj



def load(h5g,load_dataset=True,namespace=None):
  """
  Load a data from an H5File, or H5Group.

  Usage:
    h5g: H5File, H5Group or filename
    loaddataset: if True load dataset to numpy array otherwise
       leave then as H5Datasets
    dictclass: a dictionary like class for group without specific class
       information
  """
  if type(h5g) is str:
    h5g=h5py.File(h5g,'r')
  cls=None
  data={}
  for k in h5g.attrs:
    if k.endswith('__pyrepr'):
      data[k[:-8]]=eval(h5g.attrs[k],{},{})
    elif k.endswith('__pickle'):
      data[k[:-8]]=pickle.loads(h5g.attrs[k])
    else:
      data[k]=h5g.attrs[k]
  for k in h5g.keys():
    g=h5g[k]
    if hasattr(g,'dtype'):
      if load_dataset:
        data[k]=h5g[k][:]
      else:
        data[k]=h5g[k]
    else:
      data[k]=load(g,load_dataset=load_dataset,namespace=namespace)
  return _objcreate(data,namespace=namespace)


if __name__=='__main__':
  from numpy import arange
  import os

  class myclass(object):
    pass

  o1={'a':1,'b':[3,4,5],'c':arange(400)}

  o2=myclass()
  o2.o1=o1
  o2.a2=2
  o2.a3='fasdfafa'
  o3=myclass()
  o3.o2=o2

  os.system('rm o1.h5 o2.h5 o3.h5')

  dump(o1,'o1.h5')
  dump(o2,'o2.h5')
  dump(o3,'o3.h5')

  oo1=load('o1.h5')
  oo2=load('o2.h5')
  oo3=load('o3.h5')

  assert oo1['a']==1
  assert oo1['b'][2]==5
  assert oo2.o1['a']==1
  assert oo2.o1['b'][2]==5
  assert oo3.o2.o1['c'].dtype==int

  del myclass

  print 'Expecting 3 warnings...'
  oo3=load('o3.h5')
  assert oo3['o2']['o1']['c'].dtype==int
  assert oo3['__class__']=='myclass'
  assert oo3['o2']['__class__']=='myclass'

  class mydict(dict):
    pass

  oo2=load('o2.h5',namespace={'dict':mydict})







