# Copyright 2011 Riccardo de Maria

#    This file is part of PyLHC.
#
#    PyLHC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    PyLHC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with PyLHC.  If not, see <http://www.gnu.org/licenses/>.

import re
import os
import time

import wx

import matplotlib.pyplot as _p
import numpy as _n
from utils import mystr as _mystr

from pydataobj import dataobj



lglabel={
    'betx':    r'$\beta_x$',
    'bety':    r'$\beta_y$',
    'dx':    r'$D_x [m]$',
    'dy':    r'$D_y [m]$',
    'mux':    r'$\mu_y$',
    'muy':    r'$\mu_y$',
    }

axlabel={
    's':       r'$s [m]$',
    'ss':       r'$s [m]$',
    'betx':    r'$\beta [m]$',
    'bety':    r'$\beta [m]$',
    'mux':    r'$\mu/(2 \pi)$',
    'muy':    r'$\mu/(2 \pi)$',
    'dx':    r'$D [m]$',
    'dy':    r'$D [m]$',
    'x':    r'$co [m]$',
    'y':    r'$co [m]$',
    }


def _mylbl(d,x): return d.get(x,r'$%s$'%x)



class qdplot(object):
  @staticmethod
  def wx_autoupdate(*pls):
    def callback(*args):
      for pl in pls:
        pl.update()
      wx.WakeUpIdle()
    wx.EVT_IDLE(wx.GetApp(),callback)
  @staticmethod
  def wx_stopupdate():
    wx.EVT_IDLE.Unbind(wx.GetApp(),wx.ID_ANY,wx.ID_ANY,self._callback)


  def __init__(self,t,x='',yl='',yr='',idx=slice(None),
      clist='k r b g c m',lattice=None,newfig=True,pre=None,
              ):
    yl,yr,clist=map(str.split,(yl,yr,clist))
#    timeit('Init',True)
    self.color={}
    self.left=None
    self.right=None
    self.lattice=None
    self.pre=None
    self.t,self.x,self.yl,self.yr,self.idx,self.clist=t,x,yl,yr,idx,clist
    for i in self.yl+self.yr:
      self.color[i]=self.clist.pop(0)
      self.clist.append(self.color[i])
    if newfig is True:
      self.figure=_p.figure()
    elif newfig is False:
      self.figure=_p.gcf()
      self.figure.clf()
    else:
      self.figure=newfig
      self.figure.clf()
    if lattice:
      self.lattice=self._new_axes()
#      self.lattice.set_autoscale_on(False)
      self.lattice.yaxis.set_visible(False)
    if yl:
      self.left=self._new_axes()
#      self.left.set_autoscale_on(False)
      self.left.yaxis.set_label_position('left')
      self.left.yaxis.set_ticks_position('left')
    if yr:
      self.right=self._new_axes()
#      self.right.set_autoscale_on(False)
      self.right.yaxis.set_label_position('right')
      self.right.yaxis.set_ticks_position('right')
#    timeit('Setup')
    self.run()
    if lattice:
      self.lattice.set_autoscale_on(False)
    if yl:
      self.left.set_autoscale_on(False)
    if yr:
      self.right.set_autoscale_on(False)
#    timeit('Update')
  def _new_axes(self):
    if self.figure.axes:
      ax=self.figure.axes[-1]
      out=self.figure.add_axes(ax.get_position(),
          sharex=ax, frameon=False)
    else :
      #adjust plot dimensions
      out=self.figure.add_axes([.12,.10,.6,.8])
    return out

  def __repr__(self):
    return object.__repr__(self)

  def _trig(self):
    print 'optics trig'
    self.run()

  def update(self):
    if hasattr(self.t,'reload'):
      if self.t.reload():
        self.run()
    return self

#  def _wx_callback(self,*args):
#    self.update()
#    wx.WakeUpIdle()
#
#  def autoupdate(self):
#    if _p.rcParams['backend']=='WXAgg':
#      wx.EVT_IDLE.Bind(wx.GetApp(),wx.ID_ANY,wx.ID_ANY,self._wx_callback)
#    return self
#
#  def stop_update(self):
#    if _p.rcParams['backend']=='WXAgg':
#      wx.EVT_IDLE.Unbind(wx.GetApp(),wx.ID_ANY,wx.ID_ANY,self._callback)
#
#  def __del__(self):
#    if hasattr(self,'_callback'):
#      self.stop_update()

  def run(self):
#    print 'optics run'
    self.ont=self.t
    self.xaxis=getattr(self.ont,self.x)
    is_ion=_p.isinteractive()
    _p.interactive(False)
    self.lines=[]
    self.legends=[]
#    self.figure.lines=[]
#    self.figure.patches=[]
#    self.figure.texts=[]
#    self.figure.images = []
    self.figure.legends = []

    if self.lattice:
      self.lattice.patches=[]
      self._lattice(['k0l','kn0l','angle'],"#a0ffa0",'Bend h')
      self._lattice(['ks0l'],"#ffa0a0",'Bend v')
      self._lattice(['kn1l','k1l'],"#a0a0ff",'Quad')
      self._lattice(['hkick'],"#e0a0e0",'Kick h')
      self._lattice(['vkick'],"#a0e0e0",'Kick v')
      self._lattice(['kn2l','k2l'],"#e0e0a0",'Sext')
    if self.left:
      self.left.lines=[]
      for i in self.yl:
        self._column(i,self.left,self.color[i])
    if self.right:
      self.right.lines=[]
      for i in self.yr:
        self._column(i,self.right,self.color[i])
    ca=self.figure.gca()
    ca.set_xlabel(_mylbl(axlabel,self.x))
    ca.set_xlim(min(self.xaxis[self.idx]),max(self.xaxis[self.idx]))
    self.figure.legend(self.lines,self.legends,'upper right')
    ca.grid(True)
#    self.figure.canvas.mpl_connect('button_release_event',self.button_press)
    self.figure.canvas.mpl_connect('pick_event',self.pick)
    _p.interactive(is_ion)
    self.figure.canvas.draw()

  def pick(self,event):
    pos=_n.array([event.mouseevent.x,event.mouseevent.y])
    name=event.artist.elemname
    prop=event.artist.elemprop
    value=event.artist.elemvalue
    print '\n %s.%s=%s' % (name, prop,value),

#  def button_press(self,mouseevent):
#    rel=_n.array([mouseevent.x,mouseevent.y])
#    dx,dy=self.pickpos/rel
#    print 'release'
#    self.t[self.pickname][self.pickprop]*=dy
#    self.t.track()
#    self.update()
  
  #def _get_sl(ont):
    #'''
     #YIL suggestion: set s=l or opposite in case
     #only one of them are present in the table..
    #'''
    #if 's' not in ont:
        #if 'l' not in ont:
            #raise ValueError('You need at least "s" or "l" in the table')
        #else:
            #s=ont.l
    #else:
        #s=ont.s
    #if 'l' in ont:
        #l=ont.l
    #else:
        #l=ont.s
    #return s,l

  def _lattice(self,names,color,lbl):
#    timeit('start lattice %s' % names,1)
    vd=0
    sp=self.lattice
    s=self.ont.s
    l=self.ont.l
    #s,l=_get_sl(self.ont)
    for i in names:
      myvd=self.ont.__dict__.get(i,None)
      if myvd is not None:
        vdname=i
        vd=myvd[self.idx]+vd
    if vd is not 0:
      m=_n.abs(vd).max()
      if m>1E-10:
        c=_n.where(abs(vd) > m*1E-4)[0]
        if len(c)>0:
          if _n.all(l[c]>0):
            vd[c]=vd[c]/l[c]
            m=abs(vd[c]).max()
          vd[c]/=m
          if self.ont._is_s_begin:
            plt=self.lattice.bar(s[c],vd[c],l[c],picker=True)
          else:
            plt=self.lattice.bar(s[c]-l[c],vd[c],l[c],picker=True)
          _p.setp(plt,facecolor=color,edgecolor=color)
          if plt:
            self.lines.append(plt[0])
            self.legends.append(lbl)
          row_names=self.ont.name
          for r,i in zip(plt,c):
            r.elemname=row_names[i]
            r.elemprop=vdname
            r.elemvalue=getattr(self.ont,vdname)[i]
        self.lattice.set_ylim(-1.5,1.5)
#    timeit('end lattice')

  def _column(self,name,sp,color):
    fig,s=self.figure,self.xaxis
    y=self.ont(name)[self.idx]
    bxp,=sp.plot(s,y,color,label=_mylbl(lglabel,name))
    sp.set_ylabel(_mylbl(axlabel,name))
    self.lines.append(bxp)
    self.legends.append(_mylbl(lglabel,name))
    sp.autoscale_view()





from numpy import sqrt, sum, array, sin, cos,   dot, pi
from numpy.linalg import inv

from utils import pyname
from namedtuple import namedtuple

def rng(x,a,b):
  "return (x<b) & (x>a)"
  return (x<b) & (x>a)

infot=namedtuple('infot','idx betx alfx mux bety alfy muy')
from cern.rdm.data import tfs
import gzip
import os

class optics(dataobj):
  _is_s_begin=False
  _name_char=16
  _entry_char=12
  _entry_prec=3
  @classmethod
  def open(cls,fn):
    try:
      if fn.endswith('tfs.gz'):
        return cls(tfs.load(gzip.open(fn)))
      elif fn.endswith('tfs'):
        if os.path.exists(fn):
          return cls(tfs.open(fn))
        elif os.path.exists(fn+'.gz'):
          return cls(tfs.load(gzip.open(fn+'.gz')))
      raise IOError
    except IOError:
      raise IOError,"%s does not exists or wrong format" % fn
  def __init__(self,data={},idx=False):
    self.update(data)
    self._fdate=0
    if idx:
      try:
        self._mkidx()
      except KeyError:
        print 'Warning: error in idx generation'
  def reload(self):
    if 'filename' in self._data:
       fdate=os.stat(self.filename).st_ctime
       if fdate>self._fdate:
         self._data=tfs.open(self.filename)
         self._fdate=fdate
         print '%s reload' % self.filename
         return True
    return False
  def _mkidx(self):
    name=map(pyname,list(self.name))
    self.idx=dataobj()
    fields=infot._fields[1:]
    for i,name in enumerate(name):
      data=[i] + map(lambda x: self[x][i],fields)
      setattr(self.idx,name,infot(*data))

  def pattern(self,regexp):
    c=re.compile(regexp,flags=re.IGNORECASE)
    out=[c.search(n) is not None for i,n in enumerate(self.name)]
    return _n.array(out)
  __floordiv__=pattern

  def dumplist(self,rows=None,cols=None):
    if rows is None:
      rows=_n.ones(len(self.name),dtype=bool)
    elif isinstance(rows,str):
      rows=self.pattern(rows)
    rows=_n.where(rows)[0]

    if cols is None:
      colsn=self._data.keys()
      cols=[getattr(self,n.lower()) for n in colsn]
    if isinstance(cols,str):
      colsn=cols.split()
      cols=[self(n) for n in cols.split()]

    out=[]
    rowfmt=['%%-%d.%ds' % (self._name_char,self._name_char)]
    rowfmt+=['%%-%d.%ds' % (self._entry_char,self._name_char)] * len(colsn)
    rowfmt=' '.join(rowfmt)
    out.append(rowfmt % tuple(['names'] + colsn  ) )
    for i in rows:
      v=[ self.name[i] ]+ [ _mystr(c[i],self._entry_char) for c in cols ]
      out.append(rowfmt %  tuple(v))
    return out
  def dumpstr(self,rows=None,cols=None):
    return '\n'.join(self.dumplist(rows=rows,cols=cols))
  def show(self,rows=None,cols=None):
    print self.dumpstr(rows=rows,cols=cols)
  def twissdata(self,location,data):
    idx=_n.where(self.pattern(location))[0][-1]
    out=dict(location=location)
    for name in data.split():
      vec=self.__dict__.get(name)
      if vec is None:
        out[name]=0
      else:
        out[name]=vec[idx]
    out['sequence']=self.param.get('sequence')
    return out
  def range(self,pat1,pat2):
    """ return a mask relative to range"""
    try:
      id1=_n.where(self.pattern(pat1))[0][-1]
    except IndexError:
      raise ValueError,"%s pattern not found in table"%pat1
    try:
      id2=_n.where(self.pattern(pat2))[0][-1]
    except IndexError:
      raise ValueError,"%s pattern not found in table"%pat2
    out=_n.zeros(len(self.name),dtype=bool)
    if id2>id1:
      out[id1:id2+1]=True
    else:
      out[id1:]=True
      out[:id2+1]=True
    return out

  def plot(self,yl='',yr='',x='s',idx=slice(None),
      clist='k r b g c m',lattice=True,newfig=True,pre=None,
          ):
    out=qdplot(self,x=x,yl=yl,yr=yr,idx=idx,lattice=lattice,newfig=newfig,clist=clist,pre=pre)
#    self._target.append(out)
    return out

  def plotbeta(self,**nargs):
    return self.plot('betx bety','dx dy',**nargs)

  def plotcross(self,**nargs):
    return self.plot('x y','dx dy',**nargs)

  def plottune(self,lbl=''):
    _p.title(r"${\rm Tune} \quad {\rm vs} \delta$")
    _p.xlabel("$\delta$")
    _p.ylabel("Fractional tune")
    tt=r'$%s \rm{%s}$'
    _p.plot(self.deltap,self.q1-self.q1.round(),label=tt %('Q_x',lbl))
    _p.plot(self.deltap,self.q2-self.q2.round(),label=tt %('Q_y',lbl))
    qx=(self.q1-self.q1.round())[abs(self.deltap)<1E-15][0]
    qy=(self.q2-self.q2.round())[abs(self.deltap)<1E-15][0]
    _p.text(0.0,qx,r"$Q_x$")
    _p.text(0.0,qy,r"$Q_y$")
    _p.grid(True)
    _p.legend()

  def plotbetabeat(self,t1,dp='0.0003'):
    _p.title(r"$\rm{Beta beat: 1 - \beta(\delta=%s)/\beta(\delta=0)}$" % dp)
    _p.ylabel(r"$\Delta\beta/\beta$")
    _p.xlabel(r"$s$")
    _p.plot(self.s,1-t1.betx/self.betx,label=r'$\Delta\beta_x/\beta_x$')
    _p.plot(self.s,1-t1.bety/self.bety,label=r'$\Delta\beta_y/\beta_y$')
    _p.grid(True)
    _p.legend()

  def plotw(self,lbl=''):
    _p.title(r"Chromatic function: %s"%lbl)
  # _p.ylabel(r"$w=(\Delta\beta/\beta)/\delta$")
    _p.ylabel(r"$w$")
    _p.xlabel(r"$s$")
    _p.plot(self.s,self.wx,label=r'$w_x$')
    _p.plot(self.s,self.wy,label=r'$w_y$')
    _p.grid(True)
    _p.legend()

  def plotap(t,ap,nlim=30,ref=7,newfig=True,**nargs):
    t.ss=ap.s
    t.n1=ap.n1
    pl=t.plot(x='ss',yl='n1',newfig=newfig,**nargs)
    pl.figure.gca().plot(t.ss,t.ss*0+ref)
    pl.figure.gca().set_ylim(0,nlim)
    pl.figure.canvas.draw()
    return pl




  def maxbetx(f):
    return f.betx+f.alfx**2/f.betx/abs(f.kn1l/f.l)
  def maxbety(f):
    return f.bety+f.alfy**2/f.bety/abs(f.kn1l/f.l)
  def chromx(f):
    if not hasattr(f,'kn1l'):
      f.kn1l=f.k1l
    return -sum(f.kn1l*f.betx)/4/pi
  def chromy(f):
    if not hasattr(f,'kn1l'):
      f.kn1l=f.k1l
    return sum(f.kn1l*f.bety)/4/pi
  def ndx(t):
    return t.dx/sqrt(t.betx)
  def ndpx(t):
    return t.dpx*sqrt(t.betx)+t.dx/sqrt(t.betx)*t.alfx
  def alphac(t):
    return sum(t('dx*kn0l'))/sum(t.l)
  def drvterm(t,p=0,q=0,l=0,m=0):
    dv=t.betx**(abs(p)/2.)*t.bety**(abs(q)/2.)
    dv*=_n.exp(+2j*pi*((p-2*l)*t.mux+(q-2*m)*t.muy))
    return dv
  def gammatr(t):
    af=t._alphac()
    if af>0:
      return sqrt(1/af)
    else:
      return -sqrt(-1/af)
  def transferMatrix(self,i1=0,i2=-1,plane='x'):
    """Return the transfer matrix from position i1 to position i2
       see Y.Lee 2.68 pag 53 for definition
    """
    B2=self.normMat(i2,plane=plane)
    B1=self.normMat(i1,plane=plane)
    psi=2*pi*(self['mu'+plane][i2] - self['mu'+plane][i1])
    R=array([[cos(psi),sin(psi)],[-sin(psi),cos(psi)]])
    return dot(dot(B2,R),inv(B1))
  def normMat(self,i,plane='x'):
    beta=self['bet'+plane][i]
    alpha=self['alf'+plane][i]
    return array([[sqrt(beta),0],[-alpha/sqrt(beta),1/sqrt(beta)]])










