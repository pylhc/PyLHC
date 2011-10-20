
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

import matplotlib.pyplot as pl


class ScatterPlot(object):
  def __init__(self,sub):
    self.sub=sub
    self.lines=[]
    self.style='s'
  def show(self,x,y):
    self.lines=self.sub.plot(x,y,'ro',picker=3)
  def draw(self,x,y):
    for l in self.lines:
      l.set_xdata(x)
      l.set_ydata(y)
      l.recache()
  def hide(self):
    while self.lines:
      l=self.lines.pop()
      l.axes.lines.remove(l)


class TextPlot(object):
  def __init__(self,sub):
    self.sub=sub
    self.lines=[]
    self.style='t'
    self.texts=[]
    self.lines=[]
  def show(self,x,y):
    self.lines=[]
    if len(x)>len(self.texts):
      self.texts+=['none']*(len(x)-len(self.texts))
    else:
      self.texts=self.texts[:len(x)]
    ax=pl.gca()
    xform=ax.xaxis.get_major_formatter()
    yform=ax.yaxis.get_major_formatter()
    for t,xx,yy in zip(self.texts,x,y):
      if t=='none':
        t='(%s,%s)'%(xform(xx),yform(yy))
      tt=self.sub.text(xx,yy,t)
      self.lines.append(tt)
  def draw(self,x,y):
    for tt,xx,yy,t in zip(self.lines,x,y,self.texts):
      tt.set_x(xx)
      tt.set_x(yy)
      tt.set_text(t)
  def hide(self):
    while self.lines:
      tt=self.lines.pop()
      tt.axes.texts.remove(tt)


class HorPlot(object):
  def __init__(self,sub):
    self.sub=sub
    self.lines=[]
    self.style='h'
  def show(self,x,y):
    xa,xb=self.sub.get_xlim()
    for yy in y:
      self.lines.extend(self.sub.plot([xa,xb],[yy,yy],'k-'))
    self.sub.set_xlim(xa,xb)
  def draw(self,x,y):
    xa,xb=self.sub.get_xlim()
    for l,yy in zip(self.lines,y):
      l.set_xdata([xa,xb])
      l.set_ydata([yy,yy])
      l.recache()
    self.sub.set_xlim(xa,xb)
  def hide(self):
    while self.lines:
      l=self.lines.pop()
      l.axes.lines.remove(l)

class VertPlot(object):
  def __init__(self,sub):
    self.sub=sub
    self.lines=[]
    self.style='v'
  def show(self,x,y):
    ya,yb=self.sub.get_ylim()
    for xx in x:
      self.lines.extend(self.sub.plot([xx,xx],[ya,yb],'k-'))
    self.sub.set_ylim(ya,yb)
  def draw(self,x,y):
    ya,yb=self.sub.get_ylim()
    for l,xx in zip(self.lines,x):
      l.set_xdata([xx,xx])
      l.set_ydata([ya,yb])
      l.recache()
    self.sub.set_ylim(ya,yb)
  def hide(self):
    while self.lines:
      l=self.lines.pop()
      l.axes.lines.remove(l)

class Picker(object):
  """ Usage:
    p=Picker(gcf())

    Mouse:
      button2: add point
      button1: move point
      button3: delete point

    print 'data:', p.data
    print 'eventd:', p.evs

    clear(): clear events
    show(): show artists
    hide(): remove artists
  """
  def __init__(self,fig=None,style='svh'):
    self.data=[]
    self.evs=[]
    self.figs=[]
    self.artists=[]
    self.pickevent=None
    self._move=False
    self._first=True
    self.hidden=False
    self.style=style
    if fig is not None:
      self.connect(fig)
  def mkartists(self,fig):
    for sub in fig.axes:
        self.artists.append(ScatterPlot(sub))
        self.artists.append(VertPlot(sub))
        self.artists.append(HorPlot(sub))
        self.artists.append(TextPlot(sub))
  def delartists(self,fig):
    self.artits=[a for a in self.artists if a.sub.figure!=fig]
  def connect(self,fig):
    h3=fig.canvas.mpl_connect('pick_event',self._on_pick)
    h1=fig.canvas.mpl_connect('button_press_event',self._on_press)
    h2=fig.canvas.mpl_connect('key_press_event',self._on_keypress)
    h4=fig.canvas.mpl_connect('button_release_event',self._on_release)
    h5=fig.canvas.mpl_connect('motion_notify_event',self._on_motion)
    self.figs.append([fig,h1,h2,h3,h4,h5])
    self.mkartists(fig)
    return self
  def get_figs(self):
    return [fig for fig,h1,h2,h3,h4,h5 in self.figs]
  def disconnect(self,n=None):
    if n is None:
      while self.figs:
        self.disconnect(0)
    else:
      fig,h1,h2,h3,h4,h5=self.figs.pop(n)
      fig.canvas.mpl_disconnect(h1)
      fig.canvas.mpl_disconnect(h2)
      fig.canvas.mpl_disconnect(h3)
      fig.canvas.mpl_disconnect(h4)
      fig.canvas.mpl_disconnect(h5)
      self.delartists(fig)
  def __del__(self):
    self.disconnect()
  def __repr__(self):
    return "Bucket.data\n"+repr(self.data)
  def show(self,style=None):
    if style is None:
      style=self.style
    else:
      self.style=style
    if self.data:
      x,y=zip(*self.data)
      for a in self.artists:
        if a.style in style:
          a.show(x,y)
      for fig in self.get_figs():
        fig.canvas.draw()
  def hide(self,style=None):
    if style is None:
      style=self.style
    for a in self.artists:
      if a.style in style:
        a.hide()
    for fig in self.get_figs():
      fig.canvas.draw()
  def _draw(self):
      x,y=zip(*self.data)
      for a in self.artists:
        a.draw(x,y)
      for fig in self.get_figs():
        fig.canvas.draw()
  def clear(self):
    self.data=[]
    self.hide()
  def _on_press(self,ev):
    #print 'press'
    if self.pickevent!=ev and ev.button==2:
      self.data.append([ev.xdata,ev.ydata])
      self.evs.append(ev)
      self.hide()
      self.show()
  def disable_press_event(self,ev):
    self.pickevent=ev.mouseevent
  def _on_pick(self,ev):
    #print 'pick'
    xdata=ev.artist.get_xdata()
    ydata=ev.artist.get_ydata()
    self.ind=ev.ind[0]
    if ev.mouseevent.button==1:
    #print xdata,ydata,self.data[ind]
      self.disable_press_event(ev)
      self._move=True
    elif ev.mouseevent.button==2:
      self.disable_press_event(ev)
      x,y=ev.mouseevent.xdata,ev.mouseevent.ydata
      self.data.insert(self.ind,[x,y])
      self._draw()
    elif ev.mouseevent.button==3:
      del self.data[self.ind]
      self.hide()
      self.show()
  def _on_motion(self,ev):
    if self._move:
      x,y=ev.xdata,ev.ydata
      self.data[self.ind]=[x,y]
      #print self.data
      self._draw()
  def _on_release(self,event):
    if self._move:
      self._move=False
  def _on_keypress(self,event):
    if event.key=='z':
      if self.hidden:
        self.show()
        self.hidden=False
      else:
        self.hide()
        self.hidden=True
  def get_derivative(self):
    x,y=map(_np.array,zip(*self.data))
    return diff(x)/diff(y)








