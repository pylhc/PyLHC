# Copyright 2010-2011 Riccardo De Maria

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

#TODO pages not supported
#TODO tables not supported
#TODO multi dimensional array in binary are not supported

# struct format
#     >: big
#     <: little
#     |: machine
#     x: pad byte (no data);
#     c:char;
#     b:signed byte;
#     B:unsigned byte;
#     h:short;
#     H:unsigned short;
#     i:int;
#     I:unsigned int;
#     l:long;
#     L:unsigned long;
#     f:float;
#     d:double.
#     s:string (array of char);
#     p:pascal string (with count byte).
#     P:an integer type that is wide enough to hold a pointer.
#     q:long long;
#     Q:unsigned long long

import StringIO
import gzip
import numpy as n
import struct



def iterheader(o):
  c=1
  while c:
    c=o.read(1)
    if c=='!':
      while c not in '\n\r':
        c=o.read(1)
    elif c not in '\n\t\r':
       yield c

def iterbinarydata(o):
  c=1
  while c:
    c=o.read(1)
    if c=='!':
      while c not in '\n\r':
        c=o.read(1)
      if c in '\n\r': # possibly a bug
        c=o.read(1)
    else:
      yield c

def readtoken(o):
  buf=[]
  for i in o:
    buf.append(i)
    if ''.join(buf[-4:])=='&end':
      yield ''.join(buf)
      buf=[]

def myreadline(o):
  buf=o.readline()
  while buf[0]=='!':
    buf=o.readline()
  return buf

def parseheader(l):
  t,data=l.split(' ',1)
  data=data.replace(' ','')
  data=[d.split('=') for d in data.split(',')]
  data.pop()
  data=dict(data)
  data['header']=t
  return data



sddstypes={
    'short' : 'i2',
    'long'  : 'i4',
    'llong' : 'u8',
    'string': 'S',
    'float': 'f4',
    'double': 'f8',
}

def myarray(fh,typ,count,endian):
#  print typ
  typ=n.dtype(endian+sddstypes.get(typ,typ))
  size=typ.itemsize*count
  ss=fh.read(size)
  if len(ss)==size:
#  print typ,count,size,repr(ss[:16])
    return n.fromstring(ss,dtype=typ,count=count)
  else:
    return None
#  return s

def mystruct(fh,typ,count,endian):
  typ='%s%d%s' % (endian,count,typ)
  size=struct.calcsize(typ)
  ss=fh.read(size)
  if len(ss)==size:
    return struct.unpack(typ,ss)
  else:
    return None

def mysplit(fh,count):
  out=[]
  while len(out)<count:
   l=fh.readline()
   out.extend(l.split())
  return out


class sddsdata(object):
  def __init__(self,filename,endian='little',full=True):
    self.filename=filename
    if filename.endswith('.gz'):
      self.fh=gzip.open(filename)
    else:
      self.fh=file(filename)
    try:
      self.version=self.fh.readline()
      fendian=self.fh.readline().split(' ')[1].split('-')[0]
      assert fendian in ['big','little']
      endian=fendian
    except AssertionError:
      print 'Warning sddsdata: forcing endianess to %s' % endian
      fh.seek(0)
      self.version=self.fh.readline()
    self.endian={'little':'<','big':'>'}[endian]
    # read headear
#    print 'read header'
    it=readtoken(iterheader(self.fh))
    header=[]
    for i in it:
     header.append(parseheader(i))
     if header[-1]['header']=='&data':
       break
    header2=[]; istable=True
    for i in header:
      if i['header']=='&column':
        if istable==True:
          header2.append({'header':'&table'})
          header2[-1]['columns']=[i]
          istable=False
        else:
          header2[-1]['columns'].append(i)
      else:
        header2.append(i)
    self.header=header2
#    print self.header
    # read data
    if full:
      self.fh.read(1)
      self.data=[]
      if self.header[-1]['mode']=='ascii':
        data={}
        self.data.append(data)
        for i in self.header:
          if 'type' in i:
            typ=i['type']
            typ=n.dtype(sddstypes.get(typ,typ))
            if i['header']=='&parameter':
              ss=myreadline(self.fh)
              d=n.array(ss,typ)
              i['value']=d
            elif i['header']=='&array':
              dims=map(int,myreadline(self.fh).split())
              i['shape']=dims
              cnt=reduce(lambda a,b:a*b,dims)
  #            ss=myreadline(self.fh)
  #            print dims, len(ss)
              d=n.array(mysplit(self.fh,cnt),typ).reshape(dims)
            data[i['name']]=d
      elif self.header[-1]['mode']=='binary':
        while 1:
          row=myarray(self.fh,'long',1,self.endian)
          if row is None:
            break
          data={}
          self.data.append(data)
          for i in self.header:
            if 'type' in i:
              typ=i['type']
              if i['header']=='&parameter':
                if typ=='string':
                  smax=0
                  subcount=myarray(self.fh,'long',1,self.endian)[0]
                  smax=subcount<smax and smax or subcount
                  d=mystruct(self.fh,'s',subcount,self.endian)[0]
                else:
                  d=myarray(self.fh,typ,1,self.endian)
              elif i['header']=='&array':
                count=myarray(self.fh,'long',1,self.endian)[0]
                if typ=='string':
                  d=[];smax=0
                  for r in range(count):
                    subcount=myarray(self.fh,'long',1,self.endian)[0]
                    smax=subcount<smax and smax or subcount
    #                d.append(myarray(self.fh,'>S1',subcount,self.endian)[0])
                    d.append(mystruct(self.fh,'s',subcount,self.endian)[0])
                  d=n.array(d,n.dtype('S%s'%smax))
                else:
                  d=myarray(self.fh,typ,count,self.endian)
              data[i['name']]=d
    self.fh.close()
  def __str__(self):
    out=['%s: %s' % (self.filename,self.version)]
    for i in self.header:
      oo=[]
      for n,k in i.items():
        if n is not 'header':
          oo.append('%s=%s' % (n,k))
      out.append(i['header'][1:]+' '+', '.join(oo))
    return '\n'.join(out)
  __repr__=__str__


