
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

import os
import time


from datetime import datetime
from dateutil.tz import gettz,tzlocal
from dateutil.parser import parse



myzones={'bnl' : 'America/New_York',
       'cern': 'Europe/Zurich',
       'fnal': 'America/Chicago',
       'lbl' : 'America/Los_Angeles',
       'Z'   : 'UTC'}

myfmt={'myf': '%Y-%m-%d--%H-%M-%S--%z',
       'myh': '%Y-%m-%d %H:%M:%S %z',
       'myl': '%Y-%m-%d %H:%M:%S.SSS',
       'rfc': '%a, %d %b %Y %H:%M:%S %z',
       'epoch' :'%s',
       'iso' : '%Y%m%dT%H%M%S%z',
       'cernlogdb' : '%Y%m%d%H%M%SCET',
       }

def parsedate(s=None,tz=None):
  """Read a string in the '2010-06-10 00:00:00.123 TZ?' format and return
  the unix time."""
  stz=gettz(myzones.get(tz))
  if s is None:
    dt=datetime.now(tz)
  else:
    dt=parse(s,fuzzy=True)
  epoch=time.mktime(dt.timetuple())+dt.microsecond / 1000000.0
  return epoch


def parsedate_myl(s):
  """Read a string in the '2010-06-10 00:00:00.123 TZ?' format and return
  the unix time."""
  stime='00:00:00'
  ssec=0
  stz=gettz()
  parts=s.split(' ')
  sdate=parts[0]
  if len(parts)>1:
    stime=parts[1]
  if len(parts)==3:
    stz=parts[2]
  stimes=stime.split('.')
  if len(stimes)==2:
    stime=stimes[0]
    ssec=int(float('0.'+stimes[1])*1e6)
  t=time.strptime('%s %s'%(sdate,stime),'%Y-%m-%d %H:%M:%S')
  stz=gettz(myzones.get(stz))
  dt=datetime(t[0],t[1],t[2],t[3],t[4],t[5],ssec,stz)
  epoch=time.mktime(dt.timetuple())+dt.microsecond / 1000000.0
  return epoch

def dumpdate(epoch=None,fmt='myl',tz='local'):
  """Return a date string from epoch
  predefined formats are
    myf: %Y-%m-%d--%H-%M-%S--%z
    myh: %Y-%m-%d %H:%M:%S %z
    myl: %Y-%m-%d %H:%M:%S.SSS
    rfc: %a, %d %b %Y %H:%M:%S %z
    epoch :%s
    iso : %Y%m%d%H%M%S%z
  predefined timezone are:
    bnl  :  America/New_York ,
    cern :  Europe/Zurich ,
    fnal :  America/Chicago ,
    lbl  :  America/Los_Angeles ,
    Z    :  UTC'
  """
  if epoch is None:
    epoch=time.time()
  fmt=myfmt.get(fmt,fmt)
  tz=gettz(myzones.get(tz))
  dt=datetime.fromtimestamp(epoch).replace(tzinfo=gettz()).astimezone(tz)
  s=dt.strftime(fmt)
  if 'SSS' in s:
    s=s.replace('SSS',('%06d'%dt.microsecond)[:3])
  return s


def test():
  epoch=time.time()
  print epoch
  print dumpdate(epoch)
  print parsedate(dumpdate(epoch))
  epoch=parsedate('2010-08-23 10:54:12.123456')
  print dumpdate(epoch,fmt='myh',tz='bnl')

if __name__=='__main__':
  import sys
  args=' '.join(sys.argv[1:])
  opt={}
  for i in args.split():
    if i.startswith('-'):
      opt[i[1]]=i[2:]
    else:
      opt['date']+=' '+i
  tzout=opt.get('o')
  fmt=opt.get('f','rfc')
  date=opt.get('date',dumpdate(time.time()))
  print dumpdate(parsedate(date),fmt=fmt,tz=tzout)

