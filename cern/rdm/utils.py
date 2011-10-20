from numpy import log10,pi,sqrt,exp
import gzip, StringIO
import re

def pythonname(string):
  string=string.replace('[','')
  string=string.replace(']','')
  string=string.replace('.','_')
  string=string.replace('$','_')
  string=string.lower()
  return string

def numtostr(n,np=3):
  """Convert a number in a string where . has a fixed position
  >>> for i in range(-6,10):
  ...   print numtostr( .1234*10**i),numtostr( .1234567*10**i,np=6)
  ...   print numtostr(-.1234*10**i),numtostr(-.1234567*10**i,np=6)
   123.400e-09  123.456700e-09
  -123.400e-09 -123.456700e-09
     1.234e-06    1.234567e-06
    -1.234e-06   -1.234567e-06
    12.340e-06   12.345670e-06
   -12.340e-06  -12.345670e-06
   123.400e-06  123.456700e-06
  -123.400e-06 -123.456700e-06
     1.234e-03    1.234567e-03
    -1.234e-03   -1.234567e-03
    12.340e-03   12.345670e-03
   -12.340e-03  -12.345670e-03
   123.400e-03  123.456700e-03
  -123.400e-03 -123.456700e-03
     1.234        1.234567    
    -1.234       -1.234567    
    12.340       12.345670    
   -12.340      -12.345670    
   123.400      123.456700    
  -123.400     -123.456700    
     1.234e+03    1.234567e+03
    -1.234e+03   -1.234567e+03
    12.340e+03   12.345670e+03
   -12.340e+03  -12.345670e+03
   123.400e+03  123.456700e+03
  -123.400e+03 -123.456700e+03
     1.234e+06    1.234567e+06
    -1.234e+06   -1.234567e+06
    12.340e+06   12.345670e+06
   -12.340e+06  -12.345670e+06
   123.400e+06  123.456700e+06
  -123.400e+06 -123.456700e+06
  """
  n=float(n)
  if abs(n)>0:
    l=log10(abs(n))
    if l<0:
      l=int(l)
      o=(l-1)//3*3
      fmt='%%%d.%dfe%+03d' % (np+5,np,o)
      n=n/10**o
#    fill space of digits
#    elif -3<=l and l< 0: fmt='%%12.%df' % (np+4)
#    elif  0<=l and l< 3: fmt='%%12.%df' % (np+4)
#    elif -3<=l and l< 0: fmt='%%11.%df ' % (np+3)
    elif  0<=l and l< 3: fmt='%%%d.%df    ' % (np+5,np)
    elif  3<=l:
      l=int(l)
      o=(l)//3*3
      fmt='%%%d.%dfe%+03d' % (np+5,np,o)
      n=n/10**o
  else:
    fmt='%4.0f.'+' '*(np+4)
  return fmt % n


def gt(a,b):
  if a<b:
    return (b-a)**2
  else:
    return 0.

def lt(a,b):
  if a>b:
    return (b-a)**2
  else:
    return 0.

def cmp(a,b,c):
  if a<b:
    return (b-a)**2
  elif a>c:
    return (a-c)**2
  else:
    return 0.

def eq(a,b):
  return (a-b)**2

def rng(a,b,c):
  return (a>b) & (a<c)

def mystr(d,nd):
  """truncate a number or a string with a fix number of chars
  >>> for d in [ 0.443,'stre',1.321E-4,-3211233]: print mystr(d,12)
   443.000e-03
  stre        
   132.100e-06
    -3.211e+06
  """
  if hasattr(d,'__coerce__'):
    d=numtostr(d,nd-9)
  return ('%%-%d.%ds' % (nd,nd)) % d



def myflatten(lst):
  for elem in lst:
    if type(elem) in (tuple, list):
      for i in myflatten(elem):
        yield i
    else:
      yield elem

import time
def timeit(s,set=False):
  if set:
    timeit.mytime=time.time()
  print '%8.3f %s' % (time.time()-timeit.mytime,s)
timeit.mytime=0


def myopen(fn):
  try:
    if fn.endswith('.gz'):
      return gzip.open(fn)
    else:
      return open(fn)
  except IOError:
    return StringIO.StringIO(fn)


def no_dots(x):
    return x.group().replace('.','_')


madname=re.compile(r'([a-z_][a-z_0-9\.]*)')

def pyname(n):
  n=n.lower()
  n=madname.sub(no_dots,n)
  n.replace('^','**')
  if n=='from':
    n='From'
  return n





if __name__=='__main__':
  import doctest
  doctest.testmod()




