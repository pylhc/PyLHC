#!/usr/bin/python

from distutils.core import setup

# this script can be used to 
# aid people that want to install the 
# modules created by the project.. 

setup(name='PyLHC',
      version='0.1',
      description='Python Library for LHC packages',
      requires=['ROOT','numpy'],
      license='BSD',
      author='Yngve Inntjore Levinsen',
      author_email='Yngve.Inntjore.Levinsen@cern.ch',
      packages = ['PyLHC']
     )
