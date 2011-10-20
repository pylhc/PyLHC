#!/usr/bin/python

from distutils.core import setup

## 
# Various tools available for CERN stuff

setup(name='cern',
      version='0.1',
      description='Python Library for LHC packages',
      requires=['ROOT','numpy'],
      license='BSD',
      author='Yngve Inntjore Levinsen',
      author_email='Yngve.Inntjore.Levinsen@cern.ch',
      packages = ['cern']
     )
