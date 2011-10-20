#!/usr/bin/python

# Copyright 2011 Yngve Inntjore Levinsen

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
