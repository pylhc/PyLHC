
# Copyright 2011 CERN

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

'''
..author Yngve Inntjore Levinsen
'''
from cern.rdm import optics
import unittest,os
import pylab as p

class TestRDM(unittest.TestCase):
    
    def setUp(self):
        self.opt=optics.optics.open('test.tfs')
    
    # It's a bit surprising that this doesn't happen by itself.. Hmmm...
    def tearDown(self):
        del self.opt
        
    def test_pltbeta(self):
        self.opt.plotbeta()
        p.clf()
    
    def test_plot(self):
        self.opt.plot('betx bety','dx')
        p.clf()

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRDM)
    unittest.TextTestRunner(verbosity=1).run(suite)
