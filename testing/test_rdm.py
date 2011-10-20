#-------------------------------------------------------------------------------
# This file is part of PyMad.
# 
# Copyright (c) 2011, CERN. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# 	http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#-------------------------------------------------------------------------------
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
