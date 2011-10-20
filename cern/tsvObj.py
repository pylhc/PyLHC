
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

import datetime
from calendar import timegm
import numpy
from array import array
import re
import sys


##
# TSV element
# Should be used by tsvClass only.
class TSVelement:
    ##
    # @brief Initialization of new element
    # @param name name of element
    def __init__(self,name):
        ## 
        # The name of the element
        self._name=name
        ## 
        # The array of values
        # This will in many cases be set to be a numpy array once filled
        self.values=[]
        ##
        # A dictionary containing the properties of this element
        # e.g. units, description...
        self._properties={}
        ## 
        # Timestamps corresponding to each value
        # These are in datetime-format, 
        # UTC timestamps
        self._timestamps=[]
    ##
    # Return name of element
    def getName(self):
        return self._name
    ##
    # Return value list of element
    def getValues(self):
        return self.values[:]
    ##
    # Returns index when value first appeared
    # If value not found, returns None
    def _getValIndex(self,value):
        if value in self.values:
            return self.values.index(value)
        else:
            return None
    ##
    # Returns the maximum value in element.
    # Optional argument trange which is
    # a list of two datetime objects.
    def getMaxValue(self,trange=[],maxvalue=0):
        if not trange==[]:
            if not len(trange)==2:
                raise ValueError,'Wront timestamp range'
            elif type(trange[0])!=datetime.datetime or type(trange[1])!=datetime.datetime:
                raise TypeError, 'Wrong timestamp types'
            maxval=min(self.values)[:]
            for j in xrange(len(self.values)):
                if trange[0]<self._timestamps[j]<trange[1]:
                    if self.values[j]>maxval:
                        if not maxvalue or self.values[j]<maxvalue:
                            maxval=self.values[j][:]
            return maxval
        return max(self.values)[:]
    ##
    # Return value at index i
    def getValue(self,i):
        if i<-self.getNumValues() or self.getNumValues()<=i:
            raise ValueError("You tried to access element %i of %s, but it only has %i elements"
                         % (i,self._name,self.getNumValues()))
        return self.values[i]
    ##
    # Returns number of values
    def getNumValues(self):
        return len(self.values)
    ##
    # Return units of element
    def getUnit(self):
        if 'unit' in self._properties:
            return self._properties['unit']
        else:
            if sys.flags.debug:
                print("WARNING: No unit info for "+self.getName())
        return ''
    ##
    # Return description of element
    def getDescription(self):
        if 'description' in self._properties:
            return self._properties['description']
        else:
            if sys.flags.debug:
                print("WARNING: No description for "+self.getName())
        return ''
    ##
    # Return timestamps of element
    def getTimestamps(self):
        return self._timestamps[:] # note [:] means we make a copy of the list
    ##
    # Return timestamp number i of element
    def getTimestamp(self,i):
        if i>=self.getNumValues():
            raise ValueError('You tried to extract timestamp number %i, number of elements in %s is %i'
                         % (i,self.getName(),self.getNumValues()))
        return self._timestamps[i]
    def haveNone(self):
        if None in self.values or None in self._timestamps:
            return True
        else:
            return False
    ##
    # Set name of element
    def setName(self,name):
        self._name=name
    ##
    # Set the units of the values in element
    def setUnit(self,unit):
        if type(unit)!=str:
            print("ERROR: Units must be of type str")
            return 1
        self._properties['unit']=unit
        return 0
    ##
    # Set value[i] to given value
    def setValue(self,i,value):
        self.values[i]=value
    ##
    # Set the description of the element
    def setDescription(self,description):
        # description is usually returned in the form of
        # a list containing each word of the description...
        if type(description)==list:
            dsc=''
            for word in description:
                dsc+=' '+word
            description=dsc[1:]
        self._properties['description']=description
    ##
    # Returns the ROOT format of the values
    # Possible values are 'I', 'F', and 'C'
    def getROOTformat(self):
        dummyvar=numpy.zeros(1) #don't know of any other way of getting the numpy.type float64 out...
        if len(self.values)==0:
            return None
        if type(self.values[0][0])==type(0):
            return 'I'
        elif type(self.values[0][0])==type(0.0):
            return 'F'
        elif type(self.values[0][0])==type(''):
            return 'C'
        elif type(self.values[0][0])==type(dummyvar[0]):
            return 'F'
    ##
    # @brief removes the previous data if any, and adds new data as specified.
    def setAllData(self,timestamp,data):
        if len(timestamp)!=len(data):
            print 'Error, you need the same amount of timestamps and data'
            return 0
        self._timestamps=timestamp
        dummyvar=numpy.zeros(1)
        for d in data:
            if type(d)==type([]) or type(d)==type(dummyvar):
                self.values.append(d)
            else:
                self.values.append([d])

    ##
    # @brief Read one measured data with time stamp
    # @warning Assumed that values are split by TABS!
    # @todo read of datetime object is correct, but can probably be faster?
    # @return The time stamp
    def appendReading(self,line):
        q=line.split("\t")
        d=datetime.datetime(year=int(q[0][0:4]),month=int(q[0][5:7]),day=int(q[0][8:10]),hour=int(q[0][11:13]),minute=int(q[0][14:16]),second=int(q[0][17:19]),microsecond=int(q[0][20:23])*1000)
        for i in range(1,len(q)):
            if q[i].strip()=='null':
                q[i]=0.0
            else:
                q[i]=float(q[i])
        self.values.append(q[1:])
        self._timestamps.append(d)
        return d
    ##
    # @brief Returns the element name in a ROOT friendly format
    # 
    # Replaces .'s and :'s from the names...
    def getROOTName(self):
        return (self._name.replace(':','_')).replace('.','_')
    ##
    # @brief Same as appendReading, but read several lines from stream
    def appendLines(self,line,fstream,numlines):
        notfinished=True
        firsttime=datetime.datetime.now()
        count=0
        q=line.split("\t")
        print len(q)
        vappend=self.values.append #defining these should give speedup (?)
        tappend=self._timestamps.append
        dtime=datetime.datetime
        while notfinished:
            q=line.split("\t")
            d=dtime(year=int(q[0][0:4]),month=int(q[0][5:7]),day=int(q[0][8:10]),hour=int(q[0][11:13]),minute=int(q[0][14:16]),second=int(q[0][17:19]),microsecond=int(q[0][20:23])*1000)
            for i in range(1,len(q)):
                q[i]=float(q[i])
            vappend(q[1:])
            tappend(d)
            if d<firsttime:
                firsttime=d
            if count==numlines-1:
                notfinished=False
            else:
                count+=1
                line=fstream.readline()
        return firsttime
    ##
    # @brief returns true if this variable can be interpolated
    # 
    # In case it is e.g. a mode selector etc, you should repeat last value, not
    # interpolate... Like HX:BMOD
    # 
    def interpolatable(self):
        nointerplist=['HX','NO_BUNCHES','CollisionPattern']
        for n in nointerplist:
            if n in self._name:
                return False
        return True
