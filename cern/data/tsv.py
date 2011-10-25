# -*- coding: utf-8 -*-

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
from tsvObj import TSVelement
'''
.. module:: tsv
    :synopsis: Read in and manipulate data in a TSV format. 
    :todo: Not have single value arrays in [[],[]] format anymore (it's dumb!)
    :todo: Use underscore for local functions
 
     The idea is that this library could be generic enough to do
     manipulation on TSV objects for multiple projects that want to
     export data from TIMBER and handle them in python scripts.
     Future recommendation is that all data should be in numpy arrays,
     and timestamps in datetime arrays. This is not yet 100% implemented.
 
.. moduleauthor Yngve Inntjore Levinsen
'''
    
def date2num(dt):
    '''
     Stolen from matplotlib
    '''
    try:
        iter(dt)
    except TypeError: # not a list
        return _to_ordinalf(dt)
    else:
        return numpy.array([_to_ordinalf(val) for val in dt])

##
# @brief Stolen from matplotlib        
def _to_ordinalf(dt):
    base=dt.toordinal()
    if hasattr(dt, 'hour'):
        base += (dt.hour/24. + 
                    dt.minute/(60.*24) + 
                    dt.second/(3600.*24) + 
                    dt.microsecond/(3600.*24*1e6))
    return base

class TSVlist:
    '''
        TSV list object, give a TSV file name as input when initializing.
    '''
    ##
    # initialize object
    # 
    # This initialization includes reading the file
    # 
    # @param inputfile Filename of the TSV file
    def __init__(self,inputfile=''):
            ##  @var maxnamelength
            # maximum length of names in eg TFS output file
            self.maxnamelength=20 
            ## @var r
            # For radjustment in export formats like TFS
            # This variable gets readjusted as new variable names are added and removed
            self.r=[20,20]
            ## @var elementlist
            # List of TSV elements
            self.elementlist=[]
            ## @var parameters
            # list of parameters
            self.parameters=[]
            ## @var starttime
            # Start time in tsv file. 
            # Will be updated to earliest timestamp found in file
            self.starttime=datetime.datetime.now()
            if inputfile!='':
                    print "INFO: Reading file",inputfile
                    self.readFile(inputfile)
    
    # This defines the iterator for this class
    def __iter__(self):
        return iter(self.elementlist)

    
    
    def readFile(self,tsvfile):
            '''
             Read in a tab separated TSV file to object
             
             Args:
                tsvfile (str): Name of file
            '''
            infile=file(tsvfile,"r")
            thesevalues=[]
            ts=[]
            first=0
            for l in infile:
                    if len(l.split())>0:
                            if l.split()[0]=="VARIABLE:":
                                    if len(l.split()[1])>self.maxnamelength:
                                            self.maxnamelength=len(l.split()[1])
                                            self.r[1]=self.maxnamelength+2
                                    self.elementlist.append(TSVelement(name=l.split()[1]))
                            if l.split()[0]=="Unit:":
                                    self.elementlist[-1].setUnit(l.split()[1])
                            if l.split()[0]=="Description:":
                                    self.elementlist[-1].setDescription(l.split()[1:])
                            if re.search('20[0-1][0-9]', l[0:4])!=None: # year should fit to this regexp..
                                    d=self.elementlist[-1].appendReading(l)
                                    if d < self.starttime:
                                            self.starttime=d
                            if l.strip()=='>>> No data in the specified range <<<':
                                    print 'WARNING, variable without data: ',self.elementlist[-1].getName()
                                    print '        Deleting this variable from list'
                                    self.elementlist=self.elementlist[0:-1]
            infile.close()
    
    ##
    # @brief add an element to the list
    # 
    # @param name Name of new element
    # @param timearray Time array for new element
    # @param datavalues Data values corresponding to time array
    # @param unit Optionally give the units for the new element.
    # @param description Optionally give a description of the new element
    def addElement(self,name,timearray,datavalues,unit='',description=''):
            self.elementlist.append(TSVelement(name))
            self.elementlist[-1].setAllData(timearray,datavalues)
            if description:
                    self.elementlist[-1].setDescription(description)
            if unit:
                    self.elementlist[-1].setUnit(unit)

    ##
    # Iterates through the list of variables and delete empty ones
    def removeEmptyVars(self):
            i=0
            while i < len(self.elementlist):
                    if self.elementlist[i].getNumValues()==0:
                            print "INFO:",self.elementlist[i].getName(),'did not have any data, deleting'
                            del(self.elementlist[i])
                    else:
                            i=i+1
    ##
    # Print all values that have timestamp within timedelta of each other from each variable
    # 
    # @warning The timestamp is from the first variable, for the others the first entry within range is used
    # 
    # @param timedelta max difference in timestamp for measurements at "the same time"
    # @param filestream Optionally write the names to a stream
    def printValues(self,timedelta,filestream=''):
            tdlim=[datetime.timedelta(seconds=-timedelta),datetime.timedelta(seconds=timedelta)]
            for i in xrange(self.elementlist[0].getNumValues()):
                    s = str(self.elementlist[0].getTimestamps()[i]).rjust(self.r[0])+str(self.elementlist[0].getValue(i)).rjust(self.r[1])
                    for j in range(1,len(self.elementlist)):
                            for k in xrange(self.elementlist[j].getNumValues()):
                                    td=self.elementlist[j].getTimestamps()[k]-self.elementlist[0].getTimestamps()[i]
                                    if tdlim[0]<td<tdlim[1]: #measurements are within delta of each other..
                                            s+=str(self.elementlist[j].getValue(k)).rjust(self.r[1])
                                            break
                                    if k+1==self.elementlist[j].getNumValues(): #if this variable did not have any measurement in range
                                            s+="NaN".rjust(self.r[1])
            if filestream=='':
                    print s
            else:
                    filestream.write(s+'\n')
    ##
    # @brief print variable names with a # in front
    # 
    # The names are often long and confusing, hence the range to select the part you want.
    # If you want the full name, call the command with the range array [0,-1].
    # 
    # @param ranges array with two integers, defining the part of the names to be printed (default full name)
    # @param filestream Optionally write the names to a stream
    def printVarNames(self,ranges=[0,0],filestream=''):
            if len(self.elementlist[0].getName()[ranges[0]:ranges[1]])>14: #to make sure the line is readable...
                    self.r[1]=len(self.elementlist[0].getName())+2
            s="#".ljust(self.r[0])
            for i in self.elementlist:
                    if ranges[1]==0:
                            s+=i.getName().rjust(self.r[1])
                    else:
                            s+=i.getName()[ranges[0]:ranges[1]].rjust(self.r[1])
            if filestream=='':
                    print s
            else:
                    filestream.write(s+'\n')
    ##
    # Print data to ascii format
    # 
    # Plottable in e.g. gnuplot
    # @param timedelta see printValues
    # @param ranges see printVarNames
    # @param filename Name of ascii file
    # @see printValues, printVarNames
    def toAscii(self,filename='data.txt',timedelta=1,ranges=[0,0]):
            f=file(filename,'w')
            self.printVarNames(ranges)
            self.printValues(timedelta)
            f.close()
    ##
    # write table to file in tfs format
    # @param filename name of file
    # @param basecolumn select column where the time stamps in the tfs file are taken from
    def toTFS(self,filename,basecolumn=0):
            print "Writing TFS table"
            tfsfile=file(filename,"w")
            for l in self.parameters:
                    tfsfile.write("@ "+l[0].ljust(20)+l[1].rjust(15)+str(l[2]).rjust(20)+"\n")
            r=[19,15]
            s="*"+"Timestamp".rjust(self.r[0])
            for l in self.elementlist:
                    s+=l.getName().rjust(self.r[1])
            tfsfile.write(s+"\n")
            s="$"+"".rjust(self.r[0])
            for l in self.elementlist:
                    s+=l.getUnit().rjust(self.r[1])
            tfsfile.write(s+"\n")
            self.r[0]+=1
            y=self.createTableToStamps(basecolumn)
            for j in xrange(len(y[0,:])):
                    s = str(self.elementlist[basecolumn].getTimestamps()[j]).rjust(self.r[0])
                    for k in xrange(len(y[:,0])):
                            s+=str(y[k,j]).rjust(self.r[1])
                    tfsfile.write(s+"\n")
    ##
    # @brief Correlate all other variables to the given one
    # 
    # 
    # This function takes in one parameter name, and if it exists calculates the 
    # correlation between this one and all other values. It will rescale all other values
    # to have one entry at each time-stamp of the main parameter.
    # 
    # @param vname Name of variable to be correlated against
    # @param fname File name to write correlation results in (optional, no writing if not given)
    # @return The correlation vector
    #
    def correlateToValue(self, vname, fname=""):
            vnum=0
            while self.elementlist[vnum].getName()!=vname:
                    vnum+=1
                    if vnum == len(self.elementlist):
                            print "Did not find ",vname
                            return
            print "Found "+vname+", now correlating"
            y=self.createTableToStamps(vnum)
            w=numpy.corrcoef(y)
            ret=w[vnum,:]
            if fname!="":
                    rfile=file(fname,"w")
                    for x in xrange(len(self.elementlist)):
                            rfile.write(self.elementlist[x].getName().rjust(self.maxnamelength+3)+str(ret[x]).rjust(20)+"\n")
            return ret
    ##
    # Same as correlateToValue, but within a time window only
    # 
    #
    def correlateInTimeWindow(self, vname,tbegin,tend,fname=""):
            vnum=0
            while self.elementlist[vnum].getName()!=vname:
                    vnum+=1
                    if vnum == len(self.elementlist):
                            print "Did not find ",vname
                            return
            print "Found ",vname,"\t",self.elementlist[vnum].getName(),", now correlating"
            beginset=False
            endset=False
            for x in xrange(self.elementlist[vnum].getNumValues()):
                    if beginset==False and tbegin<=self.elementlist[vnum].getTimestamps()[x]:
                            istart=x
                            beginset=True
                    if endset==False and tend==self.elementlist[vnum].getTimestamps()[x]:
                            iend=x
                            endset=True
                    if endset==False and tend<self.elementlist[vnum].getTimestamps()[x]:
                            iend=x-1
                            endset=True
            if endset==False:
                    iend=self.elementlist[vnum].getNumValues()
            if beginset==False:
                    print "Wrong start time specified"
                    return
            y=self.createTableToStamps(vnum,istart,iend)
            w=numpy.corrcoef(y)
            ret=w[vnum,:]
            if fname!="":
                    rfile=file(fname,"w")
                    for x in xrange(len(self.elementlist)):
                            rfile.write(self.elementlist[x].getName().rjust(self.maxnamelength+3)+str(ret[x]).rjust(20)+"\n")
            return ret
    
    ##
    # @brief give a matrix with values from all parameters at given timestamps
    # 
    # Takes the timestamps from the given variable defined by vnum and calculates
    # the according values for all other parameters at those timestamps.
    # Used for e.g. correlateToValue and toTFS functions
    # 
    # @param vnum Number ID of given variable.
    # @param istart (optional) index to start from
    # @param iend (optional) index to end at
    # @return Matrix with values from all variables.
    # @todo this crashes for matrix elements..
    def createTableToStamps(self,vnum,istart=0,iend=0):
            if iend==0:
                    iend=self.elementlist[vnum].getNumValues()-1
            y=numpy.zeros((len(self.elementlist),iend-istart))
            mytimes=self.elementlist[vnum].getTimestamps()[istart:iend]
            for i in xrange(len(self.elementlist)):
                    if i==vnum:
                            newvals=self.elementlist[i].getValues()[istart:iend]
                            if len(newvals[0])==1:
                                    for j in xrange(len(newvals)):
                                            newvals[j]=newvals[j][0]
                    else:
                            tstampequal=True
                            for k in xrange(len(mytimes)):
                                    if mytimes[k]!=self.elementlist[i].getTimestamp(k):
                                            tstampequal=False
                                            break
                            if tstampequal:
                                    newvals=self.elementlist[i].getValues()[istart:iend]
                                    if len(newvals[0])==1:
                                            for j in xrange(len(newvals)):
                                                    newvals[j]=newvals[j][0]
                            else:
                                    newvals=self.getValAtTimes(self.elementlist[i].getName(), mytimes)
                    y[i,:]=numpy.array(newvals)
            return y
            
    ##
    # Add parameters to the object
    # 
    # @param name Name of parameter
    # @param unit Units of the parameter
    # @param value Value of the parameter
    def addParameter(self,name,unit,value):
            self.parameters.append([name,unit,value])
    ##
    # Plot a variable asfo the timestamp and save to file
    # 
    # @param vname variable name (must be exact)
    # @param fname filename of plot (must include suffix, e.g. .png)
    def plot(self,vname,fname):
            from matplotlib.dates import plot_date, savefig
            vnum=0
            while self.elementlist[vnum].getName()!=vname:
                    vnum+=1
                    if vnum == len(self.varnames):
                            print "Did not find ",vname
                            return
            print "found ",vname,"\t",self.elementlist[vnum].getName(),", now plotting"
            plot_date(date2num(self.elementlist[vnum].getTimestamps()), self.elementlist[vnum].getValues(), fmt='-')
            savefig(fname+".pdf")
    ##
    # @brief Smoothening a variable using interpolation
    # @param varname name of variable to be smoothed
    def smoothThisVariable(self,varname):
        print "Smoothening",varname
        vnum=0
        while self.elementlist[vnum].getName()!=varname:
            vnum+=1
            if vnum == len(self.elementlist):
                print "Did not find ",varname
                return
        for i in range(1,self.elementlist[vnum].getNumValues()-1):
            if self.elementlist[vnum].getValue(i)[0]!=0.0 and abs((self.elementlist[vnum].getValue(i)[0]-self.elementlist[vnum].getValue(i-1)[0])/self.elementlist[vnum].getValue(i)[0]) > 0.5:
                if abs((self.elementlist[vnum].getValue(i)[0]-self.elementlist[vnum].getValue(i+1)[0])/self.elementlist[vnum].getValue(i)[0]) > 0.5:
                    # smoothen this point:
                    dT=(date2num(self.elementlist[vnum].getTimestamps()[i+1])-date2num(self.elementlist[vnum].getTimestamps()[i-1]))
                    dT1=(date2num(self.elementlist[vnum].getTimestamps()[i])-date2num(self.elementlist[vnum].getTimestamps()[i-1]))
                if dT1 == 0.0:
                    self.elementlist[vnum].setValue(i,self.elementlist[vnum].getValue(i-1)[0])
                else:
                    self.elementlist[vnum].setValue(i,(self.elementlist[vnum].getValue(i+1)[0]-self.elementlist[vnum].getValue(i-1)[0])/dT*dT1+self.elementlist[vnum].getValue(i-1)[0])
    ##
    # @brief Removes a regexp from all variable names
    # 
    # Used to remove unneeded part of names. This is difficiult to do automatically
    # so instead you have to do it manually... A regexp can also be a string, but
    # watch out for special characters, e.g. . ^ $ [ ] ...
    # See: http://docs.python.org/library/re.html
    # 
    # @param s Regular expression to be removed
    # @param t Replacement string (optional, empty as default)
    # 
    def removeFromVarnames(self,s,t=''):
        import re
        p=re.compile(s)
        for x in xrange(len(self.elementlist)):
            self.elementlist[x].setName(p.sub(t,self.elementlist[x].getName()))
        self.setNamelength()
    ##
    # @brief Remove variables matching string
    # 
    # For example removeVariables("BLMEI") will remove all variables matching the string
    def removeVariables(self,s):
            for x in xrange(len(self.elementlist)):
                    if len(self.elementlist)==x:
                            break
                    while len(self.elementlist)>x and self.elementlist[x].getName().find(s)!=-1:
                            del(self.elementlist[x])
            self.setNamelength()
    ##
    # No documentation available
    def setNamelength(self):
        self.maxnamelength=0
        self.r[1]=20
        for n in self.elementlist:
            if self.maxnamelength<len(n.getName()):
                self.maxnamelength=len(n.getName())
                if self.r[1]< self.maxnamelength:
                    self.r[1]=self.maxnamelength+2
    
    ##
    # Return the index of the variables array at given timestamp
    # 
    # If no value set at exactly timestamp, the last entry before
    # the timestamp is used. If you turn off the optional flag 'before',
    # then the first entry after the timestamp will be chosen instead.
    # If there are no timestamps before or at the one you specified,
    # then the first value after will be used. The timestamp at the 
    # given index is returned as well.
    # 
    # @param varname Name of variable
    # @param timestamp Time stamp of type datetime
    # @param before optional parameter, default True
    # @return [indexvalue,timestamp]
    def getIndexAtTime(self,varname,timestamp,before=True):
        varid=self.findVarID(varname)
        i=0
        while self.elementlist[varid].getTimestamp(i)<timestamp and i<self.elementlist[varid].getNumValues():
                i+=1
        if i==self.elementlist[varid].getNumValues(): #we hit the end of the array
            return [i-1,self.elementlist[varid].getTimestamp(i-1)]
        if self.elementlist[varid].getTimestamp(i)==timestamp: #we had data at exactly that time
            return [i,self.elementlist[varid].getTimestamp(i)]
        if before and i>0:
            return [i-1,self.elementlist[varid].getTimestamp(i-1)]
        else: # you wanted the first data AFTER the time you specified
            return [i,self.elementlist[varid].getTimestamp(i)]
    ##
    # Return all values for the given variable.
    # @param varname Name of variable
    def getValues(self,varname):
        varid=self.findVarID(varname)
        if varid==None or self.elementlist[varid].getNumValues()==0:
            print "WARNING: Could not get data for",varname
            return None
        else:
            return self.elementlist[varid].getValues()
                
    ##
    # Return the value of the variable at given timestamp
    # 
    # If no value set at exactly timestamp, linear interpolation is done
    # A numpy.array element will be returned
    # @param varname Name of variable
    # @param timestamp Time stamp of type datetime
    # @return [numpy.array]
    def getValAtTime(self,varname,timestamp):
        varid=self.findVarID(varname)
        if varid==None or self.elementlist[varid].getNumValues()==0:
            print "WARNING: Could not return value at given timestamp"
            return None
        j=0
        while self.elementlist[varid].getTimestamp(j)<timestamp:
            j+=1
            if j==self.elementlist[varid].getNumValues()==j:
                # in this case we just give the last value...
                return numpy.array(self.elementlist[varid].getValue(j-1))
        if self.elementlist[varid].getTimestamp(j)==timestamp:
            return self.elementlist[varid].getValue(j)
        if self.elementlist[varid].interpolatable():
            retvala=numpy.array(self.elementlist[varid].getValue(j-1))
            retvalb=numpy.array(self.elementlist[varid].getValue(j))
            dta=(timestamp-self.elementlist[varid].getTimestamp(j-1)).microseconds
            dtb=(self.elementlist[varid].getTimestamp(j)-timestamp).microseconds
            return (retvala*dtb+retvalb*dta)/(dta+dtb)
        else: #means we should repeat last given number
            return numpy.array(self.elementlist[varid].getValue(j-1))
    
    ##
    # @brief give a variable a new unit
    # 
    # @param varname name of variable
    # @param newunit the new unit you want to set for variable varname
    def setUnit(self,varname,newunit):
        varid=self.findVarID(varname)
        if varid!=None:
            self.elementlist[varid].setUnit(newunit)
        
    ##
    # Return the value of the variable at given timestamps
    # 
    # If no value set at exactly timestamp, linear interpolation is done
    # A numpy.array element will be returned
    # @param varname Name of variable
    # @param timestamps Time stamp of type datetime
    def getValAtTimes(self,varname,timestamps):
        varid=self.findVarID(varname)
        if varid==None:
            print 'ERROR: Could not find variable',varname
            return None
        if len(self.elementlist[varid].getValue(0))>1: #only a necessary check if it is a matrix (like eg BWS data)
            for i in xrange(self.elementlist[varid].getNumValues()):
                if len(self.elementlist[varid].getValue(0))!=len(self.elementlist[varid].getValue(i)):
                    print 'ERROR: getValAtTimes() only works for even NxM matrices or vectors'
                    print ' ',varname,' has uneven column lengths..'
                    return None
        myy=self.elementlist[varid].getValues()
        mynx=date2num(timestamps)
        myx=date2num(self.elementlist[varid].getTimestamps())
        myytmp=numpy.zeros(len(myy))
        if len(myy[0])>1:
            out=numpy.zeros((len(timestamps),len(myy[0])))
        for j in xrange(len(myy[0])):
            for i in xrange(len(myy)):
                myytmp[i]=myy[i][j]
            if self.elementlist[varid].interpolatable():
                outtmp=numpy.interp(mynx,myx,myytmp)
            else: #this variable should not be interpolated, but repeated until new value arises:
                outtmp=numpy.zeros(len(timestamps))
                k=0
                for i in xrange(len(mynx)):
                    while k+1<len(myx) and mynx[i]>myx[k]:
                        k+=1
                    if mynx[i]<myx[k] and k>0:
                        k-=1
                    outtmp[i]=myytmp[k]
            if len(myy[0])==1:
                out=outtmp
            else:
                for i in xrange(len(mynx)):
                    out[i,j]=outtmp[i]
        return out
    
    ##
    # @brief returns maximum value 
    # @param maxvalue if different from zero, maximum value not considered garbage
    def getMaxValue(self,varname,trange=[],maxvalue=0):
        varid=self.findVarID(varname)
        if varid==None:
            print("WARNING: Could not find "+varname)
            return None
        return self.elementlist[varid].getMaxValue(trange,maxvalue)
            
        

    ##
    # @brief Returns an array of timestamps when variable had value equal to value
    # 
    def getTimeWhenVal(self,varname,value):
        varid=self.findVarID(varname)
        if varid:
            vindex=self.elementlist[varid]._getValIndex(value)
            if vindex!=None:
                return self.elementlist[varid].getTimestamp(vindex)
            else:
                if sys.flags.debug:
                    print('INFO: '+str(value)+' not found in '+varname)
                return None
        
    ##
    # @brief Export tsv table to a HDF5 tree
    # 
    # Exports the vectors to a ROOT file. 
    # In order to do so you need to define which variable the time stamps should be taken from.
    # If you do not, the variable with most entries is chosen.
    # 
    # @param hfile Name of HDF5 file object that you want the data written in.
    # @param name Name of group. It will be placed in root of file.
    # @param description Description of group.
    # @param basecolumn Optional choice of timestamp column from which other data arae aligned to
    # @param timestamps Optionally provide an array of timestamps which you want the data aligned to
    # @return A list containing the tree
    # 
    def exportToHDF5(self,hfile,name="LHCdata",description="Data read from TIMBER",basecolumn=-1,timestamps=None):
        import h5py
        if len(self.elementlist)==0:
            if sys.flags.debug:
                print "WARNING: no data in current group, not exporting to HDF5"
            return 1
        if timestamps!=None and basecolumn!=-1:
            print 'WARNING: You specified both a list of timestamps AND a basecolumn'
            print ' The timestamp array you specified will be used'
        if timestamps==None:
            if sys.flags.debug:
                print 'INFO: You did not specify timestamp for this export to HDF5'
            if basecolumn==-1: # use the one with the highest number of entries
                basecolumn=0
                gnv=self.elementlist[basecolumn].getNumValues()
            for i in range(1,len(self.elementlist)):
                if gnv<self.elementlist[i].getNumValues():
                    basecolumn=i
                    gnv=self.elementlist[basecolumn].getNumValues()
            timestamps=self.elementlist[basecolumn].getTimestamps()
        
        thegroup=hfile.create_group(name)
        thegroup.attrs['Description']=description
        if timestamps!=-1:
            ts = numpy.array([timestamps[i].microsecond*1e-6 + timegm(timestamps[i].timetuple()) for i in xrange(len(timestamps))])
            if sys.flags.debug:
                print 'DBG',numpy.shape(ts),ts[0], len(timestamps)
            vals=thegroup.create_dataset('Timestamps',numpy.shape(ts),'=f8',ts,compression='gzip', compression_opts=1)
            vals.attrs['Units']='seconds'
            vals.attrs['Description']='UNIX timestamp, ie. seconds since 1.1.1970'
        for el in self.elementlist:
            if timestamps==-1:
                myv=self.getValues(el.getName())
            else:
                myv=self.getValAtTimes(el.getName(),timestamps)
            vals=thegroup.create_dataset(el.getName(),numpy.shape(myv),'=f4',myv,compression='gzip', compression_opts=1)
            vals.attrs['Units']=el.getUnit()
            vals.attrs['Description']=el.getDescription()
        return 0
    
    ##
    # @brief Export tsv table to a root tree
    # 
    # Exports the vectors to a ROOT file. 
    # In order to do so you need to define which variable the time stamps should be taken from.
    # If you do not, the variable with most entries is chosen.
    # 
    # Expert option: timestamps==-1 means that the timestamps will NOT be added to the tree
    # 
    # @todo this fails for matrices
    # @return A list containing the tree
    # 
    def exportToROOT(self,fillnumber,name="LHCdata",description="Data read from TIMBER",basecolumn=-1,timestamps=None):
        from ROOT import gROOT, TTree, TObject,TMatrixD
                
        if len(self.elementlist)==0:
            if sys.flags.debug:
                print "WARNING: no data in current group, not exporting to ROOT"
            return 1
        #
        # Seems someone is trying to write None-vars into the object, 
        # I need to check for that :(
        #  
        j=0
        while j < len(self.elementlist):
            if self.elementlist[j].haveNone():
                print 'WARNING:',self.elementlist[j].getName(),'had a None-type in the values.\n\tI have to delete it before exporting to ROOT.'
                del(self.elementlist[j])
            else:
                j+=1
        if timestamps and basecolumn!=-1:
            print('WARNING: You specified both a list of timestamps AND a basecolumn')
            print('         The timestamp array you specified will be used')
        if not timestamps or timestamps==-1:
            if sys.flags.debug:
                print('INFO: You did not specify timestamp for this export to ROOT')
            if basecolumn==-1: # use the one with the highest number of entries
                basecolumn=0
                gnv=self.elementlist[basecolumn].getNumValues()
                for i in range(1,len(self.elementlist)):
                        if gnv<self.elementlist[i].getNumValues():
                                basecolumn=i
                                gnv=self.elementlist[basecolumn].getNumValues()
                    
            exportablevals=[self.getValAtTimes(self.elementlist[i].getName(),self.elementlist[basecolumn].getTimestamps()) for i in xrange(len(self.elementlist)) ]
        else:
            gnv=len(timestamps)
            exportablevals=[self.getValAtTimes(self.elementlist[i].getName(),timestamps) for i in xrange(len(self.elementlist))]
        vtree=TTree(name,description)
        fnum=array('i',[fillnumber])
        trickarrays=[array('d', [0.])]
        vtree.Branch('fillnumber',fnum,'fillnumber/I')
        blist=[]
        if timestamps!=-1:
            blist.append(vtree.Branch('Timestamps',trickarrays[0],'Timestamps/D'))
        for elem in self.elementlist:
            if len(elem.getValue(0))==1:
                trickarrays.append(array('f', [0.])) # tricking ROOT
                blist.append(vtree.Branch(elem.getROOTName(),trickarrays[-1],elem.getROOTName()+'/'+elem.getROOTformat()))
            else:
                trickarrays.append(array('f',len(elem.getValue(0))*[0.]))
                blist.append(vtree.Branch(elem.getROOTName(),trickarrays[-1],elem.getROOTName()+'['+str(len(elem.getValue(0)))+']/'+elem.getROOTformat()))
        for i in xrange(gnv):
            # unix timestamp with microsecond accuracy (root stores a double from this, so about 10ms accuracy)
            if timestamps==None:
                trickarrays[0][0]= self.elementlist[basecolumn].getTimestamp(i).microsecond*1e-6 + timegm(self.elementlist[basecolumn].getTimestamp(i).timetuple()) 
                jstart=1 # we include timestamps, so actual data should start at index 1 in trickarrays..
            elif timestamps!=-1:
                trickarrays[0][0]= timestamps[i].microsecond*1e-6 + timegm(timestamps[i].timetuple()) 
                jstart=1 # we include timestamps, so actual data should start at index 1 in trickarrays..
            else:
                jstart=0 # we do NOT include timestamps, so actual data should start at index 0 in trickarrays..
            for j in xrange(len(self.elementlist)):
                if len(elem.getValue(0))==1:
                    trickarrays[j+jstart][0]=exportablevals[j][i]
                else:
                    for k in xrange(len(self.elementlist[j].getValue(0))):
                        trickarrays[j+1][k]=exportablevals[j][i][k]
            vtree.Fill()
        return [vtree]
    ##
    # @brief Return the index of a given element.
    # 
    # @param varname the name of the element
    def findVarID(self,varname):
        for i in xrange(len(self.elementlist)):
            if self.elementlist[i].getName()==varname:
                return i
        if sys.flags.debug:
            print "INFO: Could not find variable:", varname
        return None
    
    ##
    # @brief Returns an array of all the timestamps held by the element
    # @param varname Name of element
    def getTimestamps(self,varname):
        for el in self.elementlist:
            if el.getName()==varname:
                return el.getTimestamps()
        if sys.flags.debug:
            print "INFO: Could not find variable:", varname
        return None
        
