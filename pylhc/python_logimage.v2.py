import sys
import os
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pylab as plt
#plt.ion()
import numpy as np
#import matplotlib.animation as animation
#import matplotlib.patches as patches
import subprocess
import time
import pyjapc
import signal
import time
import pickle
import datetime as dt
#export PATH="/user/bdisoft/operational/bin/Python/PRO/bin:$PATH

 
# Create a PyJapc instance with selector SCT.USER.ALL
# INCA is automatically configured based on the timing domain you specify here
#japc = pyjapc.PyJapc(selector="LHC.USER.ALL", incaAcceleratorName="LHC", noSet=True )

CycleName="LHC.USER.ALL"
INCAacc='LHC'
noSetFlag=True

japc=pyjapc.PyJapc(selector=CycleName,incaAcceleratorName=INCAacc,noSet=noSetFlag)
japc=pyjapc.PyJapc(selector=CycleName,noSet=noSetFlag)
japc.rbacLogin()

''' 
B1_image = japc.getParam('LHC.BSRTS.5R4.B1/Image')
B2_image = japc.getParam('LHC.BSRTS.5L4.B2/Image')

B1_bunch=B1_image['lastAcquiredBunch']
B2_bunch=B2_image['lastAcquiredBunch']
B1_CameraAOI=B1_image['acquiredImageRectangle']
B2_CameraAOI=B2_image['acquiredImageRectangle']
B1_acquiredIMG=B1_image['imageSet'].reshape(B1_CameraAOI[3],B1_CameraAOI[2])
B2_acquiredIMG=B2_image['imageSet'].reshape(B2_CameraAOI[3],B2_CameraAOI[2])

B1_acqTime=B1_image['acqTime']
B2_acqTime=B2_image['acqTime']

B2_projPosX=B1_image['projPositionSet1']
B2_projPosY=B1_image['projPositionSet2']
B2_projPosX=B2_image['projPositionSet1']
B2_projPosY=B2_image['projPositionSet2']
'''
##########################################
def parse_timestamp(thistime):
    accepted_time_input_format = ['%Y-%m-%d %H:%M:%S.%f','%Y-%m-%d %H:%M:%S','%Y-%m-%d_%H:%M:%S.%f','%Y-%m-%d_%H:%M:%S','%Y/%m/%d %H:%M:%S.%f']
    for fmat in accepted_time_input_format:
        try:
            dtobject=dt.datetime.strptime(thistime,fmat)
            return dtobject
        except ValueError:
            pass
    timefmatstring=''
    for fmat in accepted_time_input_format:
        timefmatstring=timefmatstring+'\"'+fmat+'\" ,   '
    sys.tracebacklimit = 0
    raise ValueError('No appropriate input format found for start time of scan (-s).\n ---> Accepted input formats are:   '+timefmatstring)
##########################################
def convert_to_data_output_format(dtobject): ### function to help write output from datetime objects in standard format throughout code
    output_timestamp=dtobject.strftime('%Y-%m-%d-%H-%M-%S.%f')
    return output_timestamp
##########################################
aquesitions_per_file=100
j=0
t=0
while(True):
    time.sleep(0.7)
    print (j,t)
    B1_image = japc.getParam('LHC.BSRTS.5R4.B1/Image')
    B2_image = japc.getParam('LHC.BSRTS.5L4.B2/Image')
    if t==0:
        allB1data=[]
        allB2data=[]
        B1_IMGtime=B1_image['acqTime']
        B2_IMGtime=B2_image['acqTime']
        B1_IMGtime_dt = parse_timestamp(B1_IMGtime)
        B2_IMGtime_dt = parse_timestamp(B2_IMGtime)
        B1_IMGtime_st =  convert_to_data_output_format(B1_IMGtime_dt)
        B2_IMGtime_st =  convert_to_data_output_format(B2_IMGtime_dt)

#        B1_date=B1_IMGtime.partition(' ')[0]
#        B1_time=B1_IMGtime.partition(' ')[2]
#        B2_date=B2_IMGtime.partition(' ')[0]
#        B2_time=B2_IMGtime.partition(' ')[2]


    allB1data.append(B1_image)
    allB2data.append(B2_image)
    t+=1
    if t==(aquesitions_per_file):
        j+=1
        f1name='data_BSRT_B1_'+B1_IMGtime_st+'.dat'
        f2name='data_BSRT_B2_'+B2_IMGtime_st+'.dat'
        f1=open(f1name,'wb')
        f2=open(f2name,'wb')
        pickle.dump(allB1data,f1)
        pickle.dump(allB2data,f2)
        f1.close()
        f2.close()
        os.system('gzip '+f1name)
        os.system('gzip '+f2name)
        t=0



        
                     

# Close the RBAC session
japc.rbacLogout()

'''
print (B2_bunch)
print (len(B2_acquiredIMG))
print (len(B2_acquiredIMG[0]))
print (B2_acqTime)

print (len(B2_projPosX))
print (len(B2_projPosY))
#for j in range(len(B2_acquiredIMG)):
#    print (B2_acquiredIMG[j])           
'''
