import os, sys
import pytimber
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tfs
from pytimber import pagestore
import datetime
import scipy.odr
import scipy
from constants.general import PLANES, PLANE_TO_HV, PROTON_MASS, LHC_NORM_EMITTANCE
from constants.forced_da_analysis import *





def main(directory:str, fill:int, beam:int, plane:str, energy:int = 6500, bunch_id:int = None):
    gamma = energy / PROTON_MASS  # E = gamma * m0 * c^2
    beta = np.sqrt(1 - (1 / gamma**2))
    emittance = LHC_NORM_EMITTANCE / (beta * gamma)









if __name__ == '__main__':
    FILL_NUMBER = 7391
    BEAM = 1
    BUNCH_ID = 0
    ANALYSISDIR = '/media/jdilly/Storage/Repositories/Gui_Output/2019-08-09/LHCB1/Results/b1_amplitude_det_vertical_all'
    energy = 6500  # GeV
    PLANE = PLANES[0]
    main(directory=ANALYSISDIR, fill=FILL_NUMBER, beam=BEAM, plane=PLANE)






def get_dfs_from_timber(fill: int, beam: int, bunch: int):
    # open Logging DB for specific fill and db for storing data
    db = pytimber.LoggingDB()
    filldata = db.getLHCFillData(fill)
    t1 = filldata['startTime']
    t2 = filldata['endTime']

    intensity_key = f'LHC.BCTFR.A6R4.B{beam:d}:BEAM_INTENSITY'
    x, y = db.get(intensity_key, t1, t2)[intensity_key]
    fbct_df = pd.DataFrame(data=y, index=x, columns=[INTENSITY])
    bsrt_df = {p:None for p in PLANES}

    for plane in PLANES:
        plane_hv = PLANE_TO_HV[plane]
        bunch_emittance_key = f'LHC.BSRT.5R4.B{beam:d}:BUNCH_EMITTANCE_{plane_hv}'
        col_emittance = f"{EMITTANCE}{plane}"
        col_norm_emittance = f"{NORM_EMITTANCE}{plane}"

        x, y = db.get(bunch_emittance_key, t1, t2)[bunch_emittance_key]
        if bunch is None:
            bunch = y.sum(axis=0).argmax()  # first not-all-zero column

        df = pd.DataFrame(index=x, columns=[f"{emittance}{suff}"
                                            for emittance in (col_emittance, col_norm_emittance)
                                            for suff in ("", f"_{CLEAN}", f"_{MEAN}", f"_{RMS}")])

        df[col_norm_emittance] = y[:, bunch]

        # remove entries with zero emittance as unphysical
        df = df[df[f"{col_norm_emittance}_{CLEAN}"] != 0

        bsrt_h_df['EMITTANCE_H_AV7'] = bsrt_h_df['EMITTANCE_H_CLEAN'].rolling(window=7, center=True).mean()
        bsrt_h_df['EMITTANCE_H_STD7'] = bsrt_h_df['EMITTANCE_H_CLEAN'].rolling(window=7, center=True).std()

        bsrt_h_df['NEMITTANCE_H_AV7'] = bsrt_h_df['EMITTANCE_H_CLEAN'].rolling(window=7, center=True).mean()*10**-6/(beta*gamma)
        bsrt_h_df['NEMITTANCE_H_STD7'] = bsrt_h_df['EMITTANCE_H_CLEAN'].rolling(window=7, center=True).std()*10**-6/(beta*gamma)

        bsrt_h_df['NEMITTANCE_H'] = bsrt_h_df['EMITTANCE_H_CLEAN']*10**-6/(beta*gamma)


    x , y = database.get( 'LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_V', t1, t2 )[ 'LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_V' ]
    bsrt_v_df = pd.DataFrame( data=y[:,BUNCH_ID], index=x, columns = ['EMITTANCE_V'] )

    # remove entries with zero emittance as unphysical
    bsrt_v_df['EMITTANCE_V_CLEAN'] = np.where( bsrt_v_df['EMITTANCE_V'] < 1., bsrt_v_df['EMITTANCE_V'].shift(1), bsrt_v_df['EMITTANCE_V'] )

    bsrt_v_df['EMITTANCE_V_AV7'] = bsrt_v_df['EMITTANCE_V_CLEAN'].rolling(window=7, center=True).mean()
    bsrt_v_df['EMITTANCE_V_STD7'] = bsrt_v_df['EMITTANCE_V_CLEAN'].rolling(window=7, center=True).std()

    bsrt_v_df['NEMITTANCE_V_AV7'] = bsrt_v_df['EMITTANCE_V_CLEAN'].rolling(window=7, center=True).mean()*10**-6/(beta*gamma)
    bsrt_v_df['NEMITTANCE_V_STD7'] = bsrt_v_df['EMITTANCE_V_CLEAN'].rolling(window=7, center=True).std()*10**-6/(beta*gamma)

    bsrt_v_df['NEMITTANCE_V'] = bsrt_v_df['EMITTANCE_V_CLEAN']*10**-6/(beta*gamma)
    x , y = database.get( 'LHC.BSRT.5R4.B1:FIT_SIGMA_H', t1, t2 )[ 'LHC.BSRT.5R4.B1:FIT_SIGMA_H' ]
    y = [ [np.mean(a),np.std(a)] for a in y]
    bsrt_h_sigma_df = pd.DataFrame( data=y, index=x, columns = ['EMITTANCE_H', 'EMITTANCE_H_STD'] )

    x , y = database.get( 'LHC.BSRT.5R4.B1:FIT_SIGMA_V', t1, t2 )[ 'LHC.BSRT.5R4.B1:FIT_SIGMA_V' ]
    y = [ [np.mean(a),np.std(a)] for a in y]
    bsrt_v_sigma_df = pd.DataFrame( data=y, index=x, columns = ['EMITTANCE_V', 'EMITTANCE_V_STD'] )

    bsrt_v_sigma_df['NEMITTANCE_V'] = bsrt_v_sigma_df['EMITTANCE_V'].rolling(window=3, center=True).mean()*10**-6/(beta*gamma)
    bsrt_v_sigma_df['NEMITTANCE_V_STD'] = bsrt_v_sigma_df['EMITTANCE_V'].rolling(window=3, center=True).std()*10**-6/(beta*gamma)

    x, yI = database.get( 'LHC.BWS.5R4.B1H.APP.IN:EMITTANCE_NORM', t1, t2 )[ 'LHC.BWS.5R4.B1H.APP.IN:EMITTANCE_NORM' ]
    x, yO = database.get( 'LHC.BWS.5R4.B1H.APP.OUT:EMITTANCE_NORM', t1, t2 )[ 'LHC.BWS.5R4.B1H.APP.OUT:EMITTANCE_NORM' ]
    yI = np.array([ [np.mean(a), np.std(a)] for a in yI ])
    yO = np.array([ [np.mean(a), np.std(a)] for a in yO ])
    y = np.array([yI[:,0] ,yI[:,1] ,yO[:,0] ,yO[:,1] ]).transpose()
    bws_h_df = pd.DataFrame( index=x, data=y, columns=['EMITTANCE_IN_H', 'EMITTANCE_IN_H_STD', 'EMITTANCE_OUT_H', 'EMITTANCE_OUT_H_STD'] )


    x, yI = database.get( 'LHC.BWS.5R4.B1V.APP.IN:EMITTANCE_NORM', t1, t2 )[ 'LHC.BWS.5R4.B1V.APP.IN:EMITTANCE_NORM' ]
    x, yO = database.get( 'LHC.BWS.5R4.B1V.APP.OUT:EMITTANCE_NORM', t1, t2 )[ 'LHC.BWS.5R4.B1V.APP.OUT:EMITTANCE_NORM' ]
    yI = np.array([ [np.mean(a), np.std(a)] for a in yI ])
    yO = np.array([ [np.mean(a), np.std(a)] for a in yO ])
    y = np.array([yI[:,0] ,yI[:,1] ,yO[:,0] ,yO[:,1] ]).transpose()
    bws_v_df = pd.DataFrame( index=x, data=y, columns=['EMITTANCE_IN_V', 'EMITTANCE_IN_V_STD', 'EMITTANCE_OUT_V', 'EMITTANCE_OUT_V_STD'])
# load kicks

kicks_df = tfs.read( ANALYSISDIR+'/getkickac.out' )
kicks_df.drop(columns=['DPP', 'QX', 'QXRMS', 'QY', 'QYRMS', 'NATQX', 'NATQXRMS', 'NATQY', 'NATQYRMS'] ,inplace=True )

print( kicks_df.tail(5))


def timetonms(timestamp):
    timeformat = '%Y-%m-%d %H:%M:%S.%f'
    dtobject = datetime.datetime.strptime(timestamp, timeformat)
    time = int(dtobject.strftime('%s'))*1000  + dtobject.microsecond/1000
    return time

kicks_df["timestamp"] = [ datetime.datetime.utcfromtimestamp(t) for t in kicks_df['TIME'] ]

# plot Beam intensity for both Beams
fig, ax = plt.subplots( ncols=1, nrows=1, sharex=True, sharey=False, figsize=(10,10) )

ax.plot(bsrt_v_df['EMITTANCE_V'],  # Actual BSRT measurement
        color='red',
        marker='o',
        linestyle='None',
        markersize=10,
        label='__nolegend__')

ax.errorbar(bsrt_v_df.index,   # Averaged measurement
            bsrt_v_df['EMITTANCE_V_AV7'],
            yerr = bsrt_v_df['EMITTANCE_V_STD7'],
            color='darkred',
            linewidth=4,
            label='Vertical emittance from BSRT')

ax.errorbar(bws_v_df.index,   # Wire Scanner (In)
            bws_v_df['EMITTANCE_IN_V'],
            color='darkorange',
            label='Vertical emittance from BWS',
            markersize=15,
            fmt='o'
            )

ax.errorbar(bws_v_df.index,  # Wire Scanner (Out)
            bws_v_df['EMITTANCE_OUT_V'],
            color='darkorange',
            label='__nolegend__',
            markersize=15,
            fmt='o'
            )

ax.plot(bsrt_h_df['EMITTANCE_H'],
        color='lime',
        marker='o',
        linestyle='None',
        markersize=10,
        label='__nolegend__')

ax.errorbar(bsrt_h_df.index,
            bsrt_h_df['EMITTANCE_H_AV7'],
            yerr = bsrt_h_df['EMITTANCE_H_STD7'],
            color='darkgreen',
            linewidth=4,
            label='Horizontal emittance from BSRT')

ax.errorbar(bws_h_df.index,
            bws_h_df['EMITTANCE_IN_H'],
            color='darkslategrey',
            label='Horizontal emittance from BWS',
            markersize=15,
            fmt='o'
            )

ax.errorbar(bws_h_df.index,
            bws_h_df['EMITTANCE_OUT_H'],
            color='darkslategrey',
            label='__nolegend__',
            markersize=15,
            fmt='o'
            )

for i, time in enumerate(kicks_df['TIME']):
    ax.axvline(x=time,
               color='grey',
               linestyle='--',
               alpha=0.8)


ax.set_xlim([min(kicks_df["TIME"]) - 20, max(kicks_df["TIME"]) + 20])
# ax.set_ylim([0, 5])

# timems = np.linspace( 1532433900, 1532435100, num=5 )
# timeticks = ['14:05', '14:10', '14:15', '14:20', '14:25']

# ax.set_xticks( timems )
# ax.set_xticklabels( timeticks , rotation=0 , ha='center' , fontsize = 15)

ax.set_xlabel('Time', fontsize=25)
ax.set_ylabel(r'$\epsilon_{n}$ $[\mu m]$', fontsize=25)
ax.tick_params(axis='x', pad=10)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend(loc='lower left', fontsize=25)
plt.tight_layout()
# plt.savefig( 'emittance_evolution.eps' )
# plt.savefig( 'emittance_evolution.png' )
# plt.savefig( 'emittance_evolution.pdf' )
plt.show()



### get intensity at kicktime for vertical kicks
waittime = 3

fbct_v_losses = np.zeros((len(kicks_df['TIME']),2))


for i, time in enumerate(kicks_df['TIME']):

    before_kick = fbct_df['INTENSITY'].iloc[ fbct_df.index.get_loc( float(time - 30), method='ffill'):fbct_df.index.get_loc( float(time - 5), method='ffill')  ]
    after_kick = fbct_df['INTENSITY'].iloc[ fbct_df.index.get_loc( float(time + waittime), method='ffill') : fbct_df.index.get_loc( float(time + waittime + 15), method='ffill') ]

    before_kick_av = np.mean( before_kick )
    before_kick_std = np.std( before_kick )
    after_kick_av = np.mean( after_kick )
    after_kick_std = np.std( after_kick )


    losses = before_kick_av - after_kick_av
    losses_std = np.sqrt(before_kick_std**2 + after_kick_std**2)

    fbct_v_losses[i,:] = losses  / before_kick_av , np.sqrt( (losses  / before_kick_av)**2 * ( ( losses_std / losses )**2 + ( before_kick_std / before_kick_av )**2  ) )


kicks_df = kicks_df.assign(Losses=fbct_v_losses[:,0])
kicks_df = kicks_df.assign(Losses_std=fbct_v_losses[:,1])

kicks_df['Losses_std'] = kicks_df['Losses_std'].replace(to_replace=0, method='bfill' )

kicks_df['Losses_std'] = kicks_df['Losses_std'] + 0.01

kicks_df = kicks_df.assign( cumulative_Losses=kicks_df['Losses'].cumsum())
kicks_df = kicks_df.assign( cumulative_Losses_std=kicks_df['Losses_std'])

kicks_df = kicks_df.assign( adding_Losses=kicks_df['Losses'].shift(1, fill_value=0) )
kicks_df['adding_Losses'] = np.where( kicks_df['adding_Losses'] < 0.0001, 0, kicks_df['adding_Losses'] ) + kicks_df['Losses']

kicks_df = kicks_df.assign( adding_Losses_std=kicks_df['Losses_std'])

print( kicks_df.tail(5) )

### get emittance at kicktime for vertical kicks
waittime = 3

kick_emittance = np.zeros((len(kicks_df['TIME']),2))


for i, time in enumerate(kicks_df['TIME']):

    kick_emittance[i,:] = bsrt_v_df['NEMITTANCE_V_AV7'].iloc[ bsrt_v_df.index.get_loc( float(time), method='ffill')  ], bsrt_v_df['NEMITTANCE_V_STD7'].iloc[ bsrt_v_df.index.get_loc( float(time), method='ffill') ]

#     kick_emittance[i,:] = bsrt_v_sigma_df['NEMITTANCE_V'].iloc[ bsrt_v_df.index.get_loc( float(time), method='ffill')  ], bsrt_v_sigma_df['NEMITTANCE_V_STD'].iloc[ bsrt_v_df.index.get_loc( float(time), method='ffill') ]



kicks_df = kicks_df.assign(nemittance=kick_emittance[:,0])
kicks_df = kicks_df.assign(nemittance_std=kick_emittance[:,1])

print( kicks_df.tail(5) )

fig, ax = plt.subplots( ncols=1, nrows=1, sharex=False, sharey=False, figsize=(9,9) )


ax.errorbar( kicks_df['2JYRES'], kicks_df['Losses']*100, xerr=kicks_df['2JYSTDRES'], yerr=kicks_df['Losses_std']*100, label='Single losses' )


ax.errorbar( kicks_df['2JYRES'], kicks_df['cumulative_Losses']*100, xerr=kicks_df['2JYSTDRES'], yerr=kicks_df['cumulative_Losses_std']*100, label='Cumulative losses' )


ax.errorbar( kicks_df['2JYRES'], kicks_df['adding_Losses']*100, xerr=kicks_df['2JYSTDRES'], yerr=kicks_df['adding_Losses_std']*100, label='added Losses from prev. Kick' )


ax.set_xlabel(r'$2J_y$')
ax.set_ylabel(r'Losses in %')
ax.legend()

plt.tight_layout()
plt.show()


#%%
kicks_df['Kick_sigma']  = ( np.sqrt( kicks_df['2JYRES']*10**-6 /kicks_df['nemittance']  ) )
kicks_df['Kick_sigma_std']  = 0.5 * kicks_df['2JYSTDRES']*10**-6 / np.sqrt( kicks_df['2JYRES']*10**-6 * kicks_df['nemittance'] )

kicks_df['Kick_sigma_cumulative']  = ( np.sqrt( kicks_df['2JYRES']*10**-6 /kicks_df['nemittance'].iloc[0]  ) )
kicks_df['Kick_sigma_cumulative_std']  = 0.5 * kicks_df['2JYSTDRES']*10**-6 / np.sqrt( kicks_df['2JYRES']*10**-6 * kicks_df['nemittance'].iloc[0] )

kicks_df['Kick_sigma_nominal']  = (kicks_df['Kick_sigma'] * np.sqrt(  kicks_df['nemittance']/emittance  ) )
kicks_df['Kick_sigma_nominal_std']  = 0.5 * kicks_df['2JYSTDRES']*10**-6 / np.sqrt( kicks_df['2JYRES']*10**-6 * emittance )

print( kicks_df['Kick_sigma'] )

print( kicks_df.tail(5) )

# losses formula assuming kicks and DA are normalised to sigma
def exp_decay_sigma(p, x):

    return np.exp( (x - p)/(2*1) )
# fit forced DA for single losses
kicks_df = kicks_df.iloc[0:11, :]
exp_decay_sigma_model = scipy.odr.Model(exp_decay_sigma)
data_model_sigma = scipy.odr.RealData( x = kicks_df['Kick_sigma'] ,
                                y = kicks_df['Losses'],
                                sx = kicks_df['Kick_sigma_std'],
                                sy = kicks_df['Losses_std'] )

da_odr = scipy.odr.ODR( data_model_sigma, exp_decay_sigma_model, beta0=[4.] )
# da_odr.set_job( fit_type=2 )
odr_output = da_odr.run()
odr_output.pprint()


#%%
# plot losses and forced DA fit

DA = odr_output.beta[0]
DA_err =  odr_output.sd_beta[0]

print('-------------------------------------------------------------')
print('DA vertical: ' + str(DA) + ' +- ' + str(DA_err))
print('-------------------------------------------------------------')

# plot losses and forced DA fit

DA = odr_output.beta[0]
DA_err =  odr_output.sd_beta[0]

print('-------------------------------------------------------------')
print('DA vertical: ' + str(DA) + ' +- ' + str(DA_err))
print('-------------------------------------------------------------')

# plot Kicks over losses
fig, ax = plt.subplots( ncols=1, nrows=1, sharex=False, sharey=False, figsize=(10,10) )


ax.errorbar( kicks_df['Kick_sigma'], kicks_df['Losses']*100, xerr=kicks_df['Kick_sigma_std'], yerr=kicks_df['Losses_std']*100,
            label='AC dipole', capsize=10, linestyle='', color='darkgreen', elinewidth=3., capthick=4  )

fitish = np.zeros(len(kicks_df["Kick_sigma"]))

for i in range(len(fitish)):
    fitish[i] = exp_decay_sigma( odr_output.beta[0], kicks_df['Kick_sigma'].iloc[i]  )

ax.plot( kicks_df['Kick_sigma'], fitish*100, label='Fit: DA= ${:.1f} \pm {:.1f} \sigma_{{measured}}$'.format( DA, DA_err ), color='lawngreen', linewidth=4 )

ax.set_xlabel(r'$ N \sigma_{y, measured}$', fontsize=25)
ax.set_ylabel(r'Losses in % ', fontsize=25)

ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend(loc='upper left', fontsize=25)
# ax.set_ylim([0,20])
# ax.set_xlim([0,12])
# ax_doub.set_xlim([0,6])
plt.tight_layout()
plt.savefig( 'ac_dipole_losses_sigma.eps' )
plt.show()

exit()


exp_decay_sigma_model = scipy.odr.Model(exp_decay_sigma)
data_model_sigma = scipy.odr.RealData( x = kicks_df['Kick_sigma_cumulative'] ,
                                y = kicks_df['cumulative_Losses'],
                                sx = kicks_df['Kick_sigma_cumulative_std'],
                                sy = kicks_df['cumulative_Losses_std'] )

da_odr = scipy.odr.ODR( data_model_sigma, exp_decay_sigma_model, beta0=[4.] )
# da_odr.set_job( fit_type=2 )
odr_output = da_odr.run()
odr_output.pprint()

