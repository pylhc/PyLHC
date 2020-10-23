import numpy as np


def return_resonance_coefficients(order, ac_order=0):
    """Given the order of the resonance and, optionally, the order up to with 
    AC-Dipole resonances should be accounted for, returns a list of lists with 
    the components of all the resonances up to the specified order."""
    resonances=[]

    for a in np.linspace(start=-order, stop=order, num=2*order+1, endpoint=True):
        for b in np.linspace(start=-order, stop=order, num=2*order+1, endpoint=True):
            for c in np.linspace(start=-ac_order, stop=ac_order, num=2*ac_order+1, endpoint=True):
                for d in np.linspace(start=-ac_order, stop=ac_order, num=2*ac_order+1, endpoint=True):
                     if   np.sum([ abs(a), abs(b), abs(c), abs(d)]) <= order:
                        resonances.append( [a,b,c,d] )

    return resonances


def add_resonance_lines_to_plot(Qx=[0,0.5], Qy=[0,0.5], deltax=0, deltay=0, order=5 , ac_order=0, ax=None):
    """ Given a range of tunes and, optionally, the difference between AC-Dipole Frequency and
    natural Frequency, add resonance lines up to the specified order to a matplotlib axis object.
    """

    Qx=np.array(Qx)
    Qy=np.array(Qy)
    
    ax.set_xlim( Qx )
    ax.set_ylim( Qy )
    
    
    resonances = return_resonance_coefficients(order=order, ac_order=ac_order)

    for r in resonances:
        a,b,c,d = r

        for Res in np.linspace(-order, order, num =2*order+1, endpoint=True):
            
            if (b+d)!=0:
                ResQx = Qx
                ResQy = ((Res - (a+c)*Qx -c*deltax - d*deltay )/(b+d))   

                if c==0 and d==0:
                    ax.plot( ResQx, ResQy , color='black', alpha=1./(abs(Res)+1))
                else:
                    ax.plot( ResQx, ResQy , color='grey', alpha=1./(abs(Res)+1), linestyle='--')
            elif (b+d)==0 and (a+c) !=0:    
                
                ResQx = [((Res - c*deltax - d*deltay)/(a+c)) , ((Res - c*deltax - d*deltay)/(a+c)) ]
                ResQy= Qy
                if c==0 and d==0:
                    ax.plot( ResQx, ResQy , color='black', alpha=1./(abs(Res)+1))
                else:
                    ax.plot( ResQx, ResQy , color='grey', alpha=1./(abs(Res)+1), linestyle='--')

                    
    return None