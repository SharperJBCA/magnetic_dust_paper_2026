import numpy as np 
from matplotlib import pyplot as plt 
import healpy as hp 

def _planck_Bnu(nu_hz: float, T: float) -> float:
    """
    Planck function B_nu(T) [W sr^-1 m^-2 Hz^-1] up to an overall constant.
    For our ratio B_nu/B_nu0, absolute units cancel, but we include full form for clarity.
    """
    # Physical constants (SI)
    _H = 6.62607015e-34   # Planck constant [J s]
    _KB = 1.380649e-23    # Boltzmann constant [J/K]
    _C = 299792458.0      # speed of light [m/s]

    if T <= 0:
        raise ValueError("T must be > 0")

    x = (_H * nu_hz) / (_KB * T)
    denom = np.expm1(x)
    return (2.0 * _H * nu_hz**3 / _C**2) / denom

def plot_spectrum(components, out_dir):
    """
    Plot spectrum of every pixel in all maps
    """
    

    print('HELLO', out_dir)

    nside = 16
    npix = 12*nside**2
    nu_plot = np.logspace(0,3,100)
    component_names = list(components.keys())
    n_components = len(component_names)
    data = np.zeros((n_components,nu_plot.size, npix)) 
    for j, (k,v) in enumerate(components.items()):

        for i in range(nu_plot.size):
            data[j,i] += v['component'].evaluate(nu_ghz=nu_plot[i],
                                        T=v['template'],
                                        params=v['params'])['I']

    #for ipix in range(npix):
    ipix = 0
    for i in range(n_components):
        hi = np.percentile(data[i,:,:],95,axis=1)
        lo = np.percentile(data[i,:,:],5,axis=1)
        plt.fill_between(nu_plot,lo,hi,alpha=0.75,label=component_names[i])
        #plt.plot(nu_plot,data[i,:,ipix],label=component_names[i])
            
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(np.max(data)*1e-9,np.max(data))
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Brightness [K]')
    plt.xlim(1,1000) 
    plt.savefig(f'{out_dir}/intensity_spectrum.png')
    plt.close()

    nside = 16
    npix = 12*nside**2
    nu_plot = np.logspace(0,3,100)
    print('hello')
    component_names = list(components.keys())
    n_components = len(component_names)
    data = np.zeros((n_components,nu_plot.size, npix)) 
    for j, (k,v) in enumerate(components.items()):

        for i in range(nu_plot.size):
            Q = v['component'].evaluate(nu_ghz=nu_plot[i],
                                        T=v['template'],
                                        params=v['params']).get('Q',0)
            U = v['component'].evaluate(nu_ghz=nu_plot[i],
                                        T=v['template'],
                                        params=v['params']).get('U',0)
            data[j,i,:] = np.sqrt(Q**2 + U**2)

    #for ipix in range(npix):
    ipix = 0
    for i in range(n_components):
        hi = np.percentile(data[i,:,:],95,axis=1)
        lo = np.percentile(data[i,:,:],5,axis=1)
        plt.fill_between(nu_plot,lo,hi,alpha=0.75,label=component_names[i])
            
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(np.max(data)*1e-10,np.max(data)*1e-1)
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Brightness [K]')
    plt.xlim(1,1000) 
    cmb = _planck_Bnu(nu_plot*1e9,2.73) * nu_plot**-2
    plt.plot(nu_plot,cmb/np.max(cmb)*1e-8,'k--',lw=3)
    print(f'{out_dir}/polarisation_spectrum.png')
    plt.savefig(f'{out_dir}/polarisation_spectrum.png')
    plt.close()

