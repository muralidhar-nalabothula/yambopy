# Copyright (c) 2018, Henrique Miranda
# All rights reserved.
#
# This file is part of the yambopy project
#
import os
import numpy as np
from yambopy.units import ha2ev
from yambopy.tools.string import marquee
from yambopy.plot.bandstructure import YamboBandStructure
from yambopy.plot.plotting import add_fig_kwargs
from yamboparser import YamboFile

class YamboQPDB():
    """
    Class to read yambo ndb.QP files

    These files describe the quasiparticle states calculated from yambo
    Includes the quasi-particl energies, the lifetimes and the Z factors
    """
    def __init__(self,qps):
        """
        Initialize the YamboQP class
        """
        self.qps          = qps
        self.kpoints      = np.array(qps['Kpoint'])
        self.kpoint_index = np.array(qps['Kpoint_index'],dtype=int)
        self.band_index   = np.array(qps['Band'],dtype=int)
        self.e0           = np.array(qps['Eo']).real*ha2ev
        self.e            = np.array(qps['E']).real*ha2ev
        self.linewidths   = np.array(qps['E']).imag*ha2ev
        self.eigenvalues_qp, self.eigenvalues_dft, self.lifetimes = self.get_qps()

    @classmethod
    def from_db(cls,filename='ndb.QP',folder='.'):
        """
        Create instance of this class from a ndb.QP file
        """
        db_path = os.path.join(folder,filename)
        if os.path.isfile(db_path):
            yfile = YamboFile(filename,folder)
        else:
            raise IOError('File %s not found'%db_path)
        return cls(yfile.data)
    
    def get_qps(self):
        """
        Get quasiparticle energies in a list
        """
        #start arrays
        eigenvalues_dft = np.zeros([self.nkpoints,self.nbands])
        eigenvalues_qp  = np.zeros([self.nkpoints,self.nbands])
        linewidths      = np.zeros([self.nkpoints,self.nbands])
        for ei,e0i,li,ki,ni in zip(self.e,self.e0,self.linewidths,self.kpoint_index,self.band_index):
            nkpoint = ki-self.min_kpoint
            nband = ni-self.min_band
            eigenvalues_dft[nkpoint,nband] = e0i
            eigenvalues_qp[nkpoint,nband] = ei
            linewidths[nkpoint,nband] = li

        return eigenvalues_dft, eigenvalues_qp, linewidths

    def get_filtered_qps(self,min_band=None,max_band=None):
        """Return selected QP energies as a flat list"""
        e0=[]; qp=[]; lw=[]
        for ei,e0i,li,ki,ni in zip(self.e,self.e0,self.linewidths,self.kpoint_index,self.band_index):
            if min_band and ni < min_band: continue
            if max_band and ni > max_band: continue
            e0.append(e0i)
            qp.append(ei)
            lw.append(lw)
        return e0,qp,lw

    def get_direct_gaps(self,valence):
        """
        Compute the QP and DFT gaps
        
        Arguments:
            valence: number of bands in the valence
        """
        eigenvalues_dft, eigenvalues_qp, linewidths = self.get_qps()
        na = np.newaxis
        shifted_valence = valence-self.min_band

        #direct gap
        dft_jdos = eigenvalues_dft[:,na,shifted_valence+1:]-eigenvalues_dft[:,:shifted_valence,na]
        qp_jdos  = eigenvalues_qp[:,na,shifted_valence+1:] -eigenvalues_qp[:,:shifted_valence,na]
        direct_dft_gap = np.min(dft_jdos)
        direct_qp_gap  = np.min(qp_jdos)

        #indirect gap
        #TODO take the min and max of the VBM and CBM
        return direct_dft_gap, direct_qp_gap

    def get_scissor(self,valence,verbose=1):
        """
        Compute the scissor operator replicating the QP corrections
        
        Arguments:
            valence: number of bands in the valence
        """
        from scipy import stats
        lines = []; app = lines.append
        #valence
        e0, eqp, lw = self.get_filtered_qps(self.min_band,valence)
        vslope, vintercept, r_value, p_value, std_err = stats.linregress(e0,eqp)
        app('valence bands:')
        app('slope:     {}'.format(vslope))
        app('intercept: {}'.format(vintercept))
        app('r_value:   {}'.format(r_value))
        app('p_value:   {}'.format(p_value))
        app('std_err:   {}'.format(std_err))
       
        #conduction
        e0, eqp, lw = self.get_filtered_qps(valence+1,self.max_band)
        cslope, cintercept, r_value, p_value, std_err = stats.linregress(e0,eqp)
        app('\nconduction bands:')
        app('slope:     {}'.format(cslope))
        app('intercept: {}'.format(cintercept))
        app('r_value:   {}'.format(r_value))
        app('p_value:   {}'.format(p_value))
        app('std_err:   {}'.format(std_err))

        #get gaps
        direct_dft_gap,direct_qp_gap = self.get_direct_gaps(valence) 
        shift = direct_qp_gap-direct_dft_gap

        scissor_list = [shift,cslope,vslope]
        app('\vscissor list (shift,c,v) [eV,adim,adim]: {}'.format(scissor_list))
        if verbose: print("\n".join(lines))
        return shift,cslope,vslope,cintercept,vintercept

    def plot_scissor_ax(self,ax,valence,verbose=1):
        """
        Plot the scissor on a matplotlib axis
        """
        shift,cslope,vslope,cintercept,vintercept=self.get_scissor(valence,verbose=verbose)
        #plot qps
        ve0,vqp,_ = self.get_filtered_qps(self.min_band,valence)
        ax.scatter(ve0,vqp)
        ce0,cqp,_ = self.get_filtered_qps(valence+1,self.max_band)
        ax.scatter(ce0,cqp)

        #plot the fits
        vx = np.linspace(np.min(ve0),np.max(ve0),2)
        cx = np.linspace(np.min(ce0),np.max(ce0),2)
        vy = cslope*vx+cintercept
        cy = cslope*cx+cintercept
        ax.plot(vx,vy)
        ax.plot(cx,cy)

    @add_fig_kwargs
    def plot_scissor(self,valence,verbose=1):
        """
        Plot the the QP energies and the scissor fit
        """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        self.plot_scissor_ax(ax,valence,verbose=verbose)
        return fig

    def get_bs(self,bs=None):
        """
        Get YamboBandStructure object with the KS and GW bands
        """
        if bs is None: bs = YamboBandStructure()

        eigenvalues_dft, eigenvalues_qp, lifetimes = self.get_qps()
        
        #add bands
        bs.add_bands(eigenvalues_dft,label='KS')
        bs.add_bands(eigenvalues_qp, label='GW')

        return bs

    @add_fig_kwargs
    def plot_bs(self):
        """
        Get and plot QP bandstructure
        """
        bs = self.get_bs()
        return bs.plot(show=False)

    @property
    def nqps(self):
        return len(self.e)

    @property
    def min_kpoint(self):
        return min(self.kpoint_index)

    @property
    def max_kpoint(self):
        return max(self.kpoint_index)

    @property
    def nbands(self):
        return self.max_band-self.min_band+1

    @property
    def min_band(self):
        return min(self.band_index)

    @property
    def max_band(self):
        return max(self.band_index)

    @property
    def nkpoints(self):
        return len(self.kpoints)
    
    def __str__(self):
        lines = []; app = lines.append
        app(marquee(self.__class__.__name__))
        app("nqps:     %d"%self.nqps)
        app("nkpoints: %d"%self.nkpoints)
        app("nbands:   %d"%self.nbands)
        app("min_band: %d"%self.min_band)
        app("max_band: %d"%self.max_band)
        return "\n".join(lines)

