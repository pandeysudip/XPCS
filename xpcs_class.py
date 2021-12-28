import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skbeam.core.roi as roi
import skbeam.core.correlation as corr
import skbeam.core.utils as utils
import xray_vision
import xray_vision.mpl_plotting as mpl_plot
import time
from databroker import DataBroker as db, get_table
from csxtools.utils import get_fastccd_images, get_images_to_3D, get_images_to_4D, fccd_mask, get_fastccd_flatfield
from csxtools.image import stackmean, images_mean, images_sum
from csxtools.ipynb import image_stack_to_movie, show_image_stack
from matplotlib.colors import LogNorm
from skbeam.core import recip
from skbeam.core. utils import grid3d
from scipy.signal import savgol_filter as sgf
from matplotlib import colors
from ipywidgets import interact
from matplotlib import rcParams
import lmfit
import matplotlib.image as mpimg
from PIL import Image
from pathlib import Path
from collections import OrderedDict
from lmfit.models import LorentzianModel
from bokeh.plotting import figure, output_notebook, show, reset_output
from bokeh.palettes import Category10

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams.update({'font.size': 20})
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True
rcParams['xtick.major.size'] = 10
rcParams['ytick.major.size'] = 10
rcParams['xtick.minor.size'] = 5
rcParams['ytick.minor.size'] = 5
rcParams['xtick.minor.visible'] = True
rcParams['ytick.minor.visible'] = True

class XPCS:
    def __init__(self, temp, light, dark, roi_edge=(710,730), num_levels=8, num_bufs=12):
        self.temp=temp
        self.light=light
        self.dark=dark
        self.roi_edge=roi_edge
        self.num_levels=num_levels
        self.num_bufs=num_bufs
        
    #g2 for total intensity
    def total_intensity(self):
    
        light0 = db[self.light]
        dark0 = db[self.dark[0]]
        dark1 = db[self.dark[1]]
        dark2 = db[self.dark[2]]
        
        images = get_fastccd_images(light0, (dark0, dark1, dark2), flat=None)
        stack = get_images_to_3D(images)
        mean_stack = stackmean(stack)
        
        total_inten=[]
        for i in range(0, 3600):
            I=np.sum(stack[i,:,:])
            total_inten.append(I)
        
        fig=plt.figure(figsize=(10,30))
        fig.subplots_adjust(hspace=.25)
        ax=fig.add_subplot(2,1,1)
        
        ax.plot(y=total_inten,x=range(0,3600),'o', markersize=5, color='r' )
        ax.set(ylabel='Intensity', xlabel='Frame')
        ax.set_title(temp)
        
        stack_all=np.ones((3600,960,1050))
        for i in range(3600):
            stack_all[i]=stack_all[i]*total_inten[i]
            
        label_array = roi.rings(self.roi_edge, (450, 1050), shape=mean_stack.shape)
        
        #skbeam.core.correlation.multi_tau_auto_corr(num_levels, num_bufs, labels, images)
        #The longest lag time computed is num_levels * num_bufs.
        #lag_steps: the times at which the correlation was computed
        g2, lag_steps = corr.multi_tau_auto_corr(self.num_levels, self.num_bufs, label_array, stack_all)
        
        ax=fig.add_subplot(2,1,2)
        ax.semilogx(lag_steps * 1 ,g2/g2[10]) 
        ax.set(ylabel='g2', xlabel='time(s)', ylim=(0.95, 1.004))
        ax.set_title(temp)    
    
        plt.show()
        return g2, lag_steps
    
    
    #Intensity vs q
    def intensity_q(self, light_offr, dark_offr):
        #on resonance
        light_r = db[self.light]
        dark0_r = db[self.dark[0]]
        dark1_r = db[self.dark[1]]
        dark2_r = db[self.dark[2]]
    
        #off resonance
        light0_offr = db[light_offr]
        dark0_offr = db[dark_offr[0]]
        dark1_offr = db[dark_offr[1]]
        dark2_offr = db[dark_offr[2]]
    
    
        images_r = get_fastccd_images(light_r, (dark0_r, dark1_r, dark2_r), flat=None)
        stack_r = get_images_to_3D(images_r)
    
        images_offr = get_fastccd_images(light0_offr, (dark0_offr, dark1_offr, dark2_offr), flat=None)
        stack_offr = get_images_to_3D(images_offr)
    
        mean_stack_r = stackmean(stack_r)
        mean_stack_offr= stackmean(stack_offr)
        
        I_sum = []
        J_sum = []
        for i in range(1, 100):
            roi_edges= (10*i,10*i+10)
            label_array = roi.rings(roi_edges, (450,1050), shape=mean_stack_r.shape)
            I = mean_stack_1*label_array
            J = mean_stack_2*label_array
            x=np.count_nonzero(label_array)
            I_sum.append((np.sum(I)/x))
            J_sum.append((np.sum(J)/x))
        
        x=range(10, 100)
        y_r=I_sum[10:]
        y_offr=J_sum[10:]
    
        #fig=plt.figure(figsize=(10,30))
        #fig.subplots_adjust(hspace=.25)
        #ax=fig.add_subplot(3,1,1)
        #ax.plot(x, y_r,'o', markersize=5, color='r')
        #ax.set_title(temp)
        
        #ax=fig.add_subplot(3,1,2)
        #ax.plot(x, y_r,'o', markersize=5, color='r')
        #ax.set_title(temp)
        
        fig,ax=plt.subplots()
        ax.plot(x, y_r/np.max(y_r), 'o', color='red', label='ON')
        ax.plot(x, y_offr/np.max(y_offr), 'o', color='blue',label='OFF')
        ax.legend()
        ax.grid()
        ax.set_xlabel('Q [pxl]');
        ax.set_ylabel('Norm.int[arb.u.]')
        ax.set_title(temp)
        plt.show()
    
    def sg_g2(self):
    
        light0 = db[self.light]
        dark0 = db[self.dark[0]]
        dark1 = db[self.dark[1]]
        dark2 = db[self.dark[2]]
        
        images = get_fastccd_images(light0, (dark0, dark1, dark2), flat=None)
        stack = get_images_to_3D(images)
        mean_stack = stackmean(stack)
        
        label_array = roi.rings(self.roi_edge, (450, 1050), shape=mean_stack.shape)
        
        set_aspect='auto'
    
        fig, ax=plt.subplots(1,2)
        ax[0].imshow(mean_stack, cmap='hsv', interpolation=None, norm=LogNorm())
        ax[0].set_title(temp)
        
        ax[1].imshow(mean_stack, cmap='hsv', interpolation=None, norm=LogNorm())
        mpl_plot.show_label_array(ax[1], label_array, cmap=None, norm=LogNorm(), alpha=1, **kwargs)
        ax[1].set_title(temp)
        
        #skbeam.core.correlation.multi_tau_auto_corr(num_levels, num_bufs, labels, images)
        #The longest lag time computed is num_levels * num_bufs.
        #lag_steps: the times at which the correlation was computed
        start_time = time.time()
        g2, lag_steps = corr.multi_tau_auto_corr(self.num_levels, self.num_bufs, label_array, stack)
        
        fig, ax=plt.subplots()
        ax.semilogx(lag_steps * 1 ,g2/g2[10]) 
        ax.set(ylabel='g2', xlabel='time(s)', ylim=(0.95, 1.004))
        ax.set_title(temp)
        plt.show()
        
        stack_s=("Stack shape is".format(stack.shape))
        mstack=("Mean stack shape is".format(mean_stack.shape))
        autotime=("--- Time to calculate autocorrelation function: {} seconds ---".format(time.time() - start_time))
        return stack_s, mstack, autotime, g2, lag_steps
    
    #g2 for resonance and offresonance (both)
    def sg_g2_both(self, light_offr, dark_offr):
        #on resonance
        light_r = db[self.light]
        dark0_r = db[self.dark[0]]
        dark1_r = db[self.dark[1]]
        dark2_r = db[self.dark[2]]
    
        #off resonance
        light0_offr = db[light_offr]
        dark0_offr = db[dark_offr[0]]
        dark1_offr = db[dark_offr[1]]
        dark2_offr = db[dark_offr[2]]
    
    
        images_r = get_fastccd_images(light_r, (dark0_r, dark1_r, dark2_r), flat=None)
        stack_r = get_images_to_3D(images_r)
    
        images_offr = get_fastccd_images(light_offr, (dark0_offr, dark1_offr, dark2_offr), flat=None)
        stack_offr = get_images_to_3D(images_offr)
    
        mean_stack_r = stackmean(stack_r)
        mean_stack_offr= stackmean(stack_offr)
        
        label_array = roi.rings(self.roi_edge, (450, 1050), shape=mean_stack.shape)
        
        set_aspect='auto'
        fig, ax=plt.subplots(1,2, figsize=(10,10))
        ax[0].imshow(mean_stack_r, cmap='hsv', interpolation=None, norm=LogNorm())
        ax[1].imshow(mean_stack_offr, cmap='hsv', interpolation=None, norm=LogNorm())
        ax[0].set_title(temp)
        
        fig, ax=plt.subplots(1,2, figsize=(15,10))
        ax[0].imshow(mean_stack_r, cmap='hsv', interpolation=None, norm=LogNorm())
        mpl_plot.show_label_array(ax[0], label_array, cmap=None, norm=LogNorm(), alpha=1, **kwargs)
        ax[0].set_title('Resonance')
        
        
        ax[1].imshow(mean_stack_offr, cmap='hsv', interpolation=None, norm=LogNorm())
        mpl_plot.show_label_array(ax[1], label_array, cmap=None, norm=LogNorm(), alpha=1, **kwargs)
        ax[1].set_title('Offesonance')
        
        #skbeam.core.correlation.multi_tau_auto_corr(num_levels, num_bufs, labels, images)
        #The longest lag time computed is num_levels * num_bufs.
        #lag_steps: the times at which the correlation was computed
        start_time_r = time.time()
        g2_r, lag_steps_r = corr.multi_tau_auto_corr(self.num_levels, self.num_bufs, label_array, stack_r)
        
        start_time_offr = time.time()
        g2_offr, lag_steps_offr = corr.multi_tau_auto_corr(self.num_levels, self.num_bufs, label_array, stack_offr)
        
        fig, ax=plt.subplots(figsize=(15,10))
        ax.semilogx(lag_steps_r * 1 ,g2_r/g2_r[10], label='resonance') 
        ax.semilogx(lag_steps_offr * 1 ,g2_offr/g2_offr[10], label='offresonance') 
        ax.set(ylabel='g2', xlabel='time(s)', ylim=(0.95, 1.004), title='g2 for resonance and off resonance')
        ax.legend()
        ax.set_title(temp)
        plt.show()
        
        stack_s=("Stack shape is".format(stack.shape))
        mstack=("Mean stack shape is".format(mean_stack.shape))
        autotime_r=("--- Time to calculate autocorrelation function for resonance: {} seconds ---".format(time.time() - start_time_r))
        autotime_offr=("--- Time to calculate autocorrelation function for resonance: {} seconds ---".format(time.time() - start_time_offr))
        return stack_s, mstack, autotime_r, autotime_offr
    
    #Intensity correlation for 1 and 2 cycle 
    def intensity_corr(self, light2, dark2,frame=(500, 4000, 500)):
        #1st cycle
        light_1 = db[self.light]
        dark0_1 = db[self.dark[0]]
        dark1_1 = db[self.dark[1]]
        dark2_1 = db[self.dark[2]]
    
        #2nd cycle
        light_2 = db[light2]
        dark0_2 = db[dark2[0]]
        dark1_2 = db[dark2[1]]
        dark2_2 = db[dark2[2]]
    
        images_1 = get_fastccd_images(light_1, (dark0_1, dark1_1, dark2_1), flat=None)
        stack_1 = get_images_to_3D(images_1)
    
        images_2 = get_fastccd_images(light_offr, (dark0_2, dark1_2, dark2_2), flat=None)
        stack_2 = get_images_to_3D(images_2)
    
        mean_stack_1 = stackmean(stack_1)
        mean_stack_2= stackmean(stack_2)
        
        frames_1=[]
        frames_2=[]
        for i in range(frame):  #xrange=[500, 1000, 1500, 2000, 2500, 3000, 3500]
            frame1= stack_1[i,:,:]
            frame2= stack_2[i,:,:]
            frames_1.append(frame1)
            frames_2.append(frame2)
        
        label_array = roi.rings(self.roi_edge, (450,1050), shape=mean_stack_1.shape)
        
        intensity1=[]
        intensity2=[]
        for f1 in frames1:
            int1 = f1*label_array
            #total_int1=intensity1.sum()
            intensity1.append(int1)
      
        for f2 in frames2:
            int2 = f2*label_array
            #total_int2=intensity2.sum()
            intensity2.append(int2)
            
        norm_corr=[]
        for x in intensity1:
            for y in intensity2:
                num = 0
                dem1 = 0
                dem2 = 0
                a=([x[i]*y[i] for i in range(len(x))])
                b=([x[i]*x[i] for i in range(len(x))])
                c=([y[i]*y[i] for i in range(len(y))])
                num+=np.sum(a)
                dem1+=np.sum(b)
                dem2+=np.sum(c)
                dem=np.sqrt(dem1*dem2)
                norm=(num/dem)
            norm_corr.append(norm)
            
        y=norm_corr
        x=range(frame)
        fig, ax =plt.subplots()
        ax.plot(x,y,'o', markersize=15, color='r' )
        ax.set(ylabel='Speckle cross-correlation', xlabel='Frame')
        ax.set_title(temp)
        plt.show()
        
        stack_1s=("Stack1 shape is".format(stack_1.shape))
        stack_2s=("Stack2 shape is".format(stack_2.shape))
        mstack=("Mean stack1 shape is".format(mean_stack_1.shape))
        intensity_1=("Intensity of ist cycle is".format(intensity1))
        intensity_2=("Intensity of 2nd cycle is".format(intensity2))
        norm_corrle=("normalied correlation of both cycle is".format(norm_corr))
        return stack_1s, stack_2s, mstack, intensity_1, intensity_2, norm_corrle


    #Intensity corr for 1 and 2 cycle for small pixel range
    def intensity_corr_small(self,light2, dark2, xrange=(720,760),yrange=(425,525),frame=(500, 4000, 500)):
    
        #1st cycle
        light_1 = db[self.light]
        dark0_1 = db[self.dark[0]]
        dark1_1 = db[self.dark[1]]
        dark2_1 = db[self.dark[2]]
    
        #2nd cycle
        light_2 = db[light2]
        dark0_2 = db[dark2[0]]
        dark1_2 = db[dark2[1]]
        dark2_2 = db[dark2[2]]
    
        images_1 = get_fastccd_images(light_1, (dark0_1, dark1_1, dark2_1), flat=None)
        stack_1 = get_images_to_3D(images_1)
    
        images_2 = get_fastccd_images(light_offr, (dark0_2, dark1_2, dark2_2), flat=None)
        stack_2 = get_images_to_3D(images_2)
    
        mean_stack_1 = stackmean(stack_1)
        mean_stack_2= stackmean(stack_2)
        
        frames_1=[]
        frames_2=[]
        for i in range(frame):
            frame1= stack_1[i,:,:]
            frame2= stack_2[i,:,:]
            frames_1.append(frame1)
            frames_2.append(frame2)
        
        #x=[720:760]
        #y=[450:49]
        label_array_fifty = np.zeros((960,1000))
        for y in range(yrange):
            for x in range(xrange):
                label_array_fifty[y][x]=1
        
        intensity1=[]
        intensity2=[]
        for f1 in frames1:
            int1=f1*label_array_fifty
            #total_int1=intensity1.sum()
            intensity1.append(int1)
            
        for f2 in frames2:
            int2=f2*label_array_fifty
            #total_int2=intensity2.sum()
            intensity2.append(int2)
            
        norm_corr_small=[]
        for x in intensity1:
            for y in intensity2:
                num = 0
                dem1 = 0
                dem2 = 0
                a=([x[i]*y[i] for i in range(len(x))])
                b=([x[i]*x[i] for i in range(len(x))])
                c=([y[i]*y[i] for i in range(len(y))])
                num+=np.sum(a)
                dem1+=np.sum(b)
                dem2+=np.sum(c)
                dem=np.sqrt(dem1*dem2)
                norm=(num/dem)
            norm_corr_small.append(norm)
            
        y=norm_corr_small
        x=range(frame)
        fig, ax =plt.subplots()
        ax.plot(x,y,'o', markersize=15, color='r' )
        ax.set(ylabel='Speckle cross-correlation', xlabel='Frame')
        ax.set_title(temp)
        plt.show()
        
        stack_1s=("Stack1 shape is".format(stack_1.shape))
        stack_2s=("Stack2 shape is".format(stack_2.shape))
        mstack=("Mean stack1 shape is".format(mean_stack_1.shape))
        intensity_1=("Intensity of ist cycle is".format(intensity1))
        intensity_2=("Intensity of 2nd cycle is".format(intensity2))
        norm_corrle=("normalied correlation of both cycle is".format(norm_corr_small))
        return stack_1s, stack_2s, mstack, intensity_1, intensity_2, norm_corrle

