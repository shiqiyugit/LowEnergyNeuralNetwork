#!/usr/bin/env python

############################################
# Scripts for plotting functions
# Contains functions:
#     get_RMS - used to calculate RMS for plotting statistics
#     get_FWHM - FWHM calculation method, not currently plugged in
#     plot_history - scatter line plot of loss vs epochs
#     plot_distributions_CCNC - plot energy distribution for truth, and for NN reco
#     plot_resolutions_CCNC - plot energy resoltuion for (NN reco - truth)
#     plot_2D_prediction - 2D plot of True vs Reco
#     plot_single_resolution - Resolution histogram, (NN reco - true) and can compare (old reco - true)
#     plot_compare_resolution - Histograms of resolutions for systematic sets, overlaid
#     plot_systematic_slices - "Scatter plot" with systematic sets on x axis and 68% resolution on y axis
#     plot_energy_slices - Scatter plot energy cut vs resolution
##############################################

import numpy as np
import h5py
import os, sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
import scipy.stats
import itertools

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

font = {
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

def get_RMS(resolution,weights=None):
    if weights is not None:
        import wquantiles as wq
    mean_array = np.ones_like(resolution)*np.mean(resolution)
    if weights is None:
        rms = np.sqrt( sum((mean_array - resolution)**2)/len(resolution) )
    else:
        rms = np.zeros_like(resolution)
        rms = np.sqrt( sum(weights*(mean_array - resolution)**2)/sum(weights) )
    return rms

def get_FWHM(resolution,bins):
    x_range = np.linspace(min(resolution),max(resolution),bins)
    y_values,bin_edges = np.histogram(resolution,bins=bins)
    spline = UnivariateSpline(x_range,y_values - max(y_values)/2.)
    r = spline.roots()
    if len(r) != 2:
        print("Root are weird")
        print(r)
        r1 = 0
        r2 = 0
    else:
        r1, r2 = spline.roots()
    return r1, r2

def find_contours_2D(x_values,y_values,xbins,weights=None,c1=16,c2=84):   
    """
    Find upper and lower contours and median
    x_values = array, input for hist2d for x axis (typically truth)
    y_values = array, input for hist2d for y axis (typically reconstruction)
    xbins = values for the starting edge of the x bins (output from hist2d)
    c1 = percentage for lower contour bound (16% - 84% means a 68% band, so c1 = 16)
    c2 = percentage for upper contour bound (16% - 84% means a 68% band, so c2=84)
    Returns:
        x = values for xbins, repeated for plotting (i.e. [0,0,1,1,2,2,...]
        y_median = values for y value medians per bin, repeated for plotting (i.e. [40,40,20,20,50,50,...]
        y_lower = values for y value lower limits per bin, repeated for plotting (i.e. [30,30,10,10,20,20,...]
        y_upper = values for y value upper limits per bin, repeated for plotting (i.e. [50,50,40,40,60,60,...]
    """
    if weights is not None:
        import wquantiles as wq
    y_values = np.array(y_values)
    indices = np.digitize(x_values,xbins)
    r1_save = []
    r2_save = []
    median_save = []
    for i in range(1,len(xbins)):
        mask = indices==i
        if len(y_values[mask])>0:
            if weights is None:
                r1, m, r2 = np.percentile(y_values[mask],[c1,50,c2])
            else:
                r1 = wq.quantile(y_values[mask],weights[mask],c1/100.)
                r2 = wq.quantile(y_values[mask],weights[mask],c2/100.)
                m = wq.median(y_values[mask],weights[mask])
        else:
            r1 = 0
            m = 0
            r2 = 0
        median_save.append(m)
        r1_save.append(r1)
        r2_save.append(r2)
    median = np.array(median_save)
    lower = np.array(r1_save)
    upper = np.array(r2_save)

    x = list(itertools.chain(*zip(xbins[:-1],xbins[1:])))
    y_median = list(itertools.chain(*zip(median,median)))
    y_lower = list(itertools.chain(*zip(lower,lower)))
    y_upper = list(itertools.chain(*zip(upper,upper)))
    
    return x, y_median, y_lower, y_upper



def plot_history(network_history,save=False,savefolder=None,use_logscale=False):
    """
    Plot history of neural network's loss vs. epoch
    Recieves:
        network_history = array, saved metrics from neural network training
        save = optional, bool to save plot
        savefolder = optional, output folder to save to, if not in current dir
    Returns:
        line scatter plot of epoch vs loss
    """
    plt.figure(figsize=(10,7))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if use_logscale:
        plt.yscale('log')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    if save == True:
        plt.savefig("%sloss_vs_epochs.png"%savefolder)

    plt.show()

def plot_history_from_list(loss,val,save=False,savefolder=None,logscale=False,ymin=None,ymax=None,title=None,variable="Zenith",pick_epoch=None,lr_start=None,lr_drop=None,lr_epoch=None):

    if logscale:
      print("using log scale")
      plt.yscale('log')
    
    fig,ax = plt.subplots(figsize=(10,7))
    epochs = np.arange(1,len(loss)+1)
    #ax.set_yscale('log')
    ax.plot(epochs,loss,'b',label="Training")
    ax.plot(epochs,val,'c',label="Validation")
   
    #Edit Axis
    if not ymin and not ymax:
        ymax = min(max(loss),max(val))
        ymin = min(min(loss),min(val))
        ymax = min(ymax,ymin+2)
    elif ymin and not ymax:
        ymax = min(max(loss),max(val))
    elif ymax and not ymin:
        ymin = min(min(loss),min(val))
    else:
        pass

    ax.set_ylim(ymin,ymax)
    
    if pick_epoch is not None:
        ax.axvline(pick_epoch,linewidth=4, color='g',alpha=0.5,label="Chosen Model")

    if lr_epoch is not None:
        epoch_drop = np.arange(0,len(loss),lr_epoch)
        for lr_print in range(len(epoch_drop)):
            lrate = lr_start*(lr_drop**lr_print)
            ax.axvline(epoch_drop[lr_print],linewidth=1, color='r',linestyle="--")
            ax.annotate(s='lrate='+str("{:.0e}".format(lrate)),xy=(epoch_drop[lr_print]+1,ymax),rotation=90,verticalalignment='top')

    #Add labels
    if title:
        plt.title(title,fontsize=25)
    
    plt.xlabel('Epochs',fontsize=20)
    if variable=="Energy":
        plt.ylabel(r'Loss = $\frac{100}{n}\sum_{i=1}^n \vert \frac{T_i - R_i}{T_i} \vert$',fontsize=20)
    elif variable=="Cosine Zenith":
        plt.ylabel(r'Loss = $\frac{1}{n}\sum_{i=1}^n ( T_i - R_i )^2$',fontsize=20)
    else:
        plt.ylabel('Loss',fontsize=20)
    plt.legend(loc="center right",fontsize=20)
    textstr = 'IceCube Work in Progress'
    ax=plt.gca()
    plt.text(0.48, 0.95, textstr, transform=ax.transAxes,color='gray')
    #plt.show()


    if save == True:
        plt.savefig("%sloss_vs_epochs.png"%savefolder,bbox_inches='tight') 
    plt.close()


def plot_history_from_list_split(energy_loss,val_energy_loss,zenith_loss,val_zenith_loss,save=True,savefolder=None,logscale=False,ymin=None,ymax=None,title=None):
    
    plt.figure(figsize=(10,7))
    plt.plot(energy_loss,'b',label="Zenith Training")
    plt.plot(val_energy_loss,'c',label="Zenith Validation")
    plt.plot(zenith_loss,'r',label="Zenith Training")
    plt.plot(val_zenith_loss,'m',label="Zenith Validation")
    
    #Edit Axis
    if logscale:
        plt.yscale('log')
    if ymin and ymax:
        plt.ylim(ymin,ymax)
    elif ymin:
        plt.ylim(ymin,max(max(loss),max(val)))
    elif ymax:
        plt.ylim(min(min(loss),min(val)),ymax)
    
    #Add labels
    if title:
        plt.title(title,fontsize=25)
    else:
        plt.title("Training and Validation Loss after %s Epochs"%len(energy_loss),fontsize=25)
    plt.xlabel('Epochs',fontsize=20)
    plt.ylabel('Loss',fontsize=20)
    plt.legend(fontsize=20)
    
    if save == True:
        plt.savefig("%sloss_vs_epochs_split.png"%savefolder)
    plt.close()

def plot_distributions_CCNC(truth_all_labels,truth,reco,save=False,savefolder=None):
    """
    Plot testing set distribution, with CC and NC distinguished
    Recieves:
        truth_all_labels = array, Y_test truth labels that have ALL values in them (need CC vs NC info)
        truth = array, Y_test truth labels
        reco = array, neural network prediction output
        save = optional, bool to save plot
        savefolder = optional, output folder to save to, if not in current dir
    Returns:
        1D histogram of reco - true with sepearated CC and NC distinction
    """
    CC_mask = truth_all_labels[:,11] ==1
    NC_mask = truth_all_labels[:,11] ==0
    num_CC = sum(CC_mask)
    num_NC = sum(NC_mask)
    print("CC events: %i, NC events: %i, Percent NC: %.2f"%(num_CC,num_NC,float(num_NC/(num_CC+num_NC))*100.))

    plt.figure(figsize=(10,7))
    plt.title("True Zenith Distribution",fontsize=25)
    plt.hist(truth[CC_mask], bins=100,color='b',alpha=0.5,label="CC");
    plt.hist(truth[NC_mask], bins=100,color='g',alpha=0.5,label="NC");
    plt.xlabel("Zenith",fontsize=20)
    plt.legend(fontsize=10)
    if save:
        plt.savefig("%sTrueZenithDistribution_CCNC.png"%savefolder)

    plt.figure(figsize=(10,7))
    plt.title("NN Zenith Distribution",fontsize=25)
    plt.hist(reco[CC_mask], bins=100,color='b', alpha=0.5, label="CC");
    plt.hist(reco[NC_mask], bins=100,color='g', alpha=0.5, label="NC");
    plt.xlabel("Zenith",fontsize=20)
    plt.legend(fontsize=10)
    if save:
        plt.savefig("%sNNZenithDistribution_CCNC.png"%savefolder)
    plt.close()

def plot_distributions(truth,reco=None,save=False,savefolder=None,old_reco=None,weights=None,variable="Zenith",units="",reco_name="Likelihood-based", minval=None, maxval=None,bins=100,cnn_name="CNN",log=False):
    plt.rcParams['pcolormesh.snap'] = True
    """
    Plot testing set distribution
    Recieves:
        truth = array, Y_test truth labels
        reco = array, neural network prediction output
        save = optional, bool to save plot
        savefolder = optional, output folder to save to, if not in current dir
        variable = string, variable name
        units = string, units for variable
    Returns:
        1D histogram of variable's absolute distribution for truth and for reco overlaid
    """
    if maxval is None:
        if old_reco is None:
            maxval = np.max([np.max(truth),np.max(reco)])
        else:
            maxval = np.max([np.max([np.max(truth),np.max(reco)]),np.max(old_reco)])
    if minval is None:
        if old_reco is None:
            minval = np.min([np.min(truth),np.min(reco)])
        else:
            minval = np.min([np.min([np.min(truth),np.min(reco)]),np.min(old_reco)])
    plt.figure(figsize=(10,7))
    name = ""
    if weights is not None:
        name += "Weighted"
#    plt.title("%s %s Distribution"%(name,variable),fontsize=25)
    plt.hist(truth, bins=bins,color='g',alpha=0.5,range=[minval,maxval],weights=weights,label="Truth");
    maskT = np.logical_and(truth > minval, truth < maxval)
    print("Truth Total: %i, Events in Plot: %i, Overflow: %i"%(len(truth),sum(maskT),len(truth)-sum(maskT)))
    name += "T"
    if reco is not None:
        plt.hist(reco, bins=bins,color='b', alpha=0.5,range=[minval,maxval],weights=weights,label=cnn_name);
        name += "R"
        maskR = np.logical_and(reco > minval, reco < maxval)
        print("Reco Total: %i, Events in Plot: %i, Overflow: %i"%(len(reco),sum(maskR),len(reco)-sum(maskR)))
    if old_reco is not None:
        plt.hist(old_reco, bins=bins,color='orange', alpha=0.5,range=[minval,maxval],weights=weights,label="Likelihood-based");
        name += "OR"
        maskOR = np.logical_and(old_reco > minval, old_reco < maxval)
        print("Old Reco Total: %i, Events in Plot: %i, Overflow: %i"%(len(old_reco),sum(maskOR),len(old_reco)-sum(maskOR)))
    var_label=variable
    if var_label=="Zenith":
      var_label=r"$\cos(zenith)$"
    plt.xlabel("%s %s"%(var_label,units),fontsize=22)
    plt.ylabel("Number of events")
    if log:
        #plt.yscale("log")
        plt.xscale("log")
    plt.legend(fontsize=20, loc="upper right")

    textstr = 'IceCube Work in Progress'
    ax=plt.gca()
    plt.text(0.48, 1.01, textstr, transform=ax.transAxes,color='gray')
#    plt.show()
    name += "%s"%variable.replace(" ","")
    if save:
        if log:
          plt.savefig("%s%sDistributionLog_%ito%i.png"%(savefolder,name,int(minval),int(maxval)))
        else:
          plt.savefig("%s%sDistribution_%ito%i.png"%(savefolder,name,int(minval),int(maxval)))

    plt.close()


def plot_2D_prediction(truth, nn_reco, \
                        save=False,savefolder=None,weights=None,syst_set="",\
                        bins=60,minval=None,maxval=None, switch_axis=False,\
                        cut_truth = False, axis_square =False, zmax=None,log=True,
                        variable="Zenith", units = "", epochs=None,reco_name="CNN"):
    """
    Plot testing set reconstruction vs truth
    Recieves:
        truth = array, Y_test truth
        nn_reco = array, neural network prediction output
        save = optional, bool to save plot
        savefolder = optional, output folder to save to, if not in current dir
        syst_set = string, name of the systematic set (for title and saving)
        bins = int, number of bins plot (will use for both the x and y direction)
        minval = float, minimum value to cut nn_reco results
        maxval = float, maximum value to cut nn_reco results
        cut_truth = bool, true if you want to make the value cut on truth rather than nn results
        axis_square = bool, cut axis to be square based on minval and maxval inputs
        variable = string, name of the variable you are plotting
        units = string, units for the variable you are plotting
    Returns:
        2D plot of True vs Reco
    """

    #hack!
    if cut_truth:

        if not minval:
            minval = min(truth)
        if not maxval:
            maxval= max(truth)
        mask1 = np.logical_and(truth >= minval, truth <= maxval)
        name = "True %s [%.2f, %.2f]"%(variable,minval,maxval)

    else:
        if not minval:
            minval = min([min(nn_reco),min(truth)])
        if not maxval:
            maxval= max([max(nn_reco),max(truth)])
        mask1 = np.ones(len(truth),dtype=bool) 
        #mask = np.logical_and(nn_reco >= minval, nn_reco <= maxval)
        name = "%s %s [%.2f, %.2f]"%(reco_name,variable,minval,maxval)
    
    cutting = False
    if axis_square:
        mask2 = np.logical_and(nn_reco >= minval, nn_reco <= maxval)
        overflow = abs(sum(mask1) - sum(mask2))
        print("Axis overflow: ",overflow)
        mask = np.logical_and(mask1, mask2)
    else:
        mask = mask1
    
    maxplotline = min([max(nn_reco),max(truth)])
    minplotline = max([min(nn_reco),min(truth)])
   
    truth = truth #[mask]
    nn_reco = nn_reco #[mask]
    

    #Cut axis
    if axis_square:
        xmin = minval
        ymin = minval
        xmax = maxval
        ymax = maxval
    else:
        xmin = min(truth)
        ymin = min(nn_reco)
        xmax = max(truth)
        ymax = max(nn_reco)
    if switch_axis:
        xmin, ymin = ymin, xmin
        xmax, ymax = ymax, xmax


    if weights is None:
        cmin = 1
    else:
        cmin = 1e-12
 
    plt.figure(figsize=(10,7))
    if log:
        if switch_axis:
            cts,xbin,ybin,img = plt.hist2d(nn_reco, truth, bins=bins,range=[[xmin,xmax],[ymin,ymax]], cmap='viridis_r', norm=colors.LogNorm(), weights=weights, cmax=zmax, cmin=cmin)
        else:
            cts,xbin,ybin,img = plt.hist2d(truth, nn_reco, bins=bins,range=[[xmin,xmax],[ymin,ymax]],cmap='viridis_r', norm=colors.LogNorm(), weights=weights, cmax=zmax, cmin=cmin)
    else:
        if switch_axis:
            cts,xbin,ybin,img = plt.hist2d(nn_reco, truth, bins=bins,range=[[xmin,xmax],[ymin,ymax]],cmap='viridis_r', weights=weights, cmax=zmax, cmin=cmin)
        else:
            cts,xbin,ybin,img = plt.hist2d(truth, nn_reco, bins=bins,range=[[xmin,xmax],[ymin,ymax]],cmap='viridis_r', weights=weights, cmax=zmax, cmin=cmin)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Number of events', rotation=90)
    var_label=variable
    if var_label=="Zenith":
      var_label=r"$\cos(zenith)$"
      xlabel="True %s %s"%(var_label,units)
      ylabel="%s reconstructed %s %s"%(reco_name,var_label,units)
    elif var_label=="biasvsTrueE":
      xlabel="Reco. - True cos(zenith)"
      ylabel="True Neutrino Energy (GeV)"
    elif var_label=="Energy":
      var_label=r"Energy (GeV)"
      xlabel="True %s %s"%(var_label,units)
      ylabel="%s reconstructed %s %s"%(reco_name,var_label,units)
    else:
      xlabel=var_label
      ylabel=var_label

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    if switch_axis:
        plt.ylabel(xlabel, fontsize=20)
        plt.xlabel(ylabel, fontsize=20)
    title = "%s vs Truth for %s %s"%(reco_name,variable,syst_set)
    if weights is not None:
        title += " Weighted"
    if epochs:
        title += " at %i Epochs"%epochs
#    plt.suptitle(title,fontsize=25)
    #if cutting:
    #    plt.title("%s, plotted %i, overflow %i"%(name,len(truth),overflow),fontsize=20)
    
    #Plot 1:1 line
    if axis_square:
        plt.plot([minval,maxval],[minval,maxval],'w:', linewidth=2)
    else:
        plt.plot([minplotline,maxplotline],[minplotline,maxplotline],'w:', linewidth=2)
    
    if switch_axis:
        x, y, y_l, y_u = find_contours_2D(nn_reco,truth,xbin,weights=weights)
    else:
        x, y, y_l, y_u = find_contours_2D(truth, nn_reco,xbin,weights=weights)

    plt.plot(x,y,color='r',label='Median', linewidth=2)
    plt.plot(x,y_l,color='r',label='68% band',linestyle='dashed', linewidth=2)
    plt.plot(x,y_u,color='r',linestyle='dashed', linewidth=2)
    plt.legend(fontsize=20)

    textstr = 'IceCube Work in Progress'
    ax=plt.gca()
    plt.text(0.35, 0.02, textstr, transform=ax.transAxes,color='white')

    
    reco_name = reco_name.replace(" ","")
    variable = variable.replace(" ","")
    nocut_name = ""
    if weights is not None:
        nocut_name="Weighted"
    if not axis_square:
        nocut_name ="_nolim"
    if zmax:
        nocut_name += "_zmax%i"%zmax    
    if switch_axis:
        nocut_name +="_SwitchedAxis"
    if save:
        plt.savefig("%sTruth%sReco%s_2DHist%s%s.png"%(savefolder,reco_name,variable,syst_set,nocut_name),bbox_inches='tight')
    plt.close()

def plot_2D_prediction_fraction(truth, nn_reco, \
                            save=False,savefolder=None,syst_set="",\
                            bins=60,minval=None,maxval=None,\
                            cut_truth = False, axis_square =False,
                            variable="Zenith", units = "",reco_name="CNN"):
    """
    Plot testing set reconstruction vs truth
    Recieves:
        truth = array, Y_test truth
        nn_reco = array, neural network prediction output
        save = optional, bool to save plot
        savefolder = optional, output folder to save to, if not in current dir
        syst_set = string, name of the systematic set (for title and saving)
        bins = int, number of bins plot (will use for both the x and y direction)
        minval = float, minimum value to cut (truth - nn_reco)/truth fractional results
        maxval = float, maximum value to cut (truth - nn_reco)/truth fractional results
        cut_truth = bool, true if you want to make the value cut on truth rather than nn results
        variable = string, name of the variable you are plotting
        units = string, units for the variable you are plotting
    Returns:
        2D plot of True vs (True - Reco)/True
    """

    fractional_error = abs(truth - nn_reco)/ truth
    
    if cut_truth:
        if not minval:
            minval = min(truth)
        if not maxval:
            maxval= max(truth)
        mask1 = np.logical_and(truth >= minval, truth <= maxval)
        if axis_square:
            mask2 = np.logical_and(nn_reco >= minval, nn_reco <= maxval)
            overflow = sum(mask1) - sum(mask2)
            print("OVerflow: ",overflow)
            mask = np.logical_and(mask1, mask2)
        else:
            mask = mask1
        name = "True %s [%.2f, %.2f]"%(variable,minval,maxval)

    else:
        if not minval:
            minval = min(fractional_error)
        if not maxval:
            maxval= max(fractional_error)
        mask = np.logical_and(fractional_error >= minval, fractional_error <= maxval)
        name = "%s %s in Fractional Error [%.2f, %.2f]"%(reco_name,variable,minval,maxval)

    #Check if cutting
    cutting = False
    if sum(mask)!= len(truth):
        overflow = len(nn_reco)-sum(mask)
        print("Making a cut for plotting, removing %i events"%(overflow))
        cutting = True
    maxplotline = min([max(nn_reco),max(truth)])
    minplotline = max([min(nn_reco),min(truth)])
    
    truth = truth[mask]
    fractional_error = fractional_error[mask]
    
    plt.figure(figsize=(10,7))

    cts,xbin,ybin,img = plt.hist2d(truth, fractional_error, bins=bins)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('counts', rotation=90)
    plt.set_cmap('viridis_r')
    #plt.xlim(min(truth),max(truth))
    #plt.ylim(min(fractional_truth),max(fractional_truth))
    var_label=variable
    if var_label=="Zenith":
      var_label=r"$\cos(zenith)$"
    
    plt.xlabel("True Neutrino %s %s"%(var_label,units),fontsize=20)
    plt.ylabel(r'Fractional Resolution: $\frac{reconstruction - truth}{truth}$',fontsize=20)
    plt.suptitle("%s Fractional Error vs. True %s %s"%(reco_name,variable,syst_set),fontsize=25)
    if cutting:
        plt.title("%s, plotted %i, overflow %i"%(name,len(truth),overflow),fontsize=20)
        
    #Plot 1:1 line
    #if cutting == True:
    #    plt.plot([minval,maxval],[minval,maxval],'k:',label="1:1")
    #else:
    #    plt.plot([minplotline,maxplotline],[minplotline,maxplotline],'k:',label="1:1")
    
    x, y, y_l, y_u = find_contours_2D(truth,fractional_error,xbin)
    plt.plot(x,y,color='r',label='Median')
    plt.plot(x,y_l,color='r',label='68% band',linestyle='dashed')
    plt.plot(x,y_u,color='r',linestyle='dashed')
    plt.legend(fontsize=12)
    
    variable = variable.replace(" ","")
    if cutting:
        nocut_name = ""
    else:
        nocut_name ="_nolim"
    if save:
        plt.savefig("%sTruth%sRecoFrac%s_2DHist%s%s.png"%(savefolder,reco_name,variable,syst_set,nocut_name),bbox_inches='tight')

def plot_resolution_CCNC(truth_all_labels,truth,reco,save=False,savefolder=None,variable="Zenith", units = ""):
    """
    Plot testing set resolution of reconstruction - truth, with CC and NC distinguished
    Recieves:
        truth_all_labels = array, Y_test truth labels that have ALL values in them (need CC vs NC info)
        truth = array, Y_test truth labels
        reco = array, neural network prediction output
        save = optional, bool to save plot
        savefolder = optional, output folder to save to, if not in current dir
    Returns:
        1D histogram of reco - true with sepearated CC and NC distinction
    """
    CC_mask = truth_all_labels[:,11] ==1
    NC_mask = truth_all_labels[:,11] ==0
    num_CC = sum(CC_mask)
    num_NC = sum(NC_mask)
    print("CC events: %i, NC events: %i, Percent NC: %.2f"%(num_CC,num_NC,float(num_NC/(num_CC+num_NC))*100.))

    resolution = reco - truth
    resolution_fraction = (reco - truth)/truth
    resolution = np.array(resolution)
    resolution_fraction  = np.array(resolution_fraction)

    plt.figure(figsize=(10,7))  
#    plt.title("%s Resolution"%variable)
    plt.hist(resolution[CC_mask], bins=50,color='b', alpha=0.5, label="CC");
    plt.hist(resolution[NC_mask], bins=50,color='g', alpha=0.5, label="NC");
    plt.xlabel("NN reconstruction - truth (%s)"%units)
    plt.legend()
    if save:
        plt.savefig("%s%sResolution_CCNC.png"%(savefolder,variable))

    plt.figure(figsize=(10,7))  
    plt.title("Fractional %s Resolution"%variable)
    plt.hist(resolution_fraction[CC_mask], bins=50,color='b', alpha=0.5, label="CC");
    plt.hist(resolution_fraction[NC_mask], bins=50,color='g', alpha=0.5, label="NC");
    plt.xlabel("(NN reconstruction - truth) / truth")
    plt.legend()

    variable = variable.replace(" ","")
    if save:
        plt.savefig("%s%sResolutionFrac_CCNC.png"%(savefolder,variable))
    plt.close()

def plot_single_resolution(truth,nn_reco,weights=None, \
                           bins=100, use_fraction=False,\
                           use_old_reco = False, old_reco=None,old_reco_truth=None,\
                           mintrue=None,maxtrue=None,\
                           minaxis=None,maxaxis=None,\
                           save=False,savefolder=None,
                           variable="Zenith", units = "", epochs=None,reco_name="CNN"):
    """Plots resolution for dict of inputs, one of which will be a second reco
    Recieves:
        truth = array of truth or Y_test labels
        nn_reco = array of NN predicted reco or Y_test_predicted results
        bins = int value
        use_fraction = use fractional resolution instead of absolute, where (reco - truth)/truth
        use_old_reco = True if you want to compare to another reconstruction (like pegleg)
        old_reco = optional, pegleg array of labels
        mintrue = float, min true value if cut desired
        maxtrue = float, max true value if cut desired
        minaxis = float, min x axis cut
        maxaxis = float, max x axis cut
    Returns:
        1D histogram of Reco - True (or fractional)
        Can have two distributions of NN Reco Resolution vs Pegleg Reco Resolution
    """

    if weights is not None:
        import wquantiles as wq
    fig, ax = plt.subplots(figsize=(10,7))

    ## Assume old_reco truth is the same as test sample, option to set it otherwise
    if old_reco_truth is None:
        truth2 = truth
    else:
        truth2 = old_reco_truth
    weights_reco = weights 
    
    if use_fraction:
        nn_resolution = (nn_reco - truth)/truth
        if use_old_reco:
            old_reco_resolution = (old_reco - truth2)/truth2
        title = "Fractional %s Resolution"%variable
        xlabel = r'$\frac{reconstruction - truth}{truth}$'
    else:
        nn_resolution = nn_reco - truth
        if use_old_reco:
            old_reco_resolution = old_reco - truth2
        title = "%s Resolution"%variable
        xlabel = "Reconstructed - True cos(zenith)"
    if epochs:
        title += " at %i Epochs"%epochs
    if weights is not None:
        title += " Weighted"
    original_size = len(nn_resolution)
 
    # Cut on true values
    #print(mintrue,maxtrue)
    if mintrue or maxtrue:
         truth_cut=True
    else:
         truth_cut=False
    if not mintrue:
        mintrue = int(min(truth))
    if not maxtrue:
        maxtrue = int(max(truth))+1
    #print(mintrue,maxtrue,truth_cut)
    if truth_cut:
        true_mask = np.logical_and(truth >= mintrue, truth <= maxtrue)
        title += "\n(true %s [%.2f, %.2f])"%(variable,mintrue,maxtrue)
        nn_resolution = nn_resolution[true_mask]
        if weights is not None:
            weights = weights[true_mask]
        if use_old_reco:
            true2_mask = np.logical_and(truth2 >= mintrue, truth2 <= maxtrue)
            old_reco_resolution = old_reco_resolution[true2_mask]
            if weights is not None:
                weights_reco = weights_reco[true2_mask]
    true_cut_size=len(nn_resolution)
    
    
    #Get stats before axis cut!
    rms_nn = get_RMS(nn_resolution,weights)
    if weights is not None:
        r1 = wq.quantile(nn_resolution,weights,0.16)
        r2 = wq.quantile(nn_resolution,weights,0.84)
        median = wq.median(nn_resolution,weights)
    else:
        r1, r2 = np.percentile(nn_resolution, [16,84])
        median = np.median(nn_resolution)
    if use_old_reco:
        true_cut_size_reco=len(old_reco_resolution)
        rms_old_reco = get_RMS(old_reco_resolution, weights=weights_reco)
        if weights is not None:
            r1_old_reco = wq.quantile(old_reco_resolution,weights_reco,0.16)
            r2_old_reco = wq.quantile(old_reco_resolution,weights_reco,0.84)
            median_old_reco = wq.median(old_reco_resolution,weights_reco)
        else:
            r1_old_reco, r2_old_reco = np.percentile(old_reco_resolution, [16,84])
            median_old_reco = np.median(old_reco_resolution)


    # Cut for plot axis
    #print(minaxis,maxaxis)
    if minaxis or maxaxis:
        axis_cut=True
    else:
        axis_cut=False
    if not minaxis:
        axis_cut=True
        minaxis = min(nn_resolution)
        if use_old_reco:
            if minaxis > min(old_reco_resolution):
                    minaxis = min(old_reco_resolution)
    if not maxaxis:
        axis_cut=True
        maxaxis = max(nn_resolution)
        if use_old_reco:
            if maxaxis < max(old_reco_resolution):
                    maxaxis = max(old_reco_resolution)
    if axis_cut:
        axis_mask = np.logical_and(nn_resolution >= minaxis, nn_resolution <= maxaxis)
        nn_resolution = nn_resolution[axis_mask]
        if weights is not None:
            weights = weights[axis_mask]
        if use_old_reco:
            reco_axis_mask = np.logical_and(old_reco_resolution >= minaxis, old_reco_resolution <= maxaxis)
            old_reco_resolution = old_reco_resolution[reco_axis_mask]
            reco_len = len(old_reco_resolution)
            if weights is not None:
                weights_reco = weights_reco[reco_axis_mask]
           
    true_axis_cut_size=len(nn_resolution)

    cnn_name = "CNN"  
    retro_name = "Likelihood-based"
    plt.ticklabel_format(axis="y", style="sci", scilimits=(1,0))
    hist_nn, bins, p = ax.hist(nn_resolution, bins=bins, range=[minaxis,maxaxis], weights=weights, alpha=0.5, label=cnn_name);
    if use_old_reco:
        stats1_label = cnn_name
        
    else:
        stats1_label ="Likelihood-based"

    #Statistics
    #weighted_avg_and_std(nn_resolution,weights)
    
    textstr = '\n'.join((
            r'%s' % (stats1_label),
            r'$\mathrm{events}=%i$' % (len(nn_resolution), ),
            r'$\mathrm{median}=%.2f$' % (median, ),
#            r'$\mathrm{overflow}=%i$' % (true_cut_size-true_axis_cut_size, ),
            r'$\mathrm{RMS}=%.2f$' % (rms_nn, ) ))
#            r'$\mathrm{1\sigma}=%.2f,%.2f$' % (r1,r2 )))
    props = dict(boxstyle='round',facecolor='blue', alpha=0.5)
    ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=20,
        verticalalignment='top', bbox=props)

    if use_old_reco:
        ax.hist(old_reco_resolution, bins=bins, range=[minaxis,maxaxis], weights=weights_reco, alpha=0.5, label="%s"%retro_name);
#        ax.legend(loc="upper left",fontsize=20)
        textstr = '\n'.join((
            '%s' % (retro_name),
#            r'$\mathrm{events}=%i$' % (len(old_reco_resolution), ),
            r'$\mathrm{median}=%.2f$' % (median_old_reco, ),
#            r'$\mathrm{overflow}=%i$' % (true_cut_size_reco-reco_len, ),
            r'$\mathrm{RMS}=%.2f$' % (rms_old_reco, ) ))
#            r'$\mathrm{1\sigma}=%.2f,%.2f$' % (r1_old_reco,r2_old_reco )))
        props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
        ax.text(0.6, 0.65, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)
        ymax=max(np.max(old_reco_resolution), np.max(nn_resolution))*1.3
    #if axis_cut:
    ax.set_xlim(minaxis,maxaxis)
    ax.set_xlabel(xlabel,fontsize=20)
    ax.set_ylabel("Number of events", fontsize=20)
#    ax.set_title(title,fontsize=25)
    textstr = 'IceCube Work in Progress'
    ax=plt.gca()
    plt.text(0.48, 1.01, textstr, transform=ax.transAxes,color='gray')
  
    reco_name = reco_name.replace(" ","")
    variable = variable.replace(" ","")
    savename = "%sResolution"%variable
    if weights is not None:
        savename+="Weighted"
    if use_fraction:
        savename += "Frac"
    if truth_cut:
        savename = "_Range%.2f_%.2f"%(mintrue,maxtrue)
    if use_old_reco:
        savename += "_Compare%sReco"%reco_name
    if axis_cut:
        savename += "_xlim"
    if save == True:
        plt.savefig("%s%s.png"%(savefolder,savename),bbox_inches='tight')
    plt.close()

def plot_compare_resolution(truth,nn_reco,namelist, savefolder=None,\
                            num_namelist = None,save=False,bins=100,use_fraction=False):
    """Plots resolution for dict of inputs
    Receives:
        truth = dict of truth or Y_test labels
                (contents = [key name, energy], shape = [number syst sets, number of events])
        nn_reco = dict of NN predicted or Y_test_predicted results
                    (contents = [key name, energy], shape = [number syst sets, number of events])
        namelist = list of names for the dict, to use as pretty labels
        save_folder_name = string for output file
        num_namelist = shorthand for names of sets (numerical version), for printing
        save = bool where True saves and False does not save plot
        bins = int value
        use_fraction: bool, uses fractional resolution if True
    Returns:
        Histograms of resolutions for systematic sets, overlaid
        Prints statistics for all histograms into table
    """
    
    print("Resolution")
    print('Name\t Mean\t Median\t RMS\t Percentiles\t')
    plt.figure(figsize=(10,7)) 
    if use_fraction:
        title = "Fractional Zenith Resolution"
        xlabel = "(NN reconstruction - truth) / truth"
        plt.legend(fontsize=20)
    else:
        title = "Zenith Resolution"
        xlabel = "NN reconstruction - truth "
        plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    for index in range(0,len(namelist)):
        keyname = "file_%i"%index
        if use_fraction:
            resolution = (nn_reco[keyname] - truth[keyname]) / truth[keyname]
        else:
            resolution = nn_reco[keyname] - truth[keyname]
        plt.hist(resolution, bins=60, alpha=0.5, label="%s"%namelist[index]);
        
        #Statistics
        rms = get_RMS(resolution)
        #r1, r2 = get_FWHM(resolution,bins)
        r1, r2 = np.percentile(resolution, [16,84])
        
        if num_namelist:
            names = num_namelist
        else:
            names = namelist
            
        print("%s\t %.2f\t %.2f\t %.2f\t %.2f, %.2f\t"%(names[index], \
                                                        np.mean(resolution),\
                                                        np.median(resolution),\
                                                        rms,\
                                                        r1, r2))
    plt.title(title)    
    plt.xlabel(xlabel)
    
    if save:
        if use_fraction:
            plt.savefig("%sFractionalZenithResolution_CompareSets.png"%savefolder)
        else:
            plt.savefig("%sZenithResolution_CompareSets.png"%savefolder)
    plt.close()

def plot_systematic_slices(truth_dict, nn_reco_dict,\
                           namelist, use_fraction=False, \
                           use_old_reco = False, old_reco_dict=None,\
                           save=False,savefolder=None):
    """Plots different arrays vs each other (systematic set arrays)
    Receives:
        truth_dict = dict of arrays with truth labels
                    (contents = [key name, energy], shape = [number syst sets, number of events])
        nn_reco_dict = dict of arrays that has NN predicted reco results
                        (contents = [key name, energy], shape = [number syst sets, number of events])
        namelist = list of names to be used for x_axis ticks
        use_fraction = use fractional resolution instead of absolute, where (reco - truth)/truth
        use_reco = True if you want to compare to another reconstruction (like pegleg)
                    (contents = [key name, energy], shape = [number syst sets, number of events])
        old_reco = optional, dict of pegleg arrays with the labels
    Returns:
        "scatter plot" with systematic sets on x axis,
        y axis has median of resolution with error bars containing 68% of resolution
    """
    
    number_sets = len(namelist)
    percentile_in_peak = 68.27

    left_tail_percentile  = (100.-percentile_in_peak)/2
    right_tail_percentile = 100.-left_tail_percentile
    
    medians  = np.zeros(number_sets)
    err_from = np.zeros(number_sets)
    err_to   = np.zeros(number_sets)
    
    if use_old_reco:
        medians_old_reco  = np.zeros(number_sets)
        err_from_old_reco = np.zeros(number_sets)
        err_to_old_reco   = np.zeros(number_sets)
    
    resolution = {}
    for index in range(0,number_sets):
        keyname = "file_%i"%index
        if use_fraction:
            resolution = (nn_reco_dict[keyname] - truth_dict[keyname])/truth_dict[keyname]
        else:
            resolution = (nn_reco_dict[keyname] - truth_dict[keyname])
    
        lower_lim = np.percentile(resolution, left_tail_percentile)
        upper_lim = np.percentile(resolution, right_tail_percentile)
        median = np.percentile(resolution, 50.)
        
        medians[index] = median
        err_from[index] = lower_lim
        err_to[index] = upper_lim
    
        if use_old_reco:
            if use_fraction:
                resolution_old_reco = ((old_reco_dict[keyname]-truth_dict[keyname])/truth_dict[keyname])
            else:
                resolution_old_reco = (old_reco_dict[keyname]-truth_dict[keyname])
            
            lower_lim_reco = np.percentile(resolution_old_reco, left_tail_percentile)
            upper_lim_reco = np.percentile(resolution_old_reco, right_tail_percentile)
            median_reco = np.percentile(resolution_old_reco, 50.)
            
            medians_reco[index] = median_reco
            err_from_reco[index] = lower_lim_reco
            err_to_reco[index] = upper_lim_reco


    x_range = np.linspace(1,number_sets,number_sets)
    
    cnn_name = "Neural Network"
    fig, ax = plt.subplots(figsize=(10,7))
    plt.errorbar(x_range, medians, yerr=[medians-err_from, err_to-medians],  capsize=5.0, fmt='o',label=cnn_name)
    if use_old_reco:
        plt.errorbar(x_range, medians_reco, yerr=[medians_reco-err_from_reco, err_to_reco-medians_reco], capsize=5.0,fmt='o',label="Pegleg Reco")
        plt.legend(loc="upper right")
    ax.plot([0,number_sets+1], [0,0], color='k')
    ax.set_xlim(0,number_sets+1)
    
    #rename axis
    my_xlabels = [item.get_text() for item in ax.get_xticklabels()]
    new_namelist = [" "] + namelist
    for index in range(0,number_sets+1):
        my_xlabels[index] = new_namelist[index]
    ax.set_xticklabels(my_xlabels)
    
    ax.set_xlabel("Systematic Set")
    if use_fraction:
        ax.set_ylabel(r'Fractional Resolution: $\frac{reconstruction - truth}{truth}$')
    else:
        ax.set_ylabel("Resolution: \n reconstruction - truth ")
    ax.set_title("Resolution Zenith Dependence")
    
    savename = "SystematicResolutionCompare"
    if use_fraction:
        savename += "Frac"
    if use_old_reco:
        savename += "_CompareOldReco"
    if save == True:
        plt.savefig("%s%s.png"%(savefolder,savename))
    plt.close()

def plot_bin_slices(truth, nn_reco, energy_truth=None, weights=None,\
                       energy_reco=None, energy_old_reco=None,
                       use_fraction = False, old_reco=None,old_reco_truth=None, reco_energy_truth=None,\
                       bins=20,min_val=-1.,max_val=1., ylim = None,\
                       save=False,savefolder=None,vs_predict=False,\
                       variable="Zenith",units="",xvariable="Zenith",xunits="",\
                       epochs=None,reco_name="Likelihood-based"):
    """Plots different variable slices vs each other (systematic set arrays)
    Receives:
        truth= array with truth labels for this one variable
        nn_reco = array that has NN predicted reco results for one variable (same size of truth)
        energy_truth = optional (will use if given), array that has true energy information (same size of truth)
        use_fraction = bool, use fractional resolution instead of absolute, where (reco - truth)/truth
        old_reco = optional (will use if given), array of pegleg labels for one variable
        bins = integer number of data points you want (range/bins = width)
        min_val = minimum value for variable to start cut at (default = 0.)
        max_val = maximum value for variable to end cut at (default = 60.)
        ylim = List with two entries of ymin and ymax for plot [min, max], leave as None for no ylim applied
    Returns:
        Scatter plot with energy values on x axis (median of bin width)
        y axis has median of resolution with error bars containing 68% of resolution
    """
    if weights is not None:
        import wquantiles as wq
    nn_reco = np.array(nn_reco)
    truth = np.array(truth)
     ## Assume old_reco truth is the same as test sample, option to set it otherwise
    if old_reco_truth is None:
        truth2 = np.array(truth)
    else:
        truth2 = np.array(old_reco_truth)
    if reco_energy_truth is None:
        energy_truth2 = np.array(energy_truth)
    else:
        energy_truth2 = np.array(reco_energy_truth)

    if use_fraction:
        resolution = ((nn_reco-truth)/truth) # in fraction
    else:
        resolution = (nn_reco-truth)
    resolution = np.array(resolution)
    percentile_in_peak = 68.27

    left_tail_percentile  = (100.-percentile_in_peak)/2
    right_tail_percentile = 100.-left_tail_percentile

    variable_ranges  = np.linspace(min_val,max_val, num=bins+1)
    variable_centers = (variable_ranges[1:] + variable_ranges[:-1])/2.

    medians  = np.zeros(len(variable_centers))
    err_from = np.zeros(len(variable_centers))
    err_to   = np.zeros(len(variable_centers))

    if old_reco is not None:
        if use_fraction:
            resolution_reco = ((old_reco-truth2)/truth2)
        else:
            resolution_reco = (old_reco-truth2)
        resolution_reco = np.array(resolution_reco)
        medians_reco  = np.zeros(len(variable_centers))
        err_from_reco = np.zeros(len(variable_centers))
        err_to_reco   = np.zeros(len(variable_centers))

    for i in range(len(variable_ranges)-1):
        var_from = variable_ranges[i]
        var_to   = variable_ranges[i+1]
        
        if vs_predict:
          if energy_reco is None and energy_old_reco is None:
            x_axis_array = nn_reco
            x_axis_array2 = old_reco #syu
            title="%s Resolution Dependence"%(variable)
          else:
            x_axis_array = energy_reco
            x_axis_array2 = energy_old_reco
        else:
            if energy_truth is None:
                title="%s Resolution Dependence"%(variable)
                x_axis_array = truth
                x_axis_array2 = truth2
            else:
                title="%s Resolution %s Dependence"%(variable,xvariable)
                energy_truth = np.array(energy_truth)
                x_axis_array = energy_truth
                x_axis_array2 = energy_truth2
                
        cut = (x_axis_array >= var_from) & (x_axis_array < var_to)
        cut2 = (x_axis_array2 >= var_from) & (x_axis_array2 < var_to)

        if weights is not None:
            lower_lim = wq.quantile(resolution[cut],weights[cut],0.16)
            upper_lim = wq.quantile(resolution[cut],weights[cut],0.84)
            median = wq.median(resolution[cut],weights[cut])
        else:
            lower_lim = np.percentile(resolution[cut], left_tail_percentile)
            upper_lim = np.percentile(resolution[cut], right_tail_percentile)
            median = np.percentile(resolution[cut], 50.)

        medians[i] = median
        err_from[i] = lower_lim
        err_to[i] = upper_lim

        if old_reco is not None:
            if weights is not None:
                lower_lim_reco = wq.quantile(resolution_reco[cut2],weights[cut2],0.16)
                upper_lim_reco = wq.quantile(resolution_reco[cut2],weights[cut2],0.84)
                median_reco = wq.median(resolution_reco[cut2],weights[cut2])
            else:
                lower_lim_reco = np.percentile(resolution_reco[cut2], left_tail_percentile)
                upper_lim_reco = np.percentile(resolution_reco[cut2], right_tail_percentile)
                median_reco = np.percentile(resolution_reco[cut2], 50.)

            medians_reco[i] = median_reco
            err_from_reco[i] = lower_lim_reco
            err_to_reco[i] = upper_lim_reco

    cnn_name = "CNN"
    reco_name = "Likelihood-based"
    #cnn_name = "Neural Network"
    plt.figure(figsize=(10,7))
    cmap = plt.get_cmap('Blues')
    colors = cmap(np.linspace(0, 1, 2 + 1))[1:]
    color=colors[0]
    cmap = plt.get_cmap('Oranges')
    rcolors = cmap(np.linspace(0, 1, 2 + 1))[1:]
    rcolor=rcolors[0]
    alpha=0.5
    lwid=2
    ax = plt.gca()
#    plt.errorbar(variable_centers, medians, yerr=[medians-err_from, err_to-medians], xerr=[ variable_centers-variable_ranges[:-1], variable_ranges[1:]-variable_centers ], capsize=5.0, fmt='o',label=cnn_name)
    ax.plot(variable_centers, medians,linestyle='-',label="%s median"%(cnn_name), color=color, linewidth=lwid)
    ax.fill_between(variable_centers,medians, err_from,color=color, alpha=alpha)
    ax.fill_between(variable_centers,medians, err_to, color=color, alpha=alpha) #, label=cnn_name+" 68%")
    max_y=np.max(err_to) *1.3
    min_y=np.min(err_from)*1.3
    if old_reco is not None:
#        plt.errorbar(variable_centers, medians_reco, yerr=[medians_reco-err_from_reco, err_to_reco-medians_reco], xerr=[ variable_centers-variable_ranges[:-1], variable_ranges[1:]-variable_centers ], capsize=5.0, fmt='o',label="%s"%reco_name)
        ax.plot(variable_centers,medians_reco, color=rcolor, linestyle='-', label="%s median"%reco_name, linewidth=lwid)
        ax.fill_between(variable_centers,medians_reco,err_from_reco, color=rcolor, alpha=alpha)#label=reco_name+" 68%", alpha=0.5)
        ax.fill_between(variable_centers,medians_reco,err_to_reco, color=rcolor,alpha=alpha)
        max_reco=np.max(err_to_reco)*1.3
        min_reco=np.min(err_from_reco)*1.3
#        ax.plot(variable_centers, err_from_reco, color='r', linestyle='--', label=reco_name+" 68%")
#        ax.plot(variable_centers, err_to_reco, color='r', linestyle='--')
        max_yval=max(max_reco, max_y)
        min_yval=min(min_reco,min_y)
        if type(min_yval) is not None and type(max_yval) is not None:
          plt.ylim(min_yval,max_yval)

        if type(ylim) is not None:
          plt.ylim(ylim)

    shade=ax.fill_between([-1],[-1],[-1],color='gray', alpha=0.8,label="68% of events")
    if vs_predict:
      plt.legend(loc="upper right")
    else:
      plt.legend(loc="upper right")
    plt.plot([min_val,max_val], [0,0], color='k')
    plt.xlim(min_val,max_val)
    var_label=variable
    if var_label=="Zenith":
      var_label=r"$\cos(zenith)$"
        
    if vs_predict:
      if energy_reco is None and energy_old_reco is None:
        plt.xlabel("Reconstructed %s %s"%(var_label,units),fontsize=20)
      else:
        plt.xlabel("Reconstructed Neutrino Energy (GeV)")
    else:
      if energy_truth is None:
        plt.xlabel("True %s %s"%(var_label,units),fontsize=20)
      else:
        plt.xlabel("True Neutrino Energy (GeV)")
    if use_fraction:
        plt.ylabel(r'Fractional Resolution: $\frac{reconstruction - truth}{truth}$',fontsize=20)
    else:
         plt.ylabel("Reconstructed - True %s %s"%(var_label, units),fontsize=20)
    #if epochs:
    #    title += " at %i Epochs"%epochs
    #plt.title(title,fontsize=25)

    textstr = 'IceCube Work in Progress'
    ax=plt.gca()
    plt.text(0.48, 1.01, textstr, transform=ax.transAxes,color='gray')
    
    reco_name = reco_name.replace(" ","")
    variable = variable.replace(" ","")
    savename = "%sResolutionSlices"%variable
    if vs_predict:
        savename +="VsPredict"
    if use_fraction:
        savename += "Frac"
    if weights is not None:
        savename += "Weighted"
    if energy_truth is not None:
        xvar_no_space = xvariable.replace(" ","")
        savename += "_%sBinned"%xvar_no_space
    if old_reco is not None:
        savename += "_Compare%sReco"%reco_name
    if type(ylim) is not None:
        savename += "_ylim"
    if save == True:
        plt.savefig("%s%s.png"%(savefolder,savename),bbox_inches='tight')
    plt.close()

def plot_rms_slices(truth, nn_reco, energy_truth=None, use_fraction = False,  \
                       old_reco=None,old_reco_truth=None, reco_energy_truth=None,\
                       bins=10,min_val=0.,max_val=60., ylim = None,weights=None,\
                       save=False,savefolder=None, vs_predict=False,\
                       variable="Zenith",units="",epochs=None,reco_name="PegLeg"):
    """Plots different variable slices vs each other (systematic set arrays)
    Receives:
        truth= array with truth labels for this one variable
        nn_reco = array that has NN predicted reco results for one variable (same size of truth)
        energy_truth = optional (will use if given), array that has true energy information (same size of truth)
        use_fraction = bool, use fractional resolution instead of absolute, where (reco - truth)/truth
        old_reco = optional (will use if given), array of pegleg labels for one variable
        bins = integer number of data points you want (range/bins = width)
        min_val = minimum value for variable to start cut at (default = 0.)
        max_val = maximum value for variable to end cut at (default = 60.)
        ylim = List with two entries of ymin and ymax for plot [min, max], leave as None for no ylim applied
    Returns:
        Scatter plot with energy values on x axis (median of bin width)
        y axis has median of resolution with error bars containing 68% of resolution
    """
    nn_reco = np.array(nn_reco)
    truth = np.array(truth)
     ## Assume old_reco truth is the same as test sample, option to set it otherwise
    if old_reco_truth is None:
        truth2 = np.array(truth)
    else:
        truth2 = np.array(old_reco_truth)
    if reco_energy_truth is None:
        energy_truth2 = np.array(energy_truth)
    else:
        energy_truth2 = np.array(reco_energy_truth)

    if use_fraction:
        resolution = ((nn_reco-truth)/truth) # in fraction
    else:
        resolution = (nn_reco-truth)
    resolution = np.array(resolution)

    variable_ranges  = np.linspace(min_val,max_val, num=bins+1)
    variable_centers = (variable_ranges[1:] + variable_ranges[:-1])/2.

    rms_all  = np.zeros(len(variable_centers))
    mean_all = np.zeros(len(variable_centers))
    if old_reco is not None:
        if use_fraction:
            resolution_reco = ((old_reco-truth2)/truth2)
        else:
            resolution_reco = (old_reco-truth2)
        resolution_reco = np.array(resolution_reco)
        rms_reco_all = np.zeros(len(variable_centers))
        mean_reco_all = np.zeros(len(variable_centers))
    for i in range(len(variable_ranges)-1):
        var_from = variable_ranges[i]
        var_to   = variable_ranges[i+1]
        
        if weights is None:
            title=""
        else:
            title="Weighted "

        title="%s RMS Resolution Dependence"%(variable)

        if vs_predict:
          x_axis_array = nn_reco
          x_axis_array2 = old_reco #syu

#          x_axis_array2 = nn_reco
        else:
          if energy_truth is None:
            x_axis_array = truth
            x_axis_array2 = truth2
          else:
            energy_truth = np.array(energy_truth)
            x_axis_array = energy_truth
            x_axis_array2 = energy_truth2

        cut = (x_axis_array >= var_from) & (x_axis_array < var_to)
        cut2 = (x_axis_array2 >= var_from) & (x_axis_array2 < var_to)

        if weights is not None:
            weight_here = weights[cut]
            weight_here_reco = weights[cut2]
        else:
            weight_here = None
            weight_here_reco = None
        rms = get_RMS(resolution[cut],weight_here)
        rms_all[i] = rms
        mean=np.average(resolution[cut],weights=weight_here)
        mean_all[i]=mean
        if old_reco is not None:
            rms_reco = get_RMS(resolution_reco[cut2], weight_here_reco) #syu
            rms_reco_all[i] = rms_reco
            mean_reco = np.average(resolution_reco[cut2],weights=weight_here_reco) #syu
            mean_reco_all[i]=mean_reco
    cnn_name = "CNN"
    reco_name="Likelihood-based"

    diff_width=abs(variable_ranges[1] - variable_ranges[0])
    plt.figure(figsize=(10,7))

    rms_all = np.append(rms_all,rms_all[-1])
    plt.step(variable_ranges, rms_all, where='post', color="blue",label=cnn_name)

    if old_reco is not None:
        rms_reco_all = np.append(rms_reco_all, rms_reco_all[-1])
        plt.step(variable_ranges, rms_reco_all, where='post', color="orange",label="%s"%reco_name)

    plt.legend(fontsize=20)
    ymax=max(np.max(rms_reco_all),np.max(rms_all))*1.3
    if type(ymax) is not None:
      plt.ylim(0,ymax)

    plt.xlim(min_val,max_val)

    if vs_predict:
      plt.xlabel("Reconstructed %s %s"%(variable,units),fontsize=20)
    else:
      plt.xlabel("True %s %s"%(variable,units),fontsize=20)

    if use_fraction:
        if weights is not None:
            plt.ylabel(r'Weighted RMS of Fractional Resoltion: $\frac{reconstruction - truth}{truth}$',fontsize=20)
        else:
            plt.ylabel(r'RMS of Fractional Resoltion: $\frac{reconstruction - truth}{truth}$',fontsize=20)
    else:
        plt.ylabel("RMS of Reco. - True cos(zenith)%s"%units,fontsize=20)


    textstr = 'IceCube Work in Progress'
    ax=plt.gca()
    plt.text(0.48, 0.02, textstr, transform=ax.transAxes,color='gray')

    #if epochs:
    #    title += " at %i Epochs"%epochs
#    plt.title(title,fontsize=25)
    
    reco_name = reco_name.replace(" ","")
    variable = variable.replace(" ","")
    savename = "%sRMSSlices"%variable
    if use_fraction:
        savename += "Frac"
    if vs_predict:
        savename +="VsPredict"
    if weights is not None:
        savename += "Weighted"
    if energy_truth is not None:
        savename += "_EnergyBinned"
        plt.xlabel("True Energy (GeV)",fontsize=20)
    if old_reco is not None:
        savename += "_Compare%sReco"%reco_name
    if type(ylim) is not None:
        savename += "_ylim"
    if save == True:
        plt.savefig("%s%s.png"%(savefolder,savename),bbox_inches='tight')
    plt.close()

def imshow_plot(array,name,emin,emax,tmin,tmax,zlabel,savename):
    
    afig = plt.figure(figsize=(10,7))
    plt.imshow(array,origin='lower',extent=[emin,emax,tmin,tmax],aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label(zlabel,rotation=90,fontsize=20)
    plt.set_cmap('viridis_r')
    cbar.ax.tick_params(labelsize=20) 
    plt.xlabel("True Zenith ",fontsize=20)
    plt.ylabel("True Track Length (m)",fontsize=20)
    plt.title("%s for Track Length vs. Zenith"%name,fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(savename,bbox_inches='tight')
    return afig
    
def plot_length_energy(truth, nn_reco, track_index=2,tfactor=200.,\
                        save=False,savefolder=None,use_fraction=False,\
                        ebins=10,tbins=10,emin=5.,emax=100.,tmin=0.,tmax=430.,\
                        cut_truth = False, axis_square =False, zmax=None,
                        variable="Zenith", units = "", epochs=None,reco_name="CNN"):
   

    true_energy = truth[:,0]*emax
    true_track =  truth[:,track_index]*tfactor
    #nn_reco = nn_reco[:,0]*emax
    
    #print(true_energy.shape,nn_reco.shape)
    if use_fraction:
        resolution = (nn_reco - true_energy)/true_energy
        title = "Fractional %s Resolution"%variable
        zlabel = "(reco - truth) / truth" 
    else:
        resolution = nn_reco - true_energy
        title = "%s Resolution"%variable
        zlabel = "reconstruction - truth"
    #print(nn_reco[:10],true_energy[:10])    
        
    percentile_in_peak = 68.27
    left_tail_percentile  = (100.-percentile_in_peak)/2
    right_tail_percentile = 100.-left_tail_percentile
    
    
    energy_ranges  = np.linspace(emin,emax, num=ebins+1)
    energy_centers = (energy_ranges[1:] + energy_ranges[:-1])/2.
    track_ranges  = np.linspace(tmin,tmax, num=tbins+1)
    track_centers = (track_ranges[1:] + track_ranges[:-1])/2.

    medians  = np.zeros((len(energy_centers),len(track_centers)))
    err_from = np.zeros((len(energy_centers),len(track_centers)))
    err_to   = np.zeros((len(energy_centers),len(track_centers)))
    rms      = np.zeros((len(energy_centers),len(track_centers)))
    
    #print(energy_ranges,track_ranges)
    for e in range(len(energy_ranges)-1):
        e_from = energy_ranges[e]
        e_to   = energy_ranges[e+1]
        for t in range(len(track_ranges)-1):
            t_from = track_ranges[t]
            t_to   = track_ranges[t+1]
            
        
            e_cut = (true_energy >= e_from) & (true_energy < e_to)
            t_cut = (true_track >= t_from) & (true_track < t_to)
            cut = e_cut & t_cut

            subset = resolution[cut]
            #print(subset)
            #print(e_from,e_to,t_from,t_to,true_energy[cut],true_track[cut])
            if sum(cut)==0:
                lower_lim = np.nan
                upper_lim = np.nan
                median    = np.nan
                one_rms   = np.nan
            else:
                lower_lim = np.percentile(subset, left_tail_percentile)
                upper_lim = np.percentile(subset, right_tail_percentile)
                median = np.percentile(subset, 50.)
                mean_array = np.ones_like(subset)*np.mean(subset)
                one_rms = np.sqrt( sum((mean_array - subset)**2)/len(subset) )
            #Invert saving because imshow does (M,N) where M is rows and N is columns
            medians[t,e] = median
            err_from[t,e] = lower_lim
            err_to[t,e] = upper_lim
            rms[t,e] = one_rms
    
    stat=["Median", "Lower 1 sigma", "Upper 1 sigma", "RMS"]
    z_name = [zlabel, "lower 1 sigma of " + zlabel, "upper 1 sigma of " + zlabel, "RMS of " + zlabel ]
    
    savename = "%sTrueZenithTrackReco%s_2DHist_%s.png"%(savefolder,reco_name,stat[0])
    imshow_plot(medians,stat[0],emin,emax,tmin,tmax,z_name[0],savename)
    
    savename="%sTrueZenithTrackReco%s_2DHist_%s.png"%(savefolder,reco_name,"LowSigma")
    imshow_plot(err_from,stat[1],emin,emax,tmin,tmax,z_name[1],savename)
    
    savename="%sTrueZenithTrackReco%s_2DHist_%s.png"%(savefolder,reco_name,"HighSigma")
    imshow_plot(err_to,stat[2],emin,emax,tmin,tmax,z_name[2],savename)
    
    savename="%sTrueZenithTrackReco%s_2DHist_%s.png"%(savefolder,reco_name,stat[3])
    imshow_plot(rms,stat[3],emin,emax,tmin,tmax,z_name[3],savename)

def plot_input_3D(X_values_IC,X_values_DC,Y_labels, Y_labels_pred, outdir,evtInd=0, varInd=0, filename="",cutrange=""):
    from utils.plot_detector import detector_3d
    name = ["Charge (p.e.)", "Raw Time of First Pulse (ns)", "Raw Time of Last Pulse (ns)", "Charge weighted mean of pulse times", "Charge weighted std of pulse times"]
    strings_19 = [17, 18, 19, 25, 26, 27, 28, 34, 35, 36, 37, 38, 44, 45, 46, 47, 54, 55, 56]
    strings_8 = [84, 85, 79, 80, 83, 86, 81, 82]
    if filename == "IC":
       strings = strings_19
    elif filename == "DC":
       strings = strings_8
    else:
       strings = strings_8

    print(X_values_IC.shape,X_values_DC.shape)
    IC_vals=X_values_IC[evtInd,:,:,varInd].flatten()
    DC_vals=X_values_DC[evtInd,:,:,varInd].flatten()
    print(IC_vals.shape,DC_vals.shape)
    Y_true=Y_labels[evtInd,1]
    Y_pred=Y_labels_pred[evtInd,0]
    x,y,z,val=detector_3d(strings_19, strings_8,IC_vals, DC_vals)
    print(val.shape, x.shape)
    x=x[val>0]
    y=y[val>0]
    z=z[val>0]
    val=val[val>0]
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img=ax.scatter(x,y,z,s=5,c=val, cmap=plt.get_cmap('hsv'))
    print(x,y,z)

    vx=Y_labels[evtInd,4]
    vy=Y_labels[evtInd,5]
    vz=Y_labels[evtInd,6]
#    ax.scatter(vx,vy,vz, 'rx')
    ax.set_title('T:%3.3f, P:%3.3f'%(Y_true,Y_pred))
    ax.plot(x, z, 'r+', zdir='y',zs=250)
    ax.plot(y, z, 'g+', zdir='x',zs=-250)
    ax.plot(x, y, 'k+', zdir='z',zs=-550)
    fig.colorbar(img)
#    plt.show()
    plt.savefig("%s/Input_evt_%i_Variable%i_%s_%s_3D.png"%(outdir,evtInd,varInd,filename,cutrange))
