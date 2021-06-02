import h5py
import argparse
import os, sys
import numpy as np
import dama as dm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default=None,
                    dest="input_file", help="path and name of the input file")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="output_dir", help="path of ouput file")
parser.add_argument("--numu",type=float,default=1518.,
                    dest="numu", help="number of numu files")
parser.add_argument("--nue",type=float,default=602.,
                    dest="nue", help="number of nue files")
parser.add_argument("--no_old_reco", default=False,action='store_true',
                        dest='no_old_reco',help="no old reco")
parser.add_argument("--nu_type",type=str,default="NuMu",
                    dest="nu_type", help="NuMu or NuE")
args = parser.parse_args()

input_file = args.input_file
save_folder_name = args.output_dir
numu_files = args.numu
nue_files = args.nue
nu_type = args.nu_type

f = h5py.File(input_file, "r")
truth = f["Y_test_use"][:]
predict = f["Y_predicted"][:]
try:
    info = f["additional_info"][:]
except:
    info = None
if args.no_old_reco:
    reco = None
    weights = None
else:
    try:
        reco = f["reco_test"][:]
    except:
        reco = None
    try:
        weights = f["weights_test"][:]
    except:
        weights = None
f.close()
del f


cnn_energy = np.array(predict[:,0])
try:
    cnn_class = np.array(predict[:,1])
except:
    pass
try:
    cnn_zenith = np.array(predict[:,2])
    cnn_coszenith = np.cos(cnn_zenith)
except:
    pass
try:
    cnn_x = np.array(predict[:,3])
    cnn_y = np.array(predict[:,4])
    cnn_z = np.array(predict[:,5])
except:
    pass

#Truth
true_energy = np.array(truth[:,0])
em_equiv_energy = np.array(truth[:,14])
true_x = np.array(truth[:,4])
true_y = np.array(truth[:,5])
true_z = np.array(truth[:,6])
x_origin = np.ones((len(true_x)))*46.290000915527344
y_origin = np.ones((len(true_y)))*-34.880001068115234
true_CC = np.array(truth[:,11])
true_isTrack = np.array(truth[:,8])
true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )
true_neutrino = np.array(truth[:,9],dtype=int)
true_zenith = np.array(truth[:,12])
true_coszenith = np.cos(np.array(truth[:,12]))
isNuMu = true_neutrino == 14
isNuE = true_neutrino == 12
if sum(isNuMu) == len(true_neutrino):
    all_NuMu = True
else:
    all_NuMu = False

if cnn_x is not None:
    cnn_r = np.sqrt( (cnn_x - x_origin)**2 + (cnn_y - y_origin)**2 )

# Retro
if reco is not None:
    retro_energy = np.array(reco[:,0])
    retro_zenith = np.array(reco[:,1])
    retro_time = np.array(reco[:,3])
    retro_PID_full = np.array(reco[:,12])
    retro_PID_up = np.array(reco[:,13])
    reco_x = np.array(reco[:,4])
    reco_y = np.array(reco[:,5])
    reco_z = np.array(reco[:,6])
    retro_iterations = np.array(reco[:,14])
    retro_coszenith = np.cos(retro_zenith)
    reco_r = np.sqrt( (reco_x - x_origin)**2 + (reco_y - y_origin)**2 )
    retro_nan = np.isnan(retro_energy)
    #print(sum(retro_nan)/len(retro_nan))

#Additional info
if info is not None:
    prob_nu = info[:,1]
    coin_muon = info[:,0]
    true_ndoms = info[:,2]
    fit_success = info[:,3]
    noise_class = info[:,4]
    nhit_doms = info[:,5]
    n_top15 = info[:,6]
    n_outer = info[:,7]
    prob_nu2 = info[:,8]
    hits8 = info[:,9]
    hlc_x = info[:,10]
    hlc_y = info[:,11]
    hlc_z = info[:,12]

#weights
if weights is not None:
    weights = np.array(weights[:,8])
    #modify by number of files
    mask_numu = np.array(truth[:,9]) == 14
    mask_nue = np.array(truth[:,9]) == 12
    if sum(mask_numu) > 1:
        weights[mask_numu] = weights[mask_numu]/numu_files
    if sum(mask_nue) > 1:
        weights[mask_nue] = weights[mask_nue]/nue_files


check_energy_gt5 = true_energy > 5.
assert sum(check_energy_gt5)>0, "No events > 5 GeV in true energy, is this transformed?"

#Vertex Position
x_origin = np.ones((len(true_x)))*46.290000915527344
y_origin = np.ones((len(true_y)))*-34.880001068115234
true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )
if reco is not None:
    reco_r = np.sqrt( (reco_x - x_origin)**2 + (reco_y - y_origin)**2 )
if info is not None:
    hlc_rho = np.sqrt( (hlc_x - x_origin)**2 + (hlc_y - y_origin)**2 )

#Plot
from PlottingFunctions import plot_distributions
from PlottingFunctions import plot_2D_prediction
from PlottingFunctions import plot_2D_prediction_fraction
from PlottingFunctions import plot_single_resolution
from PlottingFunctions import plot_bin_slices
from PlottingFunctions import plot_rms_slices
from PlottingFunctionsClassification import ROC
from PlottingFunctionsClassification import plot_classification_hist

save=True
if save ==True:
    print("Saving to %s"%save_folder_name)


#True Masks
maskNONE = true_energy > 0.
assert sum(maskNONE)==len(true_energy), "Some true energy < 0? Check!" 
maskCC = true_CC == 1
maskZ = np.logical_and(true_z > -505, true_z < 192)
maskR = true_r < 90.
maskDC = np.logical_and(maskZ,maskR)
maskE = np.logical_and(true_energy > 5., true_energy < 100.)
maskE2 = np.logical_and(true_energy > 1., true_energy < 200.)

#CNN Masks
maskCNNE = np.logical_and(cnn_energy > 5., cnn_energy < 100.)
maskCNNE2 = np.logical_and(cnn_energy > 1., cnn_energy < 100.)
maskCNNZenith = cnn_coszenith <= 0.3
maskCNNVertex = np.logical_and(cnn_r < 150, np.logical_and(cnn_z > -500, cnn_z < -200))

#additional info Masks
if info is not None:
    maskHits8 = hits8 == 1
    maskNu = prob_nu > 0.4
    maskNoise = noise_class > 0.95
    masknhit = nhit_doms > 2.5
    maskntop = n_top15 < 2.5
    masknouter = n_outer < 7.5
    maskHLCZ = np.logical_and(hlc_z > -500., hlc_z < -200.) 
    maskHLCR = hlc_rho < 300.
    maskHLCVertex = np.logical_and(maskHLCZ, maskHLCR)
    maskHits = np.logical_and(np.logical_and(masknhit, maskntop), masknouter)
    maskClass = np.logical_and(maskNu,maskNoise)
    maskMC = np.logical_and(maskHits,maskClass)

#Retro Masks
if reco is not None:
    maskRecoZ = np.logical_and(reco_z > -500., reco_z < -200.)
    maskRecoR = reco_r < 300.
    maskRecoDC = np.logical_and(maskRecoZ, maskRecoR)
    maskRetroZenith = np.cos(retro_zenith) <= 0.3
    maskRetroEnergy = np.logical_and(retro_energy >= 5., retro_energy <= 300.)
    maskRetroTime = retro_time < 14500.
    maskRetroIterations = retro_iterations >= 10
    maskPassRetro = np.logical_and(maskRetroIterations, fit_success)
    maskRetro = np.logical_and(maskRetroIterations, np.logical_and(fit_success, np.logical_and(np.logical_and(maskRetroZenith, maskRetroEnergy), maskRetroTime)))
    maskRetro2 = np.logical_and(maskRetroIterations, np.logical_and(fit_success, np.logical_and(maskRetroEnergy, maskRetroTime)))
    maskANA = np.logical_and(np.logical_and(np.logical_and(maskRecoDC,  maskRetro), maskMC),maskHits8)
    maskANA2 = np.logical_and(np.logical_and(np.logical_and(maskRecoDC,  maskRetro2), maskMC),maskHits8)
    assert sum(maskANA)!=len(maskANA), "No events after ANA mask"
    print(sum(weights[np.logical_and(maskANA,maskCC)]))



cut_list = [np.logical_and(maskHits8,maskCC), np.logical_and(maskANA, maskCC), np.logical_and(maskHLCVertex, np.logical_and(np.logical_and(maskCNNE2, maskCC),maskHits8)), np.logical_and(maskHLCVertex, np.logical_and(np.logical_and(maskCNNE, maskCC),maskHits8)),np.logical_and(maskCNNZenith,np.logical_and(maskHLCVertex, np.logical_and(np.logical_and(maskCNNE, maskCC),maskHits8))),np.logical_and(maskANA2, maskCC), np.logical_and(np.logical_not(maskPassRetro), np.logical_and(maskHits8,maskCC)),np.logical_and( maskE,np.logical_and(maskHits8,maskCC)) , np.logical_and(maskHLCVertex, np.logical_and(np.logical_and(maskCNNE, maskCC),maskHits8)),np.logical_and(maskCNNZenith,np.logical_and(maskCNNVertex, np.logical_and(np.logical_and(maskCNNE, maskCC),maskHits8)))]
cut_names = ["Weighted_Hits8CC", "WeightedAnalysisCuts_CC", "WeightedCNN1100_HLCVertex_Hits8CC", "WeightedCNN5100_HLCVertex_Hits8CC", "WeightedCNN5100_ZenithHLCVertex_Hits8CC","WeightedAnalysisCuts_NoZen_CC","WeightsFailedRetro_Hits8CC", "WeightedTrueE5100_Hits8CC", "WeightedCNN5100_CNNZenithVertex_Hits8CC"]
minvals_energy = [1, 5, 1, 5, 5, 5, 1,5,5]
maxvals_energy = [200, 200, 100, 100, 100., 200, 200,100,100]
minvals_zenith =  [-1., -1, -1, -1, -1, -1, -1,-1,-1]
maxvals_zenith = [1,1,1,1, 0.3, 1, 1,1,0.3]
binss_energy = [199, 195, 199, 95, 95, 195, 199,95,95]
syst_bins_energy = [20, 20, 20, 10, 10, 20, 20, 10,10]
binss_zenith = [100, 100, 100, 100, 100, 100, 100, 100,100]
syst_bins_zenith = [20, 20, 20, 20, 12, 20, 20, 20,12]
sample_names = ["CC", "CC", "CC", "CC", "CC", "CC", "CC", "CC","CC"]
plot_main = False
plot_EMequiv = False
plot_switch = True
plot_others = False
plot_philipp = False
save_base_name = save_folder_name

for cut_index in [8]: #,2,3,4]: #range(1,len(cut_list)):
    cuts = cut_list[cut_index]
    folder_name = cut_names[cut_index]
    true_energy_val = true_energy[cuts]
    true_weights = weights[cuts]
    fit_success_cut = maskPassRetro[cuts]
    if cut_index == 2 or cut_index == 4:
        reco_class = retro_PID_up
    else:
        reco_class = retro_PID_full
    reco_name = "Likelihood-Based"
    sample_name = sample_names[cut_index]

    minval_energy = minvals_energy[cut_index]
    maxval_energy = maxvals_energy[cut_index]
    syst_bin_energy = syst_bins_energy[cut_index]

    print(sum(maskCNNZenith), sum(maskHLCVertex), sum(maskCNNE), sum(maskCC), sum(maskHits8))
    print(sum(np.logical_and(maskCNNZenith,mask_nue)), sum(np.logical_and(mask_nue,maskHLCVertex)), sum(np.logical_and(mask_nue,maskCNNE)), sum(np.logical_and(mask_nue,maskCC)), sum(np.logical_and(mask_nue,maskHits8)), sum(mask_nue) )
    print(sum(cuts))
    
    print("Working on %s"%folder_name)

    save_folder_name = save_base_name + "/%s/"%folder_name
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)

    for variable in ["energy"]: 

        if variable == "energy":
            plot_name = "Energy"
            plot_units = "(GeV)"
            minval = minval_energy
            maxval = maxval_energy
            syst_bin = syst_bin_energy
            bins = binss_energy[cut_index]
            cnn_val = np.copy(cnn_energy[cuts])
            true_val = np.copy(true_energy[cuts])
            retro_val = np.copy(retro_energy[cuts])
        if variable == "zenith":
            plot_name = "Cosine Zenith"
            plot_units = ""
            minval = minvals_zenith[cut_index]
            maxval = maxvals_zenith[cut_index]
            syst_bin = syst_bins_zenith[cut_index]
            bins = binss_zenith[cut_index]
            cnn_val = np.copy(cnn_coszenith[cuts])
            true_val = np.copy(true_coszenith[cuts])
            retro_val = np.copy(retro_coszenith[cuts])


        #Cut out failed Retro or set it to none if all failed retro
        if cut_index == 6:
            retro_val = None
            retro_true_val = None
            retro_weights = None
            tretro_energy_val = None

        else:
            retro_val = retro_val[fit_success_cut]
            retro_true_val = true_val[fit_success_cut]
            retro_weights = true_weights[fit_success_cut]
            tretro_energy_val = true_energy_val[fit_success_cut]
            print(sum(fit_success_cut)/len(fit_success_cut))

        #print(plot_name, plot_units, minval, maxval, syst_bin, bins)
        #print(max(true_val), max(cnn_val), max(retro_val))
        #print(true_val[:10], cnn_val[:10], retro_val[:10])
        

        if plot_main:
            plot_distributions(true_val, cnn_val, 
                                old_reco=retro_val,old_reco_weights=retro_weights,\
                                save=save, savefolder=save_folder_name, weights=true_weights,\
                                reco_name = reco_name, variable=plot_name, units= plot_units,
                                minval=minval,maxval=maxval,bins=bins)
            plot_distributions(true_val, cnn_val,
                                old_reco=retro_val,old_reco_weights=retro_weights,\
                                save=save, savefolder=save_folder_name, weights=true_weights,\
                                reco_name = reco_name, variable=plot_name, units= plot_units,
                                minval=min(true_val), maxval=max(true_val))
        if plot_others:
            plot_distributions(true_r[cuts], reco_r[cuts],\
                                save=save, savefolder=save_folder_name, weights=true_weights,\
                                cnn_name = reco_name,
                                variable="Radial Vertex", units= "(m)",log=True)
            plot_distributions(true_z[cuts], reco_z[cuts],\
                                save=save, savefolder=save_folder_name,
                                weights=true_weights,cnn_name = reco_name,
                                variable="Z Vertex", units= "(m)",log=True)
        if plot_main:
            switch = False 
            plot_2D_prediction(true_val, cnn_val,weights=true_weights,\
                                save=save, savefolder=save_folder_name,
                                bins=bins, switch_axis=switch,
                                variable=plot_name, units=plot_units,
                                reco_name="CNN",flavor=nu_type,sample=sample_name)
            plot_2D_prediction(true_val, cnn_val,weights=true_weights,\
                                save=save, savefolder=save_folder_name,
                                bins=bins,switch_axis=switch,\
                                minval=minval, maxval=maxval, 
                                cut_truth=True, axis_square=True,\
                                variable=plot_name, units=plot_units,\
                                reco_name="CNN",flavor=nu_type,sample=sample_name)
            if retro_val is not None:
                plot_2D_prediction(retro_true_val, retro_val, weights=retro_weights,
                                save=save, savefolder=save_folder_name,
                                bins=bins,switch_axis=switch,\
                                variable=plot_name, units=plot_units,
                                reco_name=reco_name,flavor=nu_type,sample=sample_name)
                plot_2D_prediction(retro_true_val, retro_val, weights=retro_weights,
                                save=save, savefolder=save_folder_name,
                                bins=bins,switch_axis=switch,\
                                minval=minval, maxval=maxval,
                                cut_truth=True, axis_square=True,\
                                variable=plot_name, units=plot_units,
                                reco_name=reco_name,flavor=nu_type,sample=sample_name)
        if plot_EMequiv:
            plot_2D_prediction(em_equiv_energy[cuts], cnn_val,weights=true_weights,\
                                save=save, savefolder=save_folder_name,
                                bins=bins,switch_axis=switch,\
                                minval=minval, maxval=maxval,
                                cut_truth=True, axis_square=True,\
                                variable=plot_name, units=plot_units,
                                reco_name="CNN",variable_type="EM Equiv",
                                flavor=nu_type,sample=sample_name)
            if retro_val is not None:
                plot_2D_prediction(em_equiv_energy[cuts], retro_val, weights=true_weights,
                                save=save, savefolder=save_folder_name,
                                bins=bins,switch_axis=switch,\
                                minval=minval, maxval=maxval,
                                cut_truth=True, axis_square=True,\
                                variable=plot_name, units=plot_units, 
                                reco_name=reco_name, variable_type = "EM Equiv",
                                flavor=nu_type,sample=sample_name)
        if plot_switch:
            switch = True
            plot_2D_prediction(true_val, cnn_val,weights=true_weights,\
                                save=save, savefolder=save_folder_name,
                                bins=bins, switch_axis=switch,
                                variable=plot_name, units=plot_units,
                                reco_name="CNN",flavor=nu_type,sample=sample_name)
            plot_2D_prediction(true_val, cnn_val,weights=true_weights,\
                                save=save, savefolder=save_folder_name,
                                bins=bins,switch_axis=switch,\
                                minval=minval, maxval=maxval,
                                cut_truth=True, axis_square=True,\
                                variable=plot_name, units=plot_units,
                                reco_name="CNN",flavor=nu_type,sample=sample_name)
            if retro_val is not None:
                plot_2D_prediction(retro_true_val, retro_val, weights=retro_weights,
                                save=save, savefolder=save_folder_name,
                                bins=bins,switch_axis=switch,\
                                variable=plot_name, units=plot_units,
                                reco_name=reco_name,flavor=nu_type,sample=sample_name)
                plot_2D_prediction(retro_true_val, retro_val, weights=retro_weights,
                                save=save, savefolder=save_folder_name,
                                bins=bins,switch_axis=switch,\
                                minval=minval, maxval=maxval,
                                cut_truth=True, axis_square=True,\
                                variable=plot_name, units=plot_units,
                                reco_name=reco_name,flavor=nu_type,sample=sample_name)
        if plot_main:   
            if retro_val is not None:
                plot_single_resolution(true_val, cnn_val, weights=true_weights,\
                           use_old_reco = True, old_reco = retro_val,\
                           old_reco_truth=retro_true_val,
                            old_reco_weights = retro_weights,old_reco_name=reco_name,
                           minaxis=-maxval, maxaxis=maxval, bins=bins,\
                           save=save, savefolder=save_folder_name,\
                           variable=plot_name, units=plot_units,
                           flavor=nu_type,sample=sample_name)
                
            else:
                plot_single_resolution(true_val, cnn_val, weights=true_weights,\
                           use_old_reco = False,\
                           minaxis=-maxval, maxaxis=maxval, bins=bins,\
                           save=save, savefolder=save_folder_name,\
                           variable=plot_name, units=plot_units,
                           flavor=nu_type,sample=sample_name)
        

        
        if variable == "energy":
            if plot_others:
                plot_2D_prediction_fraction(true_val, cnn_val,weights=true_weights,\
                                save=save, savefolder=save_folder_name,bins=bins,\
                                xminval=minval, xmaxval=maxval, yminval=-2,ymaxval=2.,\
                                variable=plot_name, units=plot_units, reco_name="CNN")
                plot_2D_prediction_fraction(retro_true_val, retro_val, weights=retro_weights,
                                save=save, savefolder=save_folder_name,bins=bins,\
                                xminval=minval, xmaxval=maxval, yminval=-2,ymaxval=2.,\
                                variable=plot_name, units=plot_units, reco_name=reco_name)
                plot_2D_prediction_fraction(true_val, cnn_val,weights=true_weights,\
                                save=save, savefolder=save_folder_name,bins=bins,\
                                variable=plot_name, units=plot_units, reco_name="CNN")
                plot_2D_prediction_fraction(retro_true_val, retro_val, weights=retro_weights,
                                save=save, savefolder=save_folder_name,bins=bins,\
                                variable=plot_name, units=plot_units, reco_name=reco_name)
            
            if plot_main:           
                if nu_type == "NuMu" or nu_type == "numu":
                    dist_title = r'$\nu_\mu$ '
                elif nu_type == "NuE" or nu_type == "nue":
                    dist_title = r'for $\nu_e$ '
                else:
                    dist_title += nu_type
                dist_title += sample_name
                plot_distributions(true_val,log=True, 
                                    save=save, savefolder=save_folder_name, 
                                    weights=true_weights,reco_name = reco_name,
                                    variable=plot_name, units= plot_units,
                                    bins=bins,
                                    title="Testing Energy Distribution for %s"%dist_title)
                plot_distributions(true_val,log=True,minval=1,maxval=300, 
                                    save=save, savefolder=save_folder_name, 
                                    weights=true_weights,reco_name = reco_name,
                                    variable=plot_name, units= plot_units,
                                    bins=bins,
                                    title="Testing Energy Distribution for %s"%dist_title)

                if retro_val is not None:
                    plot_single_resolution(true_val, cnn_val, weights=true_weights,\
                           use_old_reco = True, old_reco = retro_val,\
                           old_reco_truth=retro_true_val, old_reco_weights=retro_weights,
                           minaxis=-2, maxaxis=2, bins=bins,old_reco_name=reco_name,\
                           save=save, savefolder=save_folder_name,use_fraction=True,\
                           variable=plot_name, units=plot_units,
                           flavor=nu_type,sample=sample_name)
                else:
                    plot_single_resolution(true_val, cnn_val, weights=true_weights,\
                           use_old_reco = False, minaxis=-2, maxaxis=2, bins=bins,\
                           save=save, savefolder=save_folder_name,use_fraction=True,\
                           variable=plot_name, units=plot_units,
                           flavor=nu_type,sample=sample_name)

            
                plot_bin_slices(true_val, cnn_val, weights=true_weights,  
                            old_reco = retro_val,old_reco_truth=retro_true_val,
                            old_reco_weights=retro_weights,use_fraction = True,
                            bins=syst_bin, min_val=minval, max_val=maxval,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units=plot_units, reco_name=reco_name,
                            flavor=nu_type,sample=sample_name)
                plot_rms_slices(true_val, cnn_val, weights=true_weights, 
                            old_reco_truth=retro_true_val, 
                            old_reco = retro_val, old_reco_weights = retro_weights,\
                            use_fraction = True, bins=syst_bin,
                            min_val=minval, max_val=maxval,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units=plot_units, reco_name=reco_name,
                            flavor=nu_type,sample=sample_name)
                plot_bin_slices(true_val, cnn_val, weights=None,  
                            old_reco = retro_val,old_reco_truth=retro_true_val,\
                            use_fraction = True, bins=syst_bin, 
                            min_val=minval, max_val=maxval,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units=plot_units, reco_name=reco_name,
                            flavor=nu_type,sample=sample_name)
                plot_bin_slices(true_val, cnn_val, weights=true_weights,  
                            old_reco = retro_val,old_reco_truth=retro_true_val,
                            old_reco_weights=retro_weights, vs_predict = True,\
                            use_fraction = True, bins=syst_bin,
                            min_val=minval, max_val=maxval,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units=plot_units, reco_name=reco_name,
                            flavor=nu_type,sample=sample_name)

            if plot_philipp:
                plot_bin_slices(true_val, cnn_val, weights=None,  
                            old_reco = retro_val,old_reco_truth=retro_true_val,\
                            use_fraction = True, bins=syst_bin, min_val=minval, max_val=maxval,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units=plot_units, reco_name=reco_name)
                
                ccon = np.logical_and(true_val > minval, true_val < maxval)
                p = dm.PointData()
                p.x = true_val[ccon]
                p.cnn_res = (cnn_val[ccon]-true_val[ccon])/true_val[ccon]
                con = np.logical_and(fit_success_cut, np.logical_and(true_val > minval, true_val < maxval))
                rcon = np.logical_and(retro_true_val > minval, retro_true_val < maxval)
                print(sum(con),sum(rcon))
                r = dm.PointData()
                r.x = retro_true_val[rcon]
                r.cnn_res =  (cnn_val[con]-retro_true_val[rcon])/retro_true_val[rcon]
                r.retro_res =  (retro_val[rcon]-retro_true_val[rcon])/retro_true_val[rcon]
                cnn_labels = ['CNN 68%', 'CNN Median']
                retro_labels = ['Likelihood 68%', 'Likelihood Median']
                #frame = r.binwise(x=np.linspace(minval,maxval,syst_bin)).quantile(q=[0.15865, 0.5, 0.8414])
                plt.figure(figsize=(10,7))
                r.binwise(x=np.linspace(minval,maxval,syst_bin)).quantile(q=[0.15865, 0.5, 0.8414]).plot_bands('retro_res',lines=True, filled=False, linestyles=['--', '-'], lw=2,labels=retro_labels,linecolors=['orange']*2)
                p.binwise(x=np.linspace(minval,maxval,syst_bin)).quantile(q=[0.15865, 0.5, 0.8414]).plot_bands('cnn_res',cmap="Blues", lw=2,labels=cnn_labels)
                plt.hlines(0.0,minval,maxval,color="k",linestyle=":",label="ideal")
                plt.title(r'Energy Resolution Dependence for $\nu_\mu$ CC',fontsize=25)
                plt.xlabel("True Energy (GeV)",fontsize=20)
                plt.ylabel(r'Fractional Resolution: $\frac{reconstruction - truth}{truth}$',fontsize=20)
                plt.legend(fontsize=15)
                plt.savefig("%sSmoothLines_Erange_%i_%i.png"%(save_folder_name,minval,maxval),bbox_inches='tight')

                plt.figure(figsize=(10,7))
                r.binwise(x=np.linspace(minval,maxval,syst_bin)).quantile(q=[0.15865, 0.5, 0.8414]).plot_bands('retro_res',lines=True, filled=True, linestyles=['--', '-'], lw=2,labels=retro_labels,cmap="Oranges",linecolors=['orange']*2,alpha=0.5)
                p.binwise(x=np.linspace(minval,maxval,syst_bin)).quantile(q=[0.15865, 0.5, 0.8414]).plot_bands('cnn_res', lines=True, filled=True,linestyles=['--', '-'],lw=2,cmap="Blues",linecolors=['blue']*2,labels=cnn_labels,alpha=0.5)
                plt.hlines(0.0,minval,maxval,color="k",linestyle=":",label="ideal")
                plt.title(r'Energy Resolution Dependence for $\nu_\mu$ CC',fontsize=25)
                plt.xlabel("True Energy (GeV)",fontsize=20)
                plt.ylabel(r'Fractional Resolution: $\frac{reconstruction - truth}{truth}$',fontsize=20)
                plt.legend(fontsize=15)
                plt.savefig("%sSmoothLines_Erange_%i_%i_v3.png"%(save_folder_name,minval,maxval),bbox_inches='tight')

                """
                cnn_labels = ['CNN 90%', 'CNN 68%', 'CNN Median']
                retro_labels = ['Likelihood 90%','Likelihood 68%', 'Likelihood Median']
                plt.figure(figsize=(10,7))
                r.binwise(x=np.linspace(minval,maxval,syst_bin)).quantile(q=[0.05,0.15865, 0.5, 0.8414,0.95]).plot_bands('retro_res',lines=True, filled=False, linestyles=[':','--', '-'], lw=2,labels=retro_labels,linecolors=['orange']*3)
                p.binwise(x=np.linspace(minval,maxval,syst_bin)).quantile(q=[0.10, 0.15865, 0.5, 0.8414, 0.9]).plot_bands('cnn_res',cmap="Blues", lw=2,labels=cnn_labels)
                plt.hlines(0.0,minval,maxval,color="k",linestyle=":",label="ideal")
                plt.title(r'Energy Resolution Dependence for $\nu_\mu$ CC',fontsize=25)
                plt.xlabel("True Energy (GeV)",fontsize=20)
                plt.ylabel(r'Fractional Resolution: $\frac{reconstruction - truth}{truth}$',fontsize=20)
                plt.legend(fontsize=15)
                plt.savefig("%sSmoothLines_Erange_%i_%i_v2.png"%(save_folder_name,minval,maxval),bbox_inches='tight')
                """
                plt.figure()
                #r.binwise(x=syst_bin).quantile(q=[0.15865, 0.5, 0.8414]).plot_bands('res')
                r.binwise(x=syst_bin).quantile(q=[0.15865, 0.5, 0.8414]).plot_bands('cnn_res',filled=False)
                plt.savefig("%sStepLines_Erange_%i_%i.png"%(save_folder_name,minval,maxval),bbox_inches='tight')
            
            
            

        
        if variable == "zenith":
            plot_single_resolution(true_val, cnn_val, weights=true_weights,\
                           use_old_reco = True, old_reco = retro_val,\
                           old_reco_weights = retro_weights,
                           minaxis=-2, maxaxis=2, bins=bins,\
                           save=save, savefolder=save_folder_name,\
                           variable=plot_name, units=plot_units, reco_name=reco_name)
            plot_bin_slices(true_val, cnn_val, weights=true_weights,  
                            old_reco = retro_val,old_reco_weights=retro_weights,\
                            use_fraction = False, bins=syst_bin, min_val=minval, max_val=maxval,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units=plot_units, reco_name=reco_name)
            plot_distributions(true_val,
                                    save=save, savefolder=save_folder_name, weights=true_weights,\
                                    reco_name = reco_name, variable=plot_name, units= plot_units,
                                    bins=bins)
            plot_distributions(true_val,minval=-1,maxval=1., 
                                    save=save, savefolder=save_folder_name, weights=true_weights,\
                                    reco_name = reco_name, variable=plot_name, units= plot_units,
                                    bins=bins)
            """
            plot_bin_slices(true_val, cnn_val, weights=true_weights,  
                            old_reco = retro_val,vs_predict = True,\
                            use_fraction = False, bins=syst_bin, min_val=minval, max_val=maxval,\
                            save=save, savefolder=save_folder_name,\
                            variable=plot_name, units=plot_units, reco_name=reco_name)
            """
            plot_bin_slices(true_val, cnn_val, weights=true_weights, \
                            old_reco = retro_val,energy_truth= true_energy_val,\
                            reco_energy_truth=tretro_energy_val,
                            old_reco_weights=retro_weights,
                            use_fraction = False, bins=syst_bin_energy, \
                            min_val=minval_energy, max_val=maxval_energy,\
                            save=save, savefolder=save_folder_name,\
                            variable="True Neutrino Energy", units=plot_units, reco_name=reco_name)
            plot_rms_slices(true_val, cnn_val, weights=true_weights,  
                            old_reco = retro_val, energy_truth= true_energy_val,\
                            reco_energy_truth=tretro_energy_val,
                            old_reco_weights=retro_weights,
                            use_fraction = False, bins=syst_bin_energy,
                            min_val=minval_energy, max_val=maxval_energy,\
                            save=save, savefolder=save_folder_name,\
                            variable="True Neutrino Energy", units=plot_units, reco_name=reco_name)


    if all_NuMu and (cut_index > 7):
        ROC(true_isTrack,cnn_class,mask=cuts,mask_name="",reco=reco_class,save=save,save_folder_name=save_folder_name)
        plot_classification_hist(true_isTrack,cnn_class,reco=reco_class,mask=cuts,mask_name="", variable="Classification",units="",bins=50,log=False,save=save,save_folder_name=save_folder_name)
