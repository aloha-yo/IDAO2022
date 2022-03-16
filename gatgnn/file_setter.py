import pandas as pd
import numpy as np
from shutil import copyfile

def use_property(property_name,source, do_prediction = False):

    print('> Preparing dataset to use for Property Prediction. Please wait ...')

    if property_name   in ['band','bandgap','band-gap']:                      filename = 'bandgap.csv'         ;p=1;num_T  = 36720
    elif property_name in ['bulk','bulkmodulus','bulk-modulus','bulk-moduli']:filename = 'bulkmodulus.csv'     ;p=3;num_T  = 4664
    elif property_name in ['energy-1','formationenergy','formation-energy']:  filename = 'formationenergy.csv' ;p=2;num_T  = 60000
    elif property_name in ['energy-2','fermienergy','fermi-energy']:          filename = 'fermienergy.csv'     ;p=2;num_T  = 60000
    elif property_name in ['energy-3','absoluteenergy','absolute-energy']:    filename = 'absoluteenergy.csv'  ;p=2;num_T  = 60000               
    elif property_name in ['shear','shearmodulus','shear-modulus','shear-moduli']:filename = 'shearmodulus.csv';p=4;num_T  = 4664
    elif property_name in ['poisson','poissonratio','poisson-ratio']:         filename = 'poissonratio.csv'    ;p=4;num_T  = 4664
    elif property_name in ['is_metal','is_not_metal']:                        filename = 'ismetal.csv'         ;p=2;num_T  = 55391
    elif property_name == 'new-property'             :                        filename = 'newproperty.csv'     ;p=None;num_T  = None

    df     = pd.read_csv(f'DATA/properties-reference/{filename}',names=['material_id','value']).replace(to_replace='None',value=np.nan).dropna()


    if source == 'NEW':
        cif_dir  = 'CIF-DATA_NEW'
        CIF_dict = {'radius':8,'step':0.2,'max_num_nbr':12}
        d_src    = 'DATA'
        src, dst = d_src+'/CIF-DATA/atom_init.json',d_src+'/CIF-DATA_NEW/atom_init.json'
        copyfile(src, dst)


    # ADDITIONAL CLEANING
    if p in [3,4]:
        df        = df[df.value>0]


    df.to_csv(f'DATA/{cif_dir}/id_prop.csv',index=False,header=False)
    if not do_prediction:    print(f'> Dataset for {source}---{property_name} ready !\n\n')
    return source,num_T,CIF_dict
