#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 17:58:18 2021

@author: jonathandurand
"""

import xarray as xr
from contrack import contrack
import netCDF4
from netCDF4 import Dataset
import matplotlib.pylab as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import cartopy
from math import sin, cos, sqrt, atan2, radians
from mpl_toolkits.basemap import Basemap
import netCDF4
from netCDF4 import Dataset
import matplotlib.pylab as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
#from carto import scale_bar
import xarray as xr
import numpy.ma as ma
import seaborn as sns
import pandas as pd
import csv
import matplotlib.ticker as ticker
import os
from datetime import datetime, timedelta

path="/Users/jonathandurand/Documents/UQAM/blocage/blocagemix"
os.chdir(path)


def mean_coord(londeg,latdeg):
    
    lon = [i *(np.pi/180) for i in londeg]
    lat = [i *(np.pi/180) for i in latdeg]
    
    X=[]
    Y=[]
    Z=[]
    for i in range(len(lat)):
    #Convert lat/lon (must be in radians) to Cartesian coordinates for each location.
        X.append(cos(lat[i]) * cos(lon[i]))
        Y.append(cos(lat[i]) * sin(lon[i]))
        Z.append(sin(lat[i]))
    
    # #Compute average x, y and z coordinates.
    x=np.sum(X)/len(X)
    y=np.sum(Y)/len(Y)
    z=np.sum(Z)/len(Z)
    
    # #Convert average x, y, z coordinate to latitude and longitude.
    Lon = atan2(y, x)
    hyp = sqrt(x * x + y * y)
    Lat = atan2(z, hyp)
    
    return Lon,Lat
      
def distance(longitude1,longitude2,latitude1,latitude2):
    # approximate radius of earth in km
    R = 6373.0
    lat1 = radians(longitude1)
    lon1 = radians(longitude2)
    lat2 = radians(latitude1)
    lon2 = radians(latitude2)   
    dlon = lon2 - lon1
    dlat = lat2 - lat1    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))    
    distance = R * c    
    print("Result:", distance)
    print("Should be:", 278.546, "km")

def get_unique_numbers(numbers):
    list_of_unique_numbers = []
    unique_numbers = set(numbers)
    for number in unique_numbers:
        list_of_unique_numbers.append(number)
    return list_of_unique_numbers

yi=1950
yf=2020

##########################################
####LOAD DATA FILES
##########################################

za=xr.open_mfdataset("./data/LR/era5_z500_NH_dailymean_"+str(year)+".nc" for year in range(yi,yf))
if za.geopotential.ndim==4:
    ds = za.reduce(np.nansum, dim='expver')
    ds.geopotential.attrs['units']='m**2 s**-2'
else:
    ds=xr.open_mfdataset("./data/LR/era5_z500_NH_dailymean_"+str(year)+".nc" for year in range(yi,yf))
    
### Z500 height climatology loading ###
clim=xr.open_mfdataset("./data/LR_z_clim.nc")
z_clim=clim.z_clim
datime=ds.time.values
### Convert to height geopotential ###    
g = 9.80665
zgp=ds.geopotential/g
z_clim=z_clim/g
###Dataarray to array to iterate on element ###
zgp_data_ori=np.array(zgp.data)
zgp_data_ori=np.flip(zgp_data_ori,axis=1)
lat=ds.latitude.values
lat=np.flipud(lat)

### SHIFT ALL THE DATA TO +260 TO CENTER CANADA ### REMOVED FOR A WHILE, WE KEEP THE ORIGINAL SHAPE AND WILL WORK WITH LONGITUDE INSTEAD OF GRID POINT
long_ori=np.array(ds.longitude)
long_ori=long_ori.tolist()
zgp_data=zgp_data_ori
longitude=np.array(long_ori)
#idx = (long_ori.index(280))
# #zgp_data=np.array(zgp_data_ori)
# #longitude=np.array(long_ori)
#zgp_data = np.roll(zgp_data_ori, idx, axis=2)
#longitude=np.roll(long_ori, idx, axis=0)

##########################################
####INITIALISATION OF TEMP VARIABLES
##########################################

dx=long_ori[1]-long_ori[0]
### Lamdba values for latitude : ###
delta = [-5, -2.5, 0, 2.5, 5] 
phin=[]
phio=[]
phis=[]
idxn=[]
idxo=[]
idxs=[]
GHGN=np.zeros((len(zgp_data),len(zgp_data[0]),len(zgp_data[0][0])),dtype=float)
GHGS=np.zeros((len(zgp_data),len(zgp_data[0]),len(zgp_data[0][0])),dtype=float)
ind_block_lon=np.zeros((len(zgp_data),len(zgp_data[0]),len(zgp_data[0][0])),dtype=float)
ind_block_lat=np.zeros((len(zgp_data),len(zgp_data[0]),len(zgp_data[0][0])),dtype=float)
lon_blo=np.zeros((len(zgp_data),len(zgp_data[0][0])),dtype=float)
lon_blo_filtered1=np.zeros((len(zgp_data),len(zgp_data[0][0])),dtype=float)
lon_blo_ind=np.zeros((len(zgp_data),len(zgp_data[0][0])),dtype=float)
lon_blo_ind2=np.zeros((len(zgp_data),len(zgp_data[0][0])),dtype=float)
lon_blo_filtered3=np.zeros((len(zgp_data),len(zgp_data[0][0])),dtype=float)

for lamb in delta:     
        phi_N = 77.5 + lamb
        phi_O = 60.0 + lamb
        phi_S = 40.0 + lamb      
        idx_n = np.argmin(np.abs(lat - phi_N))
        idx_o = np.argmin(np.abs(lat - phi_O))
        idx_s = np.argmin(np.abs(lat - phi_S))
        phin.append(phi_N)
        phio.append(phi_O)
        phis.append(phi_S)
        idxn.append(idx_n)
        idxo.append(idx_o)
        idxs.append(idx_s)
        
##########################################
####COMPUTE FIRST PART GHGS + GHGS AND FILTER 1 TEMPORAL
##########################################
print("Compute GPM gradients")
### COMPUTE GHGN AND GHGS ##
for nt in range(0, len(zgp_data)):
#      print("progress nt ", nt, len(zgp_data))
      for i in range(len(phin)):     
          for ni in range(0, len(zgp_data[0][0])):         
              GHGN[nt,idxn[i],ni]=((zgp_data[nt,idxn[i],ni])-(zgp_data[nt,idxo[i],ni]))/(phin[i]-phio[i])
              GHGS[nt,idxs[i],ni]=((zgp_data[nt,idxo[i],ni])-(zgp_data[nt,idxs[i],ni]))/(phio[i]-phis[i])

print("Find blocks from blocked longitudes")

### CREATE NEW VARIABLE WITH 1 FOR LONGITUDE BLOCKED AND NaN FOR NOT BLOCKED
z_clim=np.array(z_clim)
for nt in range(0, len(zgp_data)):
#      print("progress nt ", nt, len(zgp_data))
      indice=0
      for i in range(len(phin)): 
          indice=indice+1
          for ni in range(0, len(zgp_data[0][0])):   
              if GHGN[nt,idxn[i],ni] < -10 and GHGS[nt,idxs[i],ni] > 0 and zgp_data[nt,idxo[i],ni]-z_clim[idxo[i],ni] >0 :
                  lon_blo[nt,ni]=1
              else:
                  if i>0 and lon_blo[nt,ni]==1:
                      lon_blo[nt,ni]=1
                  else:
                      lon_blo[nt,ni]=np.nan              


print("FILTER 1 : Spatial extension criteria")

### FILTER 1 : EXTENSION CRITERIA. ALLOWING ONE NON BLOCKED BETWEEN TWO BLOCKED LONGITUDE/ MINIMUM 12.5° = 5 LONGITUDE
for nt in range(0, len(zgp_data)):
#for nt in range(54, 55):
    #print("nt", nt)
    v=0
    ni=-1
    out=0
    keep_running=True
    while keep_running ==True:
        ni=ni+1

        if ni<len(zgp_data[0][0]):
            if lon_blo[nt,ni]==1:
                v=v+1
                out=0
            ##cas ou on est entre 0 et 360 et verification +-1 à 1. Dans ce cas meme si ni=0 > blocage
            else :
                if ni>0 and ni+1 < len(zgp_data[0][0]) and lon_blo[nt,ni-1]==1 and lon_blo[nt,ni+1]==1 : 
                    v=v+1
                    out=0
                else:
                    out=1

        ##cas ou on est a +360 -1
        if ni==len(zgp_data[0][0])-1 :
            if lon_blo[nt,ni]==1:
                v=v+1
                out=0            
            else:
                if lon_blo[nt,ni-1]==1 and lon_blo[nt,ni-len(zgp_data[0][0])+1]==1 :
                    out=0
                    v=v+1
                else : 
                    out=1
                    
        ##cas ou on est a pile +360
        if ni==len(zgp_data[0][0]) :
            if lon_blo[nt,ni-len(zgp_data[0][0])]==1:
                v=v+1
                out=0            
            else:
                if lon_blo[nt,ni-1]==1 and lon_blo[nt,ni-len(zgp_data[0][0])+1]==1 :
                    out=0
                    v=v+1
                else : 
                    out=1
                                              
        ##cas ou on est >360
        if ni>len(zgp_data[0][0]) :
            if lon_blo[nt,ni-len(zgp_data[0][0])]==1:
                out=0
                v=v+1              
            else:
                if lon_blo[nt,ni-len(zgp_data[0][0])-1]==1 and lon_blo[nt,ni-len(zgp_data[0][0])+1]==1 :
                    out=0
                    v=v+1
                else:
                    out=1
                              
        ##longitude blockée sarrete, on regarde si >5
        if out==1:
            #print("out",v , ni)
            if ni<len(zgp_data[0][0]):
                if v >= 5 :
                    #print("v", v, ni)
                    lon_blo_filtered1[nt,ni-v:ni]=1
                    v=0
                else:
                    lon_blo_filtered1[nt,ni-v:ni]=np.nan
                    v=0
            if ni==len(zgp_data[0][0])-1:
                if v >= 5 :
                    lon_blo_filtered1[nt,ni-v:ni]=1
                    v=0
                else:
                    #print("hello")
                    lon_blo_filtered1[nt,ni-v:ni]=np.nan
                    v=0
                    keep_running=False
            if ni>=len(zgp_data[0][0]):
                #print("on est au apres" ,ni ,v )
                ##ici on est forcement a +360°
                if ni-len(zgp_data[0][0])>v and v<5:
                    #print("1", ni ,v)
                    lon_blo_filtered1[nt,ni-len(zgp_data[0][0]):ni-len(zgp_data[0][0])+v]=np.nan
                    keep_running=False                      
                if v>ni-len(zgp_data[0][0]) and v>=5:
                    #print("2", ni ,v)
                    lon_blo_filtered1[nt,ni-v+1:len(zgp_data[0][0])]=1                        
                    lon_blo_filtered1[nt,0:ni-len(zgp_data[0][0])]=1
                    keep_running=False                        
                if v>ni-len(zgp_data[0][0]) and v<5:
                    #print("3", ni ,v)
                    lon_blo_filtered1[nt,ni-v:len(zgp_data[0][0])]=np.nan                      
                    lon_blo_filtered1[nt,0:ni-len(zgp_data[0][0])]=np.nan
                    keep_running=False                   

### 0 to Nan for filtered array ###
lon_blo_filtered1[lon_blo_filtered1==0]=np.nan

######################################################
#CHECKPOINT GRAPH BLOCKED LONGITUDE 
######################################################

##CREATE DATASET TO STORE LON_BLO WITH TIME                 
da = xr.DataArray(
    data=lon_blo_filtered1,
    dims=["time", "lon"],
    coords=dict(
    time=(["time"], datime),
    lon=(["lon"], long_ori),
    ),
    attrs=dict(
    description="Longitude blocked",
    units="none",
    ),)           
###PLOT LONGITUDE DISTRIBUTION ###
##annual mean of blocked longitude
db=da.groupby('time.year').sum('time')
a=db.values
lon=db.lon
df = pd.DataFrame(a).melt()
ax=sns.lineplot(x=df.variable*2.5, y="value", data=df,ci="sd")# Add std deviation bars to the previous plot
ax.set(xlabel='Longitude', ylabel='Annual frequency of blocked longitudes')
ax.set(xlim=(0,360))
plt.xticks(range(0, 390, 30))
plt.title('Annual frequency of blocked longitudes - 1950-2020')
plt.savefig("output.png",dpi=200)
# mean = df.mean(axis=1)
# std  = df.std(axis=1)
# ax.errorbar(df.index, mean, yerr=std, fmt='-o')
# b=np.nanmean(a, axis=0)
# c=np.nansum(lon_blo, axis=0)
# c_fil=np.nansum(lon_blo_filtered, axis=0)
# plt.plot(long_ori, b)
#plt.plot(long_ori, a_fil)

##########################
### COMPUTE BLOCKING CENTER WITH 5° EXTENSION EAST/WEST ###
#########################

## 2 = 5° ##
ext=2.
## threshold for min max lat box BARRIOPEDRO 2006 ##
minlat=idxs[-1]
maxlat=idxn[0]
### Indice = blocage number
indice=0
loop=0
log_indice=[]
log_date=[]
log_lon=[]
log_lat=[]
log_s=[]
log_max_val=[]
zlist1=[]
zlist=[]
total_blocks=0

print("Attribute a number to each block")

### ATTRIBUTE A NUMBER TO EACH BLOCK###
for t in range(len(zgp_data)):
#for t in range(0,20,1):
    v=0
    vm=0
    block_ind=0    
    for ni in range(len(lon_blo_filtered1[0])):
        ind=0
        i=0
        #check si il y a un block à 357.2°
        if ni==0 and lon_blo_filtered1[t,ni] ==1:
            while True:
                i=i+1
                if lon_blo_filtered1[t,len(lon_blo_filtered1[0])-i] ==1:
                    vm=vm+1
                else:
                    break
            total_blocks=total_blocks+1
            lon_blo_ind[t,len(lon_blo_filtered1[0])-vm:len(lon_blo_filtered1[0])]=total_blocks            
            vm=0  
            i=0
            
            while True:
                if lon_blo_filtered1[t,ni+i] ==1:
                    vm=vm+1
                else:
                    break
                i=i+1        
            lon_blo_ind[t,ni:ni+vm]=total_blocks            
            vm=0     


        #check in the normal range, and if lon_blo_ind already attributed > skip    
        if ni>0 and lon_blo_filtered1[t,ni] ==1 and lon_blo_ind[t,ni] ==0 :
            v=v+1
        else:
            if v !=0:
                total_blocks=total_blocks+1
                lon_blo_ind[t,ni-v:ni]=total_blocks
            v=0

print("Compute center for each blocked longitudes")

##CENTER BLOCKED LONGITUDES DETECTION
###threhold for min max lon box is 5° East/West from the blocked longitude ##
###threshold for min max lat box is max phiS and min phiN
#the longitude center is chosen as that longitude within the box with maximum height latitudinally averaged for box limits.
#Once the blocking longitude is detected, the latitudinal center is that for the selected longitude center displaying
#the highest longitudinal averaged height value within the box

#latbox:
lat_box=np.arange(minlat,maxlat+1,1)
#lonbox:
for nt in range(0, len(zgp_data)):
#for nt in range(0,1,1):
    v=0
    #for ni in range(0, len(zgp_data[0][0])):
    ##get all unique block value for each day
    unique_block_value=np.unique(lon_blo_ind[nt,:])[1:]
    
    for i in range(len(unique_block_value)):
        block_val=int(unique_block_value[i])
        ##take all index for block value
        block_indices,=np.where(lon_blo_ind[nt,:]==block_val)
        ##+-2 longitude 
        lon_box=[]
        lon_pos=0
        lon_pos_ref=0
        lon_ind=0
        lat_pos_ref=0
        size=0
        maxanti=0
        ##cas ou 143 et 0
        if 0 in block_indices and len(zgp_data[0][0])-1 in block_indices:
            
            east=np.argmax(block_indices>len(zgp_data[0][0])/2)
            east_val=block_indices[east]
            east_val=(east_val-2,east_val-1)
            
            west=np.argmin(block_indices<len(zgp_data[0][0])/2)
            west_val=block_indices[west-1]
            west_val=(west_val+1,west_val+2)

            lastpart=np.arange(east_val[-1],len(zgp_data[0][0]),1)
            firstpart=np.arange(0,west_val[0],1)
            lon_boxw=np.append(east_val,lastpart)
            lon_boxe=np.append(firstpart,west_val)
            lon_box=np.append(lon_boxw,lon_boxe)
            
        #cas ou 0 mais pas 143
        if 0 in block_indices and len(zgp_data[0][0])-1 not in block_indices:
            temp=(len(zgp_data[0][0])-2, len(zgp_data[0][0])-1)
            lon_box=np.append(temp,block_indices)
            
        #cas ou 143 mais pas 0    
        if len(zgp_data[0][0])-1  in block_indices and 0 not in block_indices:
            temp=(0, 1)
            lon_box=np.append(block_indices,temp)

        #cas ou on est dans le milieu de l'array 
        if 0 not in block_indices and len(zgp_data[0][0])-1 not in block_indices:
            temp1=(block_indices[0]-2,block_indices[-1]+2)
            lon_box=np.arange(temp1[0],temp1[-1]+1)
            
        #cas ou 0 absent et 1 present
        if 0 not in block_indices and 1 in block_indices:
            temp=(len(zgp_data[0][0])-1)
            temp2=np.arange(0,block_indices[-1]+3)
            lon_box=np.append(temp,temp2)
            
        #cas 142 present mais pas 143             
        if len(zgp_data[0][0])-1 not in block_indices and len(zgp_data[0][0])-2 in block_indices:
            temp=(len(zgp_data[0][0])-1,0)
            temp2=np.arange(block_indices[0]-2,block_indices[-1]+1)
            lon_box=np.append(temp2,temp)       
#        print("lonbox", nt, lon_box)
        
        ##compute size of longitude blocked - 4 artificial blocked 
        ###on compute la moyenne latitudinale de la gpm , on regarde quelle longitude a le gpm le plus haut
#        if nt==12 or nt==13 or nt==13 or nt==14 or nt==15 or nt==16:
#            print("lon",nt, lon_box)
        size=len(lon_box)-4
        for i in range(len(lon_box)):
            lon_pos=np.mean(zgp_data[nt,minlat:maxlat+1,lon_box[i]])
            maxgpm=np.amax(zgp_data[nt,minlat:maxlat+1,lon_box[i]])
            #if i==0:
                #print("test", zgp_data[nt,minlat:maxlat+1,lon_box[i]])
            if lon_pos > lon_pos_ref:
                lon_pos_ref=lon_pos
                lon_ind=lon_box[i]
            if maxgpm>maxanti:
                maxanti=maxgpm
                log_max_val.append(maxanti) 
        for i in range(len(lat_box)):
            lat_pos=np.mean(zgp_data[nt,lat_box[i],lon_box])     
            #if i==0:
            #    print("test2", zgp_data[nt,lat_box[i],lon_box])
            if lat_pos > lat_pos_ref:
                lat_pos_ref=lat_pos
                lat_ind=lat_box[i]
        #if nt==12 or nt==13 or nt==13 or nt==14 or nt==15 or nt==16:
        if nt==5482 :
            print("lonind",nt, lon_ind, longitude[lon_ind], block_val)                
        log_indice.append(block_val)
        log_s.append(size)
        log_date.append(datime[nt])
        log_lon.append(longitude[lon_ind])
        log_lat.append(lat[lat_ind])        
                
##SAVE all daily blocking event - without temporal 5 days criteria
df_lon_dai = pd.DataFrame (log_lon, columns = ['longitude'])
df_lon_dai.to_csv('./logs/lon_daily_event_no_temp_criteria.csv', sep=',')
df_lat_dai = pd.DataFrame (log_lat, columns = ['latitude'])
df_lat_dai.to_csv('./logs/lat_daily_event_no_temp_criteria.csv', sep=',')
df_date_dai = pd.DataFrame (log_date, columns = ['date'])
df_date_dai.to_csv('./logs/date_daily_event_no_temp_criteria.csv', sep=',')                
df_indice_dai = pd.DataFrame (log_indice, columns = ['indice'])
df_indice_dai.to_csv('./logs/indice_daily_event_no_temp_criteria.csv', sep=',')         
df_size_dai = pd.DataFrame (log_s, columns = ['size'])
df_size_dai.to_csv('./logs/size_daily_event_no_temp_criteria.csv', sep=',')                       
                
 
print("FILTER 2: Blocking under 45deg are unique")

## FILTER 2 : SPATIAL CRITERIA. Blockings within 45° is unique, keeping only the one with highest height. 
log_indice_spa=log_indice.copy()
log_date_spa=log_date.copy()
log_lon_spa=log_lon.copy()
log_lat_spa=log_lat.copy()
log_s_spa=log_s.copy()
log_max_val_spa=log_max_val.copy()
lon_blo_filtered2=lon_blo_ind.copy()
indexx=[]
indd=0
for t in range(len(log_date)-1):
    v=0
    end=0
    west1=0
    west2=0
    east1=0
    east2=0
    westbl=0
    if str(log_date[t]) == str(log_date[t+1]) :
        bl_lon1=radians(log_lon[t])
        bl_lon2=radians(log_lon[t+1])
        dlon = abs(bl_lon2 - bl_lon1)
        degrad=dlon*57.2958
        if degrad <= 45.0 :
            for i in range(0,len(long_ori)):
                ##on index le t avec le tt de l'ensemble des dates
                compadate=np.where(log_date[t] == datime)
                indate=compadate[0]
                tt=indate[0]
                if lon_blo_ind[tt,i] >=1 :#and end==0:
                    v=v+1
                else:
                    if lon_blo_ind[tt,i] !=1 and v>0:
                        if westbl == 0 :
                            z=v
                            west1=i-z
                            east1=i
                            v=0
                            westbl=westbl+1
                        else:
                            z=v
                            west2=i-z
                            east2=i
                            v=0
                            end=1                       
                if (west2-east1) <= (22.5/dx) and end==1 :
                    ### Take blocking center height ###                    
                    index_lon1 = longitude.tolist().index(log_lon[t])
                    index_lat1 = lat.tolist().index(log_lat[t])
                    index_lon2 = longitude.tolist().index(log_lon[t+1])
                    index_lat2 = lat.tolist().index(log_lat[t+1])                    
                    block1=zgp_data[tt,index_lat1,index_lon1]
                    block2=zgp_data[tt,index_lat2,index_lon2]
                    
                    if block1 >= block2 :
                        lon_blo_filtered2[tt,west2:east2+1] = 0                       
                        indexx.append(t+1)
                    else :
                        lon_blo_filtered2[tt,west1:east1+1] = 0
                        indexx.append(t)
 
### Remove fake blocking <45°
numbers=indexx       
listunique=get_unique_numbers(numbers)      
listunique.sort()   
for index in sorted(listunique, reverse=True):
    del log_indice_spa[index]                    
    del log_date_spa[index]                    
    del log_lon_spa[index]                    
    del log_lat_spa[index]                    
    del log_s_spa[index]                    
    del log_s_spa[index]                    
    del log_max_val_spa[index]
     
####################################             
### BLOCKING INTENSITIES COMPUTATION   
####################################
print("Compute blocking intensities")

val_deg=20
log_intensity_spa=[]
for i in range(len(log_date_spa)):
    index_low=[]
    index_high=[]
    block_max=log_max_val_spa[i]
    block_lon=log_lon_spa[i]
    block_lat=log_lat_spa[i]
    ### LINK DATE LOGFILE TO GLOBAL FILE
    compadate=np.where(log_date[i] == datime)
    indate=compadate[0]
    ii=indate[0]
    ### TAKE 
    mindeg=block_lon-val_deg
    maxdeg=block_lon+val_deg
    
    ### Find longitudes area even if below/above 0/360 deg
    if mindeg <0:
        low=np.arange(360+(mindeg),360,dx).tolist()
        low2=np.arange(0,block_lon+dx,dx).tolist()
        low.extend(low2)
    if maxdeg >= 360:
        high2=np.arange(0,(maxdeg+dx)-360,dx).tolist()
        high=np.arange(block_lon,360,dx).tolist()
        high.extend(high2)
    if maxdeg < 360:
        high=np.arange(block_lon,maxdeg+dx,dx).tolist()
    if mindeg >= 0:
        low=np.arange(mindeg,block_lon+dx,dx).tolist()
    
       
    ### FIND INDICES FROM BLOCKING CENTER ###
    block_lat_index = lat.tolist().index(block_lat)
    block_lon_index = longitude.tolist().index(block_lon)

    ### FIND indices from longitude blocked area
    for i in range(len(low)):
        index_low1 = longitude.tolist().index(low[i])
        index_low.append(index_low1)
    for i in range(len(high)):
        index_high1 = longitude.tolist().index(high[i])
        index_high.append(index_high1)
 
    ### COMPUTE MIN UP AND DOWNSTREAM  
    blockdown=min(zgp_data[ii,block_lat_index,index_low])
    blockup=min(zgp_data[ii,block_lat_index,index_high])
    #blockcent=zgp_data[ii,block_lat_index,block_lon_index]
    blockcentmax=block_max
    ###COMPUTE RC THEN BI###
    z1=(blockdown+blockcentmax)/2.
    z2=(blockup+blockcentmax)/2.
    RC=(z1+z2)/2.
    #RC=(blockdown+blockup)/2 ## second method
    BI=100*((blockcentmax/RC)-1.)
    if BI < 0:
        BI=0
    log_intensity_spa.append(BI)

#######################
### TEMPORAL FILTER ###
#######################
lon_blo_filtered2[np.isnan(lon_blo_filtered2)]=0
total_blocks=0
lon_blo_filtered3[lon_blo_filtered3==0]=np.nan
print("Temporal filter : 5 days at least. Longest part to compute, be patient")

### ATTRIBUTE A NUMBER TO EACH BLOCK PART 2 ###

for t in range(len(zgp_data)):
#for t in range(0,20,1):
    v=0
    vm=0
    block_ind=0    
    for ni in range(len(lon_blo_filtered2[0])):
        ind=0
        i=0
        #check si il y a un block à 357.2°
        if ni==0 and lon_blo_filtered2[t,ni] >=1:
            while True:
                i=i+1
                if lon_blo_filtered2[t,len(lon_blo_filtered2[0])-i] >=1:
                    vm=vm+1
                else:
                    break
            total_blocks=total_blocks+1
            lon_blo_ind2[t,len(lon_blo_filtered2[0])-vm:len(lon_blo_filtered2[0])]=total_blocks            
            vm=0  
            i=0
            
            while True:
                if lon_blo_filtered2[t,ni+i] >=1:
                    vm=vm+1
                else:
                    break
                i=i+1        
            lon_blo_ind2[t,ni:ni+vm]=total_blocks            
            vm=0     

        #check in the normal range, and if lon_blo_ind already attributed > skip    
        if ni>0 and lon_blo_filtered2[t,ni] >=1 and lon_blo_ind2[t,ni] ==0 :
            v=v+1
        else:
            if v !=0:
                total_blocks=total_blocks+1
                lon_blo_ind2[t,ni-v:ni]=total_blocks
            v=0

#### CHECK OVER UNTIL HERE


##on copy lon_blo_ind2 pour la suite des calculs > filter 3
lon_blo_filtered3=lon_blo_ind2.copy()
lon_blo_filtered3[lon_blo_filtered3==0]=np.nan
###Huge loop to process with all the temporal conditions (min d+5, one day can be missing...) barriopedro 2006
block_checked=[] 
block_true=[]     
final_blo=[] 
final_duration=[]
index_right=[]
index_left=[]
p=0
for t in range(len(zgp_data)):
#for t in range(479,485+1,1):
      stop=0
      nb_day=1
      for ni in range(len(lon_blo_filtered3[0])):
        if np.isnan(lon_blo_filtered3[t,ni]) == False :
            #print("on est la 1")
            ###On recupere la valeur du bloc###
            block_true=[]
            block_val=int(lon_blo_filtered3[t,ni])
            #if block_val ==312:
            #    print("block check 312 before", block_checked )
            #if block_val ==310:
            #    print("block check 310 before", block_checked )
            #print("on est la 2", block_val)
            if block_val not in block_checked or len(block_checked)==0:              
                idx=0
                #test
                #block_val=233
                ###on ajoute cette valeur à la liste des block utilisé
                block_checked.append(block_val)
                block_checked=list(set(block_checked))
                ###On recupere list valeur des longitudes blocké pour block_val
                idx = np.where( lon_blo_filtered3 == block_val)
                idx=idx[1].tolist()
                keep_running=True
                #if block_val ==312:
                #    print("block check 312", block_checked )
                #if block_val ==310:
                #    print("block check 310", block_checked )
                while keep_running ==True:
                    out=0
                    indloop=0
                    for dt in range(1,50,1):
                        time=t+dt
                        ### On verifie si a t+1 on a au moins une longitude en commun
                        if time < len(zgp_data) :
                            if np.nansum(lon_blo_filtered3[time,idx]) >= 1:
                                nb_day=nb_day+1
                                #print("check",lon_blo_filtered3[time,idx])
                                if out==0:
                                    ###on ajoute cette valeur à la liste des blocages identifiés
                                    block_true.append(int(block_val))
                                
                                #print("blockval", block_val)
                                #long expression to keep only block candidate values at d+1
                                
                                block_candidate=np.unique(lon_blo_filtered3[time,idx][~np.isnan(lon_blo_filtered3[time,idx])])
                                #print("block_candidate", block_candidate)
                                block_candidate_list=(block_candidate).tolist()
                                block_candidate_list = list(map(int, block_candidate_list))

                                ##which candidate has the most longitude in common
                                most_block_candidate = max(block_candidate_list, key = block_candidate_list.count)

                                ### check if the new candidate is in the block_checked. If yes we break the loop and going forward
                                if most_block_candidate in block_checked and indloop==0:
                                    ##on ajoute candidats a la liste checkée
                                    block_checked.extend(block_candidate_list)
                                    block_checked=list(set(block_checked))
                                    break
                                ###on ajoute cette valeur à la liste des blocages identifiés
                                block_true.append(int(most_block_candidate))
                                block_true=list(set(block_true))
                                ###on suit les indices du block confirmé a d+1
                                idx = np.where( lon_blo_filtered3 == most_block_candidate)
                                idx=idx[1].tolist()
                                indloop=indloop+1
                                ##on ajoute candidats a la liste checkée
                                block_checked.extend(block_candidate_list)
                                block_checked=list(set(block_checked))
                                #if block_val ==312:
                                #    print("most_block_candidate", most_block_candidate, block_checked )
                            else:
                                nb_day=nb_day+1
                                ### Sinon si on a pas une longitude en commun, on doit vérifier à longitude+-22.5 si il y a une longitude de bloquée
                                index_left=[]
                                index_right=[]                            
                                left=longitude[idx[0]]
                                right=longitude[idx[-1]]
                                left_e=left-22.5
                                right_e=right+22.5
                                #print("final_blo", final_blo)

                                if left_e<0:
                                    left_ext=np.arange(left_e+360,360,dx).tolist()
                                    left_ext1=np.arange(0,left,dx).tolist()
                                    left_ext.extend(left_ext1)
                                if right_e>=360:
                                    right_ext=np.arange(right,360,dx).tolist()
                                    right_ext1=np.arange(0,right_e-360,dx).tolist()
                                    right_ext.extend(right_ext1)
                                if left_e>0:
                                    left_ext=np.arange(left_e,left,dx).tolist()
                                if right_e<360:
                                    right_ext=np.arange(right,right_e,dx).tolist()
    
                                ### FIND indices from longitude blocked area
                                for i in range(len(left_ext)):
                                    index_left1 = longitude.tolist().index(left_ext[i])
                                    index_left.append(index_left1)
                                for i in range(len(right_ext)):
                                    index_right1 = longitude.tolist().index(right_ext[i])
                                    index_right.append(index_right1)       
                                    
                                ### BLOCK LEFT
                                #long expression to keep only block candidate values at d+1
                                block_candidate_left=np.unique(lon_blo_filtered3[time,index_left][~np.isnan(lon_blo_filtered3[time,index_left])])
                                block_candidate_left_list=(block_candidate_left).tolist()
                                block_candidate_left_list = list(map(int, block_candidate_left_list))
    
                                if len(block_candidate_left) !=0:
                                    most_block_candidate_left = max(block_candidate_left_list, key = block_candidate_left_list.count)
                                    leftcount = pd.Series(lon_blo_filtered3[time,index_left]).value_counts().to_dict()
                                    leftcount_o=leftcount[most_block_candidate_left]
                                else:
                                    most_block_candidate_left = np.nan
                                    leftcount_o =0                               
                               
                                ###BLOCK RIGHT
                                block_candidate_right=np.unique(lon_blo_filtered3[time,index_right][~np.isnan(lon_blo_filtered3[time,index_right])])
                                block_candidate_right_list=(block_candidate_right).tolist()
                                block_candidate_right_list = list(map(int, block_candidate_right_list))
         
                                if len(block_candidate_right) !=0:
                                    most_block_candidate_right = max(block_candidate_right_list, key = block_candidate_right_list.count)
                                    rightcount = pd.Series(lon_blo_filtered3[time,index_right]).value_counts().to_dict()
                                    rightcount_o=rightcount[most_block_candidate_right]
                                else:
                                    most_block_candidate_right=np.nan
                                    rightcount_o =0
                                #if block_val ==311:
                                #    print("most_block_LR", most_block_candidate_right, most_block_candidate_left )

                                ###condition si l'un des deux block a déjà été traité on zap:
                                if most_block_candidate_left not in block_checked and most_block_candidate_right not in block_checked : 
                                    ##on ajoute candidats a la liste checkée
                                    block_checked.extend(block_candidate_left_list)
                                    block_checked.extend(block_candidate_right_list)
                                    block_checked=list(set(block_checked))
                                    block_checked.sort()
        
                                    if leftcount_o > rightcount_o :
                                        ###on ajoute cette valeur à la liste des blocages identifiés
                                        candidate_true=most_block_candidate_left
                                        block_true.append(int(most_block_candidate_left))
                                        block_true=list(set(block_true))      
                                        ###on suit les indices du block confirmé a d+1
                                        idx = np.where( lon_blo_filtered3 == most_block_candidate_left)
                                        idx=idx[1].tolist()
                                    if leftcount_o < rightcount_o :
                                        candidate_true=most_block_candidate_right
                                        ###on ajoute cette valeur à la liste des blocages identifiés
                                        block_true.append(int(most_block_candidate_right))
                                        block_true=list(set(block_true))                              
                                        ###on suit les indices du block confirmé a d+1
                                        idx = np.where( lon_blo_filtered3 == most_block_candidate_right)
                                        idx=idx[1].tolist()
    
                                    if leftcount_o ==0 and rightcount_o ==0:
                                        if p==1:
                                            out=1
                                            p=0
                                        p=p+1
    
                                    ### Sortie finale+écriture de final_blo
                                    block_true.sort()
                                    #if candidate_true in block_checked:
                                    #    break
                                    if out==1 :
                                        ###keep only blocking >=5
                                        if len(block_true)>=5:
                                            #print("end block_true",block_true)
                                            block_true.sort()
                                            final_blo.append(block_true)
                                            #print("end final", final_blo)
                                            final_duration.append(nb_day-2)
                                        p=0
                                        block_true=[]
                                        block_checked.sort()
                                        break
                                else:
                                    break
                    keep_running=False
             
##################################################################################################
### COMPUTE FINAL BLOCKING CARACTERISTICS FROM BLOCKING LONGITUDE ATTRIBUTED TO ONE TRUE BLOCK ###
##################################################################################################
temp_list=[]
final_lon=[]
final_lat=[]
final_intensity=[]
final_date=[]
print("Compute final blocking lon/lat coordinates : New method")

##NEW WAY TO COMPUTE COORDINATES MEAN
###!!!! -1 car faut trouver solution pour le dernier blocage
for i in range(len(final_blo)):
    #print(final_blo[i])
    val_int_sum=0
    val_lon_sum=0
    val_lat_sum=0
    val_dat_sum=0
    #val_duration_sum=0
    temp_list=final_blo[i]
    temp_lon_list=[]
    temp_lat_list=[]
    temp_int_list=[]

    for y in range(len(temp_list)):
        indice=0
        val=temp_list[y]
        
        ### IMPORTANT : MUST FIND THE BLOCK VALUE CORRESPONDING TO FIRST STEP WHEN WE HAVE CALCULATED THE BLOCK CENTER
        
        index_first_tab=np.where(lon_blo_ind2==val)
        tt=index_first_tab[0][0]
        ii=index_first_tab[1][0]
        ##retrieve value in first tab with all blocks non filtered
        
        real_center_value=int(lon_blo_ind[tt,ii])
        
        temp_int_list.append(log_intensity_spa[val-1])
        temp_lon_list.append(log_lon[real_center_value-1])
        temp_lat_list.append(log_lat[real_center_value-1])

        if y==0:
        #    val_date=log_date_spa[val-1]     
            val_date=log_date[real_center_value-1]     
        
        #print("val", val,log_intensity_spa[val] )
        ###longitude latitude
        #print("debug", log_lon_spa[val-1], val )
        
        
        
        
        
        # ###intensity
        # val_int_sum=val_int_sum+log_intensity_spa[val-1]       
        # temp_lon_list.append(log_lon_spa[val-1])
        # temp_lat_list.append(log_lat_spa[val-1])
        # #val_lon_sum=val_lon_sum+log_lon_spa[val]
        # #val_lat_sum=val_lat_sum+log_lat_spa[val]
        # #val_duration_sum=val_duration_sum+log_date_spa[val]
    
    val_int_moy=np.mean(temp_int_list)     

    #val_int_moy=val_int_sum/len(temp_list)
    #val_lon_moy=val_lon_sum/len(temp_list)
    #val_lat_moy=val_lat_sum/len(temp_list)

    meancoord=mean_coord(temp_lon_list,temp_lat_list)
    lonrad=meancoord[0]
    latrad=meancoord[1]
    
    londeg=lonrad*180/np.pi
    latdeg=latrad*180/np.pi
    if londeg<0:
        londeg=360+londeg

    final_lon.append(londeg)
    final_lat.append(latdeg)    
    final_intensity.append(val_int_moy)
    final_date.append(val_date)
    if latdeg>73:
        print("list ",temp_list)
        print("debug lon ", temp_lon_list, val )
        print("debug lat ", temp_lat_list, val )
        print("lonlat mean ",londeg , latdeg)
    if val==307:
        print("list ",temp_list)
        print("debug lon ", temp_lon_list, val )
        print("debug lat ", temp_lat_list, val )
        print("lonlat mean ",londeg , latdeg)

arr=np.array(final_duration)
#final_duration=np.array(final_duration)

print("Creating plot and save all datas")

df_lon = pd.DataFrame (final_lon, columns = ['longitude'])
df_lon.to_csv('./logs/final_lon.csv', sep=',')
df_lat = pd.DataFrame (final_lat, columns = ['latitude'])
df_lat.to_csv('./logs/final_lat.csv', sep=',')
df_date = pd.DataFrame (final_date, columns = ['date'])
df_date.to_csv('./logs/final_date.csv', sep=',')
df_int = pd.DataFrame (final_intensity, columns = ['intentity'])
df_int.to_csv('./logs/final_int.csv', sep=',')
df_dur = pd.DataFrame (arr, columns = ['duration'])
df_dur.to_csv('./logs/final_duration.csv', sep=',')

df_longitude_block = pd.DataFrame (log_lon_spa, columns = ['longitude'])
df_longitude_block.to_csv('./logs/longitude_block.csv', sep=',')
df_latitude_block = pd.DataFrame (log_lat_spa, columns = ['latitude'])
df_latitude_block.to_csv('./logs/latitude_block.csv', sep=',')

x=np.asarray(final_lon)
y=np.asarray(final_lat)


fig = plt.figure(figsize=(26,9))    
bbox = [-170,30,170,75]   

m = Basemap(llcrnrlon=bbox[0],llcrnrlat=bbox[1],urcrnrlon=bbox[2],
            urcrnrlat=bbox[3],resolution='i', projection='mill')
m.fillcontinents(color='#d9b38c',lake_color='#bdd5d5') # continent colors
m.drawmapboundary(fill_color='#bdd5d5') # ocean color
m.drawcoastlines() 
m.drawcountries()
states = m.drawstates() # draw state boundaries
m.drawparallels(np.arange(-90,90,10),labels=[True,False,False,False])
m.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1])

m.scatter(final_lon,final_lat,c=final_intensity,latlon=True, s=20, marker='o', alpha=1, edgecolor='k', linewidth=1, zorder=2,cmap=plt.cm.jet)
#m.scatter(final_lon,final_lat,latlon=True, s=20, marker='o', alpha=1, edgecolor='k', linewidth=1, zorder=2,cmap=plt.cm.jet)
#m.hexbin(log_lon_spa,log_lat_spa,c=log_intensity_spa,gridsize=50, cmap='inferno')
#m.hexbin(x,y, cmap='inferno',zorder=2)
plt.colorbar(fraction=0.030)
plt.clim(0,4)
string_title=u'Blocking events intensities - ATL ANNUAL ALL -  '+str(yi)+'-'+str(yf)
plt.title(string_title, size='xx-large')
plt.savefig('./logs/blocage_atl_all_size_'+str(yi)+'-'+str(yf)+'.png', bbox_inches='tight', pad_inches=0.1)

# fig = plt.figure(figsize=(26,9))    
# bbox = [-170,30,170,75]   

# m = Basemap(llcrnrlon=bbox[0],llcrnrlat=bbox[1],urcrnrlon=bbox[2],
#             urcrnrlat=bbox[3],resolution='i', projection='mill')
# m.fillcontinents(color='#d9b38c',lake_color='#bdd5d5') # continent colors
# m.drawmapboundary(fill_color='#bdd5d5') # ocean color
# m.drawcoastlines() 
# m.drawcountries()
# states = m.drawstates() # draw state boundaries
# m.drawparallels(np.arange(-90,90,10),labels=[True,False,False,False])
# m.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1])

# #m.scatter(log_lon_spa,log_lat_spa,latlon=True, s=20, marker='o', alpha=1, edgecolor='k', linewidth=1, zorder=2,cmap=plt.cm.jet)
# ##m.scatter(log_lon_spa,log_lat_spa,c=log_intensity_spa,latlon=True, s=20, marker='o', alpha=1, edgecolor='k', linewidth=1, zorder=2,cmap=plt.cm.jet)
# ##m.hexbin(log_lon_spa,log_lat_spa,c=log_intensity_spa,gridsize=50, cmap='inferno')
# ##m.hexbin(x,y, cmap='inferno',zorder=2)
# #plt.colorbar(fraction=0.030)
# #plt.clim(0,4)
# #string_title=u'Blocking events intensities - ATL ANNUAL ALL -  '+str(yi)+'-'+str(yf)
# #plt.title(string_title, size='xx-large')
# #plt.savefig('./logs/longitude_'+str(yi)+'-'+str(yf)+'.png', bbox_inches='tight', pad_inches=0.1)