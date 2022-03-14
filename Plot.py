#########################################################################################################
### Contributers :Ahmed Ali                                                                           ###
### Date: Feb-18-2021                                                                                 ###
### Description:  Plotting and visiualizations                                                        ###
###                                                                                                   ###
### Refrences:                                                                                        ###
###           1.https://gadm.org/download_country.html                                                ###
###           2.https://stackoverflow.com/questions/48874113/concat-multiple-shapefiles-via-geopandas ###
###           3.#https://gis.stackexchange.com/questions/131716/plot-shapefile-with-matplotlib        ###
#########################################################################################################





#Preface
##########
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from pathlib import Path
import pandas
import geopandas

#PSetting Path
##############
ncFile='Output/outputNC/low2High_1992.nc'
outDir='Output/Plot/'

#Read Files
############
def readNCFile(ncFile,label='precip'):
    data=Dataset(ncFile, 'r')   
    return data.variables[label][:]

if __name__ == "__main__":   

    data=readNCFile(ncFile,'precip')
    plt.pcolormesh(data[0,:,:])
    plt.colorbar()
    #Plot Data (0-365) Days Range
    for day in range(len(data)):
        plt.pcolormesh(data[day,:,:])
        plt.grid()
        plt.show()
        #Save Plot Daywise
        plt.savefig(f'{outDir}plot_{day}.png')
