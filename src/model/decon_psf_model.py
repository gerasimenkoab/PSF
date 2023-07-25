import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from .ImageRaw_class import ImageRaw
from .decon_methods import DeconPSF

import logging

class DeconPsfModel():
    """
    Class model for PSF restoration from averaged bead photo.
    """
    def __init__(self):
        super().__init__
        self.logger = logging.getLogger('__main__.'+__name__)
        self.logger.info("Decon PSF object created")
        self._psfImage = ImageRaw( None, [0.2, 0.089, 0.089], np.zeros((10, 200, 200)) )
        self._resultImage = None
        self._beadDiameter = 0.2
        self._iterationNumber = 1
        self._regularizationParameter = 0.00001

    @property
    def PSFImage(self):
        return self._psfImage

    def SetPSFImage(self, fname=None, voxel=None, array=None):
        self._psfImage = ImageRaw(fname, voxel, array)

    @property
    def voxel(self):
        return self._psfImage.voxel

    def SetVoxelByAxis(self, axisName, value):
        try:
            value = float(value)
        except:
            raise ValueError("Wrong voxel value at " + axisName, "voxel-value-incorrect")

        if value > 0:
            self._psfImage.voxel[axisName] = value
        else:
            raise ValueError("Wrong voxel value at " + axisName, "voxel-value-incorrect")

    def SetVoxel(self, newVoxelSizeList):
        """Bead voxel size change"""
        try:
            self._psfImage.SetVoxel(newVoxelSizeList)
        except ValueError as vErr:
            raise ValueError(vErr.args[0],vErr.args[1])


    @property
    def resultImage(self):
        return self._resultImage

    @property
    def beadDiameter(self):
        return self._beadDiameter

    @beadDiameter.setter
    def beadDiameter(self, value):
        try:
            value = float(value)
        except:
            raise ValueError("Wrong bead diameter value", "beadDiameter_incorrect")

        if value > 0:
            self._beadDiameter = value
        else:
            raise ValueError("Wrong bead diameter value", "beadDiameter_incorrect")

    @property
    def iterationNumber(self):
        return self._iterationNumber

    @iterationNumber.setter
    def iterationNumber(self, value):
        try:
            value = int(value)
        except:
            raise ValueError("Wrong iteration number", "iteration-number-incorrect")
        if value > 0 :
            self._iterationNumber = value
        else:
            raise ValueError("Wrong iteration number", "iteration-number-incorrect")
        
    @property
    def regularizationParameter(self):
        return self._regularizationParameter

    @regularizationParameter.setter
    def regularizationParameter(self, value):
        try:
            value = float(value)
        except:
            raise ValueError("Wrong regularisation parameter value", "regularization-parameter-incorrect")
        if value < 1 and value > 0 :
            self._regularizationParameter = value
        else:
            raise ValueError("Wrong regularisation parameter value", "regularization-parameter-incorrect")

    def CalculatePSF(self, deconMethodIn : str, progBarIn, masterWidget):
        try:
            PSF = DeconPSF(
                self._psfImage.imArray,
                self._beadDiameter,
                self._iterationNumber,
                deconMethodIn,
                self._regularizationParameter,
                progBar=progBarIn,
                parentWin=masterWidget
                )
        except Exception as e:
            self.logger.debug(str(e))
            return
        try:
            self._resultImage = ImageRaw( None, list(self._psfImage.voxel.values()), PSF )
        except Exception as e:
            self.logger.debug(str(e))
            return
        self.logger.info("Experimental PSF calculated")
