import numpy as np
import os
from scipy.ndimage import gaussian_filter, median_filter
from .ImageRaw_class import ImageRaw
from .decon_methods import DeconImage

import logging

# TODO : Maybe we need to add here neural denoizing
class PreprocessImageModel:
    """Image preprocessing module"""

    def __init__(self):
        super().__init__
        self.logger = logging.getLogger("__main__." + __name__)
        self.logger.info("Preprocessing object created")

        self._preprocImage = ImageRaw(None, [0.2, 0.089, 0.089], np.zeros((10, 200, 200)))
        self._preprocResult = (
            None  # ImageRaw( None, [0.2, 0.089, 0.089], np.zeros((10, 200, 200)) )
        )

        self._isNeedGaussBlur = False
        self._gaussBlurRad = 2

        self._isNeedMaximizeIntensity = False

    # Manipulating preprocessed image methods
    @property
    def preprocImage(self):
        return self._preprocImage

    @preprocImage.setter
    def preprocImage(self, value: ImageRaw):
        self._preprocImage = value

    def SetPreprocImage(self, fname=None, voxel=None, array=None):
        self._preprocImage = ImageRaw(fname, voxel, array)

    # Manipulating gauss blurring params methods
    @property
    def isNeedGaussBlur(self):
        return self._isNeedGaussBlur

    @isNeedGaussBlur.setter
    def isNeedGaussBlur(self, value: bool):
        self._isNeedGaussBlur = bool(value)

    @property
    def gaussBlurRad(self):
        return self._gaussBlurRad

    @isNeedGaussBlur.setter
    def gaussBlurRad(self, value):
        try:
            value = int(value)
        except:
            raise ValueError("Wrong iteration number: cant convert input value to integer value", "iteration-number-incorrect")
        if value > 0:
            self._gaussBlurRad = value
        else:
            raise ValueError("Wrong iteration number: uncorrect integer value", "iteration-number-incorrect")

    # Manipulating maximize intensities param methods
    @property
    def isNeedMaximizeIntensity(self):
        return self._isNeedMaximizeIntensity

    @isNeedMaximizeIntensity.setter
    def isNeedMaximizeIntensity(self, value: bool):
        self._isNeedMaximizeIntensity = bool(value)

    
    # Manipulating preprocessing result methods
    @property
    def preprocResult(self):
        return self._preprocResult

    # Main preprocess function
    # TODO : maybe implement 'progBarIn' and 'masterWidget' in functions, which provides 
    def PreprocessImage(self, progBarIn, masterWidget):
        try:
            # if there is no preprocessing
            result = self._preprocImage.imArray
            image_dtype = str(result.dtype)
            
            # Make max intensivity
            if self._isNeedMaximizeIntensity:
                # if image data is integer -> values in each point are in [0; type_max] interval
                if "uint" in image_dtype:
                    result = (result / np.amax(result) * np.iinfo(image_dtype).max).astype(image_dtype)
                # if image data is float -> values in each point are in [0; 1] interval
                else:
                    # TODO : multiplying by 255 is a bad habbit! we need to depend on some standarts! (like descibed upper)
                    result = (result / np.amax(result) * 255).astype(image_dtype)
                
            # Make blurring
            # TODO : Maybe it will not work with multi-channel image; maybe we need to make tuple of radiuses with 
            if self._isNeedGaussBlur:
                radiuses = [0] * len(result.shape)
                radiuses[1] = radiuses[2] = self._gaussBlurRad # only blur each layer
                print(radiuses, len(result.shape))
                result = gaussian_filter(result, sigma=5, radius=radiuses).astype(image_dtype)

        except Exception as e:
            self.logger.debug(str(e))
            return
        try:
            self._preprocResult = ImageRaw(
                None, list(self._preprocImage.voxel.values()), result
            )
        except Exception as e:
            self.logger.debug(str(e))
            return
        self.logger.info("Image preprocessed")
