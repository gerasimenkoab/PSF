import numpy as np
import os
from scipy.ndimage import gaussian_filter, median_filter
from .ImageRaw_class import ImageRaw
from ..cnn.CNN_Deconvolution.DeblurPredictor import DeblurPredictor

import logging

# TODO : Add model from server loading
class CNNDeconvModel:
    """Image Deconvolution module"""

    def __init__(self):
        super().__init__
        self.logger = logging.getLogger("__main__." + __name__)
        self.logger.info("Decon Image object created")

        self._deconImage = ImageRaw(None, [0.2, 0.089, 0.089], np.zeros((10, 200, 200)))
        self._deconResult = None

    @property
    def deconImage(self):
        return self._deconImage

    @deconImage.setter
    def deconImage(self, value: ImageRaw):
        self._deconImage = value

    def SetDeconImage(self, fname=None, voxel=None, array=None):
        self._deconImage = ImageRaw(fname, voxel, array)

    @property
    def deconResult(self):
        return self._deconResult

    @deconResult.setter
    def deconResult(self, value: ImageRaw):
        self._deconResult = value

    def DeconvolveImage(self, progBarIn, masterWidget):
        try:
            print(f"In deconv method")
            predictor = DeblurPredictor()
            
            print(f"predictor: {predictor}")

            # TODO : JUST FOR DEBUG - DELETE THIS ROW LATER
            self._deconImage.imArray = self._deconImage.imArray[0:self._deconImage.imArray.shape[0] - self._deconImage.imArray.shape[0] % 8,:,:]
            predictor.initPredictModel(self._deconImage.imArray.shape[0],
                                            self._deconImage.imArray.shape[1],
                                            self._deconImage.imArray.shape[2],
                                            "3d deconvolution")
            
            print(f"predictor is inited: {predictor.isInited}")
            result_img = predictor.makePrediction(self._deconImage.imArray, None)
            print(f"res: {result_img}")
        except Exception as e:
            self.logger.debug(str(e))
            return
        try:
            self._deconResult = ImageRaw(
                None, list(self._deconImage.voxel.values()), result_img
            )
        except Exception as e:
            self.logger.debug(str(e))
            return
        self.logger.info("CNN deconv completed.")
