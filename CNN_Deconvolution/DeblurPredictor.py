import numpy as np

from CNN_Deconvolution.DeblurCNNModelMini3D import DeblurCNNModelMini3D
from CNN_Deconvolution.BigImageManager import BigImageManager

import copy

# Class, which provides predicting of output data
class DeblurPredictor:
    # constructor
    def __init__(self):
        # CONSTANTS
        self.CHUNK_SIZE = 48
        self.OFFSET_SIZE = 8

        self.isInited = False
        return

    # Method which provides neural network model initialization to predict
    def initPredictModel(self, layers, rows, cols, modelPath):
        self.model = DeblurCNNModelMini3D.ModelBuilder(input_shape=(layers, rows, cols, 1))
        self.model.load_weights(modelPath)
        self.isInited = True
        return

    # Msethod which provides post-processing
    def makePostprocessing(self, result, layers, rows, cols):
        for k in range(layers):
            for i in range(rows):
                for j in range(cols):
                    result[k][i][j] = min(1, max(0, result[k][i][j]))
        #result = result / np.amax(result)
        return result

    # Method which provides image's prediction
    def makePrediction(self, img):
        if not self.isInited:
            raise Exception("Model isnt inited!")

        # prepaire to predict
        img = img.astype('float32') / 255
        imgToPredict = img.copy()

        # make chunks
        chunksMaker = BigImageManager(imgToPredict, self.CHUNK_SIZE, self.OFFSET_SIZE)
        chunks = chunksMaker.SeparateInChunks()

        results = []
        for chunk in chunks:
            chunkToPredict = copy.copy(chunk)

            chunkToPredict.chunkData = chunk.chunkData.reshape(1, chunk.dataLayers, chunk.dataRows, chunk.dataCols, 1)

            chunkToPredict.chunkData = self.model.predict(chunkToPredict.chunkData)

            chunkToPredict.chunkData = chunkToPredict.chunkData.reshape(chunk.dataLayers, chunk.dataRows, chunk.dataCols)
            
            results.append(chunkToPredict)

        # Init back to save
        result = chunksMaker.ConcatenateChunksIntoImage(results)
        result = self.makePostprocessing(result, result.shape[0], result.shape[1], result.shape[2])
        result = (result * 255).astype('uint8')
        return result
