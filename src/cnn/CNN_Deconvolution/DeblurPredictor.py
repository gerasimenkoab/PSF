import copy
import tkinter as tk
import tkinter.ttk as ttk

import os
import pathlib

import numpy as np

from .BigImageManager import BigImageManager
from .DeblurCNNModel2D import DeblurCNNModel2D
from .DeblurCNNModelMini3D import DeblurCNNModelMini3D
from .DeblurCNNModel3DExp import DeblurCNNModel3DExp

# Class, which provides predicting of output data
class DeblurPredictor:
    # constructor
    def __init__(self):
        # CONSTANTS
        self.CHUNK_SIZE = 64
        self.OFFSET_SIZE = 32

        self.CNN_MODEL_PATH_3D = "./3d_gaus_blur.h5"
        self.CNN_MODEL_PATH_2D = "./models/2d_gaus_blur.h5"

        self.isInited = False
        self.currentType = None
        return

    # Method which provides neural network model initialization to predict
    def initPredictModel(self, layers, rows, cols, _type):
        print(f"inputed: {layers, rows, cols, _type}")
        try:
            self.currentType = _type
            if _type == "3d deconvolution":
                _layers = layers
                _rows = self.CHUNK_SIZE + 2 * self.OFFSET_SIZE  # rows
                _cols = self.CHUNK_SIZE + 2 * self.OFFSET_SIZE  # cols

                # TODO : Maybe here we need to load model from server???
                print(f"1).")
                self.model = DeblurCNNModel3DExp.ModelBuilder(
                    input_shape=(_layers, _rows, _cols, 1)
                )
                print(f"2) path: {os.getcwd()}, {os.path.abspath(os.getcwd())}, {os.listdir(os.getcwd())}")
                self.model.load_weights(self.CNN_MODEL_PATH_3D)
                print(f"3).")
                self.isInited = True
            elif _type == "2d stack deconvolution":
                self.model = DeblurCNNModel2D.ModelBuilder(input_shape=(rows, cols, 1))
                self.model.load_weights(self.CNN_MODEL_PATH_2D)
                self.isInited = True
            else:
                raise Exception("Unknown deconvolution type")
            return
        except Exception as e:
            print(str(e))
            return

    # Msethod which provides post-processing
    def makePostprocessing(self, result, layers, rows, cols):
        result = result / np.amax(result)
        return result

    # TODO : maybe made progress bar with callbacks?
    # Method which provides 2d prediction on each layer
    def make2dStackPrediction(self, imgToPredict, window):
        layers = imgToPredict.shape[0]
        rows = imgToPredict.shape[1]
        cols = imgToPredict.shape[2]

        # split on layers
        imgLayers = [imgToPredict[i] for i in range(layers)]

        # make graphic indication progressbar
        pb = None
        if window != None:
            pb = ttk.Progressbar(
                window,
                orient="horizontal",
                mode="determinate",
                maximum=len(imgLayers),
                value=0,
            )
            pb.grid(row=10, column=2)

        # deconvolve layers
        resLayers = []
        for layer in imgLayers:
            resLayers.append(
                self.model.predict(layer.reshape(1, rows, cols, 1)).reshape(rows, cols)
            )
            if pb != None:
                pb["value"] = len(resLayers)
                window.update()

        # concatenate layers
        predictedImage = np.zeros(shape=(layers, rows, cols), dtype=np.float32)
        for i in range(len(resLayers)):
            predictedImage[i, :, :] = resLayers[i][:, :]
        return predictedImage, pb

    # Method which provides 3d prediction in whole image
    def make3dPrediction(self, imgToPredict, window):
        # make chunks
        chunksMaker = BigImageManager(imgToPredict, self.CHUNK_SIZE, self.OFFSET_SIZE)
        chunks = chunksMaker.SeparateInChunks()

        # make graphic indication progressbar
        pb = None
        if window != None:
            pb = ttk.Progressbar(
                window,
                orient="horizontal",
                mode="determinate",
                maximum=len(chunks),
                value=0,
            )
            pb.grid(row=10, column=2)

        results = []
        for chunk in chunks:
            chunkToPredict = chunk

            chunkToPredict.chunkData = chunk.chunkData.reshape(
                1, chunk.dataLayers, chunk.dataRows, chunk.dataCols, 1
            )

            chunkToPredict.chunkData = self.model.predict(chunkToPredict.chunkData)

            chunkToPredict.chunkData = chunkToPredict.chunkData.reshape(
                chunk.dataLayers, chunk.dataRows, chunk.dataCols
            )

            results.append(chunkToPredict)
            if pb != None:
                pb["value"] = len(results)
                window.update()

        # Init back to save
        result = chunksMaker.ConcatenateChunksIntoImage(results)
        return result, pb

    # Method which provides image's prediction
    def makePrediction(self, img, window):
        try:
            if not self.isInited:
                raise Exception("Model isnt inited!")

            print(f"4).")
            # filter noizes
            noize_lvl = 12
            zeros = np.zeros(shape=img.shape)
            img = np.where(img >= noize_lvl, img, zeros)
            print(f"5).")
            # prepaire to predict
            img = img.astype("float32") / 255
            imgToPredict = img.copy()
            print(f"6).")
            # predict
            if self.currentType == "3d deconvolution":
                result, pb = self.make3dPrediction(imgToPredict, window)
            elif self.currentType == "2d stack deconvolution":
                result, pb = self.make2dStackPrediction(imgToPredict, window)
            print(f"7).")
            # save results
            result = self.makePostprocessing(
                result, result.shape[0], result.shape[1], result.shape[2]
            )
            result = (result * 255).astype("uint8")

            # destroy progress bar
            if pb != None:
                pb.grid_remove()
            return result
        except Exception as e:
            print(str(e))
            return
