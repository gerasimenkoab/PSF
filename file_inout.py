import numpy as np
import tifffile as tff  # https://pypi.org/project/tifffile/
from PIL import Image


def ReadTiffStackFile(fileName):
    """Function ReadTiffStackFile() reads tiff stack from file and return np.array"""
    print("Loading Image from tiff stack file..... ")
    try:
        image_tiff = Image.open(fileName)
        ncols, nrows = image_tiff.size
        nlayers = image_tiff.n_frames
        imgArray = np.ndarray([nlayers, nrows, ncols])
        for i in range(nlayers):
            image_tiff.seek(i)
            imgArray[i, :, :] = np.array(image_tiff)
        print("Done!")
        return imgArray
    except FileNotFoundError:
        print("ReadTiffStackFile: Error. File not found!")
        return 0


def ReadTiffStackFileTFF(fileName):
    """Function ReadTiffStackFile() reads tiff stack from file and return np.array"""
    print("Loading Image from tiff stack file..... ", end=" ")
    try:
        image_stack = tff.imread(fileName)
        print("Done.")
        return image_stack
    except FileNotFoundError:
        print("ReadTiffStackFileTFF: Error. File not found!")
        return 0


def SaveTiffFiles(tiffDraw=np.zeros([3, 4, 6]), dirName="img", filePrefix=""):
    """Print files for any input arrray of intensity values
    tiffDraw - numpy ndarray of intensity values"""
    layerNumber = tiffDraw.shape[0]
    for i in range(layerNumber):
        im = Image.fromarray(tiffDraw[i, :, :])
        im.save(dirName + "\\" + filePrefix + str(i).zfill(2) + ".tiff")


def SaveTiffStack(tiffDraw=np.zeros([3, 4, 6]), dirName="img", filePrefix="!stack"):
    """Print files for any input arrray of intensity values
    tiffDraw - numpy ndarray of intensity values"""
    print("trying to save file")
    path = dirName + "\\" + filePrefix + ".tif"
    imlist = []
    for tmp in tiffDraw:
        #        print(tmp.shape,type(tmp))
        imlist.append(Image.fromarray(tmp.astype("uint16")))

    imlist[0].save(path, save_all=True, append_images=imlist[1:])
    print("file saved in one tiff", dirName + "\\" + filePrefix + ".tiff")


def SaveTiffStackTFF(tiffDraw=np.zeros([3, 4, 6]), dirName="img", filePrefix="!stack"):
    """Print files for any input arrray of intensity values
    tiffDraw - numpy ndarray of intensity values"""
    print("trying to save file")
    outTiff = np.rint(tiffDraw).astype("uint16")
    print("outTiff type: ", tiffDraw.dtype)
    #    tff.imwrite(dirName+"\\"+filePrefix+".tiff", outTiff)
    tff.imwrite(dirName + "\\" + filePrefix + ".tiff", tiffDraw, dtype=tiffDraw.dtype)
    print("file saved in one tiff", dirName + "\\" + filePrefix + ".tiff")
