from scipy import signal
from scipy import misc
import numpy as np
import itertools
from scipy.special import jv
from scipy.ndimage import laplace
from tkinter.messagebox import showerror, showinfo

from scipy.signal import convolve, fftconvolve
from scipy.ndimage import gaussian_filter


def DeconPSF(
    image: np.ndarray, beadSizePx: int, iterNum: int,
    deconType: str, lambdaR: float, progBar, parentWin
):
    """
    General function for restoration of PSF
    Input:
        image: np.ndarray  - averaged single bead image
        beadSizePx: int - actual bead size in pixels
        iterNum: int - number of iteration steps
        deconType: str  - deconvolution method name ( default - RL)
        lambdaR: float - regularization coefficient
        progBar : ttk.progressbar - progressbar to update
        parentWin : tkinter window widget
    Returns:
        imagePSF: np.ndarray
    """
    imagePSF: np.ndarray
    # if deconType == "RL":
    #     # Richardson Lucy
    #     print(deconType," selected")
    #     imagePSF = MaxLikelhoodEstimationFFT_3D(
    #         image,
    #         MakeIdealSphereArray(image.shape[0], beadSizePx),
    #         iterNum, False, progBar, parentWin
    #     )
    # elif deconType == "RLTMR":
    #     # Richardson Lucy with Tikhonov-Miller regularisation
    #     imagePSF = DeconvolutionRLTMR(
    #         image,
    #         MakeIdealSphereArray(image.shape[0], beadSizePx),
    #         lambdaR,
    #         iterNum, False, progBar, parentWin
    #     )
    # elif deconType == "RLTVR":
    #     # Richardson Lucy with Total Variation regularisation
    #     imagePSF = DeconvolutionRLTVR(
    #         image,
    #         MakeIdealSphereArray(image.shape[0], beadSizePx),
    #         lambdaR,
    #         iterNum, False, progBar, parentWin
    #     )
    # else:
    #     imagePSF = None
    #     print("DeconPSF: Invalid option. Please choose a valid option.")


    match deconType:
        case "RL":
            # Richardson Lucy
            print(deconType," selected")
            imagePSF = MaxLikelhoodEstimationFFT_3D(
                image,
                MakeIdealSphereArray(image.shape[0], beadSizePx),
                iterNum, False, progBar, parentWin
            )
        case "RLTMR":
            # Richardson Lucy with Tikhonov-Miller regularisation
            imagePSF = DeconvolutionRLTMR(
                image,
                MakeIdealSphereArray(image.shape[0], beadSizePx),
                lambdaR,
                iterNum, False, progBar, parentWin
            )
        case "RLTVR":
            # Richardson Lucy with Total Variation regularisation
            imagePSF = DeconvolutionRLTVR(
                image,
                MakeIdealSphereArray(image.shape[0], beadSizePx),
                lambdaR,
                iterNum, False, progBar, parentWin
            )
        case _:
            imagePSF = None
            print("DeconPSF: Invalid option. Please choose a valid option.")


    return imagePSF


def DeconImage(
    image: np.ndarray, kernell: np.ndarray,
    iterNum: int, deconType: str, lambdaR: float, progBar, parentWin
):
    """
    General function for restoration of image with known PSF(kernell)
    Input:
        image: np.ndarray  - averaged single bead image
        kernell: np.ndarray - PSF or other kernell
        iterNum: int - number of iteration steps
        deconType: str  - deconvolution method name ( default - RL)
        lambdaR: float - regularization coefficient
        progBar : ttk.progressbar - progressbar to update
        parentWin : tkinter window widget

    Returns:
        imagePSF: np.ndarray
    """
    imageDeconvolved: np.ndarray

    if deconType == "RL":
        # Richardson Lucy
        imageDeconvolved = MaxLikelhoodEstimationFFT_3D(
            image, kernell,
            iterNum, False, progBar, parentWin
        )
    elif deconType == "RLTMR":
        # Richardson Lucy with Tikhonov-Miller regularisation
        imageDeconvolved = DeconvolutionRLTMR(
            image, kernell,
            lambdaR, iterNum, False, progBar, parentWin
        )
    elif deconType == "RLTVR":
        # Richardson Lucy with Total Variation regularisation
        imageDeconvolved = DeconvolutionRLTVR(
            image, kernell,
            lambdaR, iterNum, False, progBar, parentWin
        )
    else:
        imagePSF = None
        print("DeconImage: Invalid option. Please choose a valid option.")

    return imageDeconvolved


def PointFunction(pt, r0, r, maxIntensity):
    """
    Function of sphere of radius r with center in r0.
    Function return maxIntensity if pt in sphere and 0 if out of sphere.
    pt and r0 are np.array vectors : np.array([x,y,z])
    """
    if (pt - r0).dot(pt - r0) <= r * r:
        result = maxIntensity
    else:
        result = 0
    return result


def PointFunctionAiryNotZoomed(pt, r0, maxIntensity=255, zoomfactor=2.6):
    """
    Function of sphere of radius r with center in r0.
    Function return Airy disk intesity within first circle if pt in sphere and 0 if out of sphere.
    pt and r0 are np.array vectors : np.array([x,y,z])
    All  dimension in pixels are equal to x-dimension
    """
    beadVoxelSize = 0.044
    beadDiameter = 0.2
    pt = pt * beadVoxelSize
    r0 = r0 * beadVoxelSize
    r = beadDiameter * zoomfactor / 2.0
    distSq = (pt - r0).dot(pt - r0)
    dist = np.sqrt(distSq)
    if distSq <= r * r:
        x = dist / r * 4.0
        # NOTE: If 'x' is equal zero - result == nan!. To prevent it - make result equal 'maxIntensity'
        if abs(x) >= 0.00001:  # Zero-criterion
            result = (2.0 * jv(1, x) / x) ** 2 * maxIntensity
        else:
            result = maxIntensity
    else:
        result = 0
    return result


def PointFunctionAiry(pt, r0, maxIntensity=255, zoomfactor=2.6):
    """
    Zoom of bead circle  from microscope
    Radius = self.BeadDiameter / 2
    Center  r0 - np.array[0:2].
    Function return Airy disk intesity within first circle if pt in sphere and 0 if out of sphere.
    pt and r0 are np.array vectors : np.array([x,y,z])
    All  dimension in pixels are equal to x-dimension
    """
    beadVoxelSize = 0.044
    beadDiameter = 0.2

    l = abs(r0[0] - pt[0]) * beadVoxelSize
    r = beadDiameter / 2.0
    if r**2 - l**2 > 0:
        R = np.sqrt(r**2 - l**2) * zoomfactor
        ptd = pt * beadVoxelSize
        r0d = r0 * beadVoxelSize

        distSq = (ptd[1] - r0d[1]) ** 2 + (ptd[2] - r0d[2]) ** 2
        dist = np.sqrt(distSq)
        x = dist / R * 4.0
        if abs(x) >= 0.00001:  # Zero-criterion
        #     # NOTE: If 'x' is equal zero - result == nan!. To prevent it - make result equal 'maxIntensity'
            result = (2.0 * jv(1, x) / x) ** 2 * maxIntensity
        else:
            result = maxIntensity

    else:
        result = 0

    return result


def MakeIdealSphereArray(imgSize=36, sphRadius=5):
    """
    Create ideal  sphere array corresponding to sphere_type
    """
    imgMidCoord = int(imgSize / 2)
    # imgSize = self.sideHalf *2
    imgCenter = np.array([imgMidCoord, imgMidCoord, imgMidCoord])
    tiffDraw = np.ndarray([imgSize, imgSize, imgSize])
    print("image size:", imgSize)
    print("sphere radius:", sphRadius)

    # tiffBit = self.tiffMenuBitDict[self.tiffSaveBitType.get()]

    # NOTE: get max intensity for different output bits types
    # lightIntensity = np.iinfo(tiffBit).max
    lightIntensity = 255

    for i, j, k in itertools.product(range(imgSize), repeat=3):
        tiffDraw[i, j, k] = PointFunctionAiry(
            np.array([i, j, k]), imgCenter, lightIntensity
        )
    print("Sphere created")
    return tiffDraw


def LoadIdealSphereArray(imgSize=36, sphRadius=5):
    """create ideall sphere array"""
    imgMidCoord = 0.5 * (imgSize)
    imgCenter = np.array([imgMidCoord, imgMidCoord, imgMidCoord])
    tiffDraw = np.ndarray([imgSize, imgSize, imgSize])
    lightIntensity = 1000
    for i in range(imgSize):
        for j in range(imgSize):
            for k in range(imgSize):
                tiffDraw[i, j, k] = PointFunction(
                    np.array([i, j, k]), imgCenter, sphRadius, lightIntensity
                )
    return tiffDraw


def MaxLikelhoodEstimationFFT_3D(pImg, idSphImg, iterLimit=20, debug_flag=False, pb = None , parentWin=None):
    """
    Function for  convolution with (Maximum likelihood estimaton)Richardson-Lucy method
    """
    hm = pImg
    # if there is NAN in image array(seems from source image) replace it with zeros
    hm[np.isnan(hm)] = 0
    beadMaxInt = np.amax(pImg)
#    pImg = pImg / beadMaxInt * np.amax(idSphImg)
    padSh = idSphImg.shape
    hm = np.pad(hm, list(zip(padSh, padSh)), "edge")
    b_noize = (np.mean(hm[0, 0, :]) + np.mean(hm[0, :, 0]) + np.mean(hm[:, 0, 0])) / 3

    if debug_flag:
        print("Debug output:")
        print(np.mean(hm[0, 0, :]), np.mean(hm[0, :, 0]), np.mean(hm[:, 0, 0]))
        print(np.amax(hm[0, 0, :]), np.amax(hm[0, :, 0]), np.amax(hm[:, 0, 0]))
        print(hm[0, 0, 56], hm[0, 56, 0], hm[56, 0, 0])
    #        input("debug end")
    #    b_noize = 0.1
    #    print("Background intensity:", b_noize)
    #    print('max intensity value:', np.amax(hm))
    p = idSphImg
    # preparing for start of iteration cycle
    f_old = hm
    P = np.fft.fftn(p)
    if pb != None:
        # initializing progressbar
        pb['value'] = 0
        pb_step = 100 / iterLimit
    # starting iteration cycle
    for k in range(0, iterLimit):
        print("\r", "iter:", k, end=" ")
        # first convolution
        e = signal.fftconvolve(f_old, p, mode="same")
        # e = np.real(e)
        e = e + b_noize
        r = hm / e
        # second convolution
        p1 = np.flip(p)
        rnew = signal.fftconvolve(r, p1, mode="same")
        rnew = np.real(rnew)
        #      rnew = rnew.clip(min=0)
        f_old = f_old * rnew

        f_old = f_old / np.amax(f_old) * beadMaxInt

        if pb != None:             # updaiting progressbar
            pb.step(pb_step)
            parentWin.update_idletasks()

    # end of iteration cycle

    f_old = f_old[padSh[0]:-padSh[0], padSh[1]:-padSh[1], padSh[2]:-padSh[2]]
    return f_old


def DeconvolutionRL(image, psf, iterLimit=20, debug_flag=False):
    """Function for  convolution"""

    hm = image
    # if there is NAN in image array(seems from source image) replace it with zeros
    hm[np.isnan(hm)] = 0.0
    # do image padding
    pad = psf.shape
    hm = np.pad(hm, ((pad[0], pad[0]), (pad[1], pad[1]), (pad[2], pad[2])), "edge")
    beadMaxInt = np.amax(image)

    p = psf

    b_noize = (np.mean(hm[0, 0, :]) + np.mean(hm[0, :, 0]) + np.mean(hm[:, 0, 0])) / 3

    if debug_flag:
        print("Debug output:")
        print("Background intensity:", b_noize, "Max intensity:", np.amax(hm))
        print(
            "Deconvoluiton input shape (image, padded, psf):",
            image.shape,
            hm.shape,
            psf.shape,
        )
    # preparing for start of iteration cycle
    f_old = hm
    #    Hm = np.fft.fftn(hm)
    #    P = np.fft.fftn(p)
    # starting iteration cycle
    for k in range(0, iterLimit):
        print("\r", "iter:", k, end=" ")
        # first convolution
        e = signal.fftconvolve(f_old, p, mode="same")
        # e = np.real(e)
        e = e + b_noize
        r = hm / e
        # second convolution
        p1 = np.flip(p)
        rnew = signal.fftconvolve(r, p1, mode="same")
        rnew = np.real(rnew)
        #      rnew = rnew.clip(min=0)
        f_old = f_old * rnew
        f_old = f_old  # / np.amax(f_old) * beadMaxInt
    # end of iteration cycle
    imSh = hm.shape
    pad = psf.shape
    f_old = f_old[
        pad[0] : imSh[0] - pad[0], pad[1] : imSh[1] - pad[1], pad[2] : imSh[2] - pad[2]
    ]
    if debug_flag:
        print("Debug output:")
        print("Deconvoluiton output shape :", f_old.shape)

    return f_old


def DeconvolutionRLTMR(
    image: np.ndarray,
    psf: np.ndarray,
    lambdaTM=0.0001,
    iterLimit=20,
    debug_flag=False, pb = None , parentWin=None
):
    """
    Function for  convolution Richardson Lucy Tikhonov Miller Regularization

    Parameters:
        image (ndarray): The 3D image to be deconvolved.
        psf (ndarray): The point spread function (PSF) of the imaging system.
        lambdaTM (float): The weight of the tikhonov miller regularization term. Defaults to 0.0001.
        iterLimit (int): The number of iterations to perform. Defaults to 10.
        debug_flag (bool) : debug output
        pb : tkinter.ttk.progressbar progressbar widget
        parentWin : tkinter window widget instance
 
    Returns:
        ndarray: The deconvolved image.    
    """

    hm = image
    # if there is NAN in image array(seems from source image) replace it with zeros
    hm[np.isnan(hm)] = 0.0
    padSh = psf.shape
    hm = np.pad(hm, list(zip(padSh, padSh)), "edge")
    beadMaxInt = np.amax(image)
    p = psf
    b_noize = (np.mean(hm[0, 0, :]) + np.mean(hm[0, :, 0]) + np.mean(hm[:, 0, 0])) / 3

    if debug_flag:
        print("Debug output:")
        print("Background intensity:", b_noize, "Max intensity:", np.amax(hm))
        print(
            "Deconvoluiton input shape (image, padded, psf):",
            image.shape,
            hm.shape,
            psf.shape,
        )
    #    b_noize = 0.1
    if pb != None:
        # initializing progressbar
        pb['value'] = 0
        pb_step = 100 / iterLimit
    # preparing for start of iteration cycle
    f_old = hm
    # starting iteration cycle
    for k in range(0, iterLimit):
        print("\r", "iter:", k, end=" ")
        # first convolution
        e = signal.fftconvolve(f_old, p, mode="same")
        # e = np.real(e)
        e = e + b_noize
        r = hm / e
        # second convolution
        p1 = np.flip(p)
        rnew = signal.fftconvolve(r, p1, mode="same")
        rnew = np.real(rnew)
        regTM = 1.0 + 2.0 * lambdaTM * laplace(f_old)
        f_old = f_old * rnew / regTM
        f_old = f_old / np.amax(f_old) * beadMaxInt
        if pb != None:
            # updating progressbar
            pb.step(pb_step)
            parentWin.update_idletasks()

    # xdim = f_old.shape[1]
    # xstart = xdim // 4
    # xend = xstart + xdim // 2
    # f_old = f_old[xstart:xend, xstart:xend, xstart:xend]
    f_old = f_old[padSh[0]:-padSh[0], padSh[1]:-padSh[1], padSh[2]:-padSh[2]]

    if debug_flag:
        print("Debug output:")
        print("Input image shape: ", image.shape)
        print("Deconvoluiton output shape :", f_old.shape)

    return f_old


def DeconvolutionRLTVR(
    image: np.ndarray,
    psf: np.ndarray,
    lambdaTV=0.0001,
    iterLimit=10,
    debug_flag=False, pb = None , parentWin=None
):
    """
    Perform Richardson-Lucy deconvolution on a 3D image using a given point spread function (psf)
    with total variation regularization.

    Parameters:
        image (ndarray): The 3D image to be deconvolved.
        psf (ndarray): The point spread function (PSF) of the imaging system.
        lambdaTV (float): The weight of the total variation regularization term. Defaults to 0.0001.
        iterLimit (int): The number of iterations to perform. Defaults to 10.
        debug_flag (bool) : debug output
        pb : tkinter.ttk.progressbar progressbar widget
        parentWin : tkinter window widget instance
 
    Returns:
        ndarray: The deconvolved image.    
    """

    hm = image
    # if there is NAN in image array(seems from source image) replace it with zeros
    hm[np.isnan(hm)] = 0.0
    padSh = psf.shape
    hm = np.pad(hm, list(zip(padSh, padSh)), "edge")
    beadMaxInt = np.amax(image)
    p = psf
    b_noize = (np.mean(hm[0, 0, :]) + np.mean(hm[0, :, 0]) + np.mean(hm[:, 0, 0])) / 3

    if debug_flag:
        print("Debug output:")
        print("Background intensity:", b_noize, "Max intensity:", np.amax(hm))
        print(
            "Deconvoluiton input shape (image, padded, psf):",
            image.shape,
            hm.shape,
            psf.shape,
        )
    #    b_noize = 0.1
    if pb != None:
        # initializing progressbar
        pb['value'] = 0
        pb_step = 100 / iterLimit
    # preparing for start of iteration cycle
    f_old = hm
    # starting iteration cycle
    lambdaTV = 0.005
    for k in range(0, iterLimit):
        print("\r", "iter:", k, end=" ")
        # first convolution
        e = signal.fftconvolve(f_old, p, mode="same")
        # e = np.real(e)
        e = e + b_noize
        r = hm / e
        # second convolution
        p1 = np.flip(p)
        rnew = signal.fftconvolve(r, p1, mode="same")
        rnew = np.real(rnew)
        #      rnew = rnew.clip(min=0)
        # https://stackoverflow.com/questions/11435809/compute-divergence-of-vector-field-using-python
        # regTV = 1.0 - lambdaTV * divergence(np.gradient(f_old))
        gr = np.gradient(f_old)
        regTV = 1.0 - lambdaTV * np.sqrt(gr[0] ** 2 + gr[1] ** 2 + gr[2] ** 2)
        f_old = f_old * rnew / regTV
        f_old = f_old / np.amax(f_old) * beadMaxInt
        if pb != None:
            # updaiting progressbar
            pb.step(pb_step)
            parentWin.update_idletasks()

    # end of iteration cycle
    # xdim = f_old.shape[1]
    # xstart = xdim // 4
    # xend = xstart + xdim // 2
    # f_old = f_old[xstart:xend, xstart:xend, xstart:xend]
    padSh = psf.shape
    f_old = f_old[padSh[0]:-padSh[0], padSh[1]:-padSh[1], padSh[2]:-padSh[2]]

 
    if debug_flag:
        print("Debug output:")
        print("Input image shape: ", image.shape)
        print("Deconvoluiton output shape :", f_old.shape)

    return f_old


# def EM_MLE_3D(pImg, idSphImg, iterLimit=20, debug_flag=True):
#     """Function for  convolution"""
#     hm = pImg
#     beadMaxInt = np.amax(pImg)

#     # if there is NAN in image array(seems from source image) replace it with zeros
#     hm[np.isnan(hm)] = 0
#     print("starting convolution:", pImg.shape, idSphImg.shape, hm.shape)
#     b_noize = (np.mean(hm[0, 0, :]) + np.mean(hm[0, :, 0]) + np.mean(hm[:, 0, 0])) / 3

#     if debug_flag:
#         print("Debug output:")
#         print(np.mean(hm[0, 0, :]), np.mean(hm[0, :, 0]), np.mean(hm[:, 0, 0]))
#         print(np.amax(hm[0, 0, :]), np.amax(hm[0, :, 0]), np.amax(hm[:, 0, 0]))
#         print(hm[0, 0, 56], hm[0, 56, 0], hm[56, 0, 0])
#     #    print("Background intensity:", b_noize)
#     #    print('max intensity value:', np.amax(hm))
#     p = idSphImg
#     # preparing for start of iteration cycle
#     f_old = hm
#     #    f_old = p
#     Hm = np.fft.fftn(hm)
#     P = np.fft.fftn(p)
#     # P_hat = np.fft.fftn(np.flip(p)) # spatially inverted p
#     # starting iteration cycle
#     for k in range(0, iterLimit):
#         print("\r", "iter:", k, end=" ")
#         # first convolution
#         e = signal.fftconvolve(f_old, p, mode="same")
#         # e = np.real(e)
#         e = e + b_noize
#         r = hm / e
#         # second convolution
#         p1 = np.flip(p)
#         rnew = signal.fftconvolve(r, p1, mode="same")
#         rnew = np.real(rnew)
#         #      rnew = rnew.clip(min=0)
#         f_old = f_old * rnew
#         # applying intensity regulatization according to Conchello(1996)
#         #      constr = np.amax(f_old)/np.amax(hm)
#         #      f_old = (-1.0+np.sqrt(1.0 + 2.0*constr*f_old))/(constr)
#         #      print("result:",hm[36,36,36],f_old[36,36,36],r[36,36,36],p[36,36,36],e[36,36,36],rnew[36,36,36])

#         f_old = f_old / np.amax(f_old) * beadMaxInt
#     #  maximum  entropy regularisation - seems to work bad
#     #      f_old = f_old * rnew - 0.00001*rnew *np.log(rnew)
#     # end of iteration cycle

#     xdim = f_old.shape[1]
#     print("shape: ", xdim)
#     xstart = xdim // 4
#     xend = xstart + xdim // 2
#     hm = hm[xstart:xend, xstart:xend, xstart:xend]
#     p = p[xstart:xend, xstart:xend, xstart:xend]
#     f_old = f_old[xstart:xend, xstart:xend, xstart:xend]
#     print("End of MaxLikelhoodEstimation fft")
#     return f_old


# def Tikhonov_Miller_3D(pImg, idSphImg, alpha=0.1, debug_flag=True):
#     """Function for Tikhonov_Miller convolution
#     A*x = b
#     pImp -> b
#     idSphImg -> A
#     """
#     hm = pImg
#     # if there is NAN in image array(seems from source image) replace it with zeros
#     hm[np.isnan(hm)] = 0
#     beadMaxInt = np.amax(pImg)
#     print("starting Tikhonov:", pImg.shape, idSphImg.shape, hm.shape)

#     if debug_flag:
#         print("Debug output:")
#         print(np.mean(hm[0, 0, :]), np.mean(hm[0, :, 0]), np.mean(hm[:, 0, 0]))
#         print(np.amax(hm[0, 0, :]), np.amax(hm[0, :, 0]), np.amax(hm[:, 0, 0]))
#         print(hm[0, 0, 56], hm[0, 56, 0], hm[56, 0, 0])
#     G = alpha * np.identity
#     # x = (At*A + Gt*G)^{-1} *At*b
#     x = np.transpose(A) * A

#     p = idSphImg
#     # preparing for start of iteration cycle
#     f_old = hm
#     #    f_old = p
#     Hm = np.fft.fftn(hm)
#     P = np.fft.fftn(p)
#     # P_hat = np.fft.fftn(np.flip(p)) # spatially inverted p
#     # starting iteration cycle
#     for k in range(0, iterLimit):
#         print("\r", "iter:", k, end=" ")
#         # first convolution
#         e = signal.fftconvolve(f_old, p, mode="same")
#         # e = np.real(e)
#         e = e + b_noize
#         r = hm / e
#         # second convolution
#         p1 = np.flip(p)
#         rnew = signal.fftconvolve(r, p1, mode="same")
#         rnew = np.real(rnew)
#         #      rnew = rnew.clip(min=0)
#         f_old = f_old * rnew
#         # applying intensity regulatization according to Conchello(1996)
#         #      constr = np.amax(f_old)/np.amax(hm)
#         #      f_old = (-1.0+np.sqrt(1.0 + 2.0*constr*f_old))/(constr)
#         #      print("result:",hm[36,36,36],f_old[36,36,36],r[36,36,36],p[36,36,36],e[36,36,36],rnew[36,36,36])

#         f_old = f_old / np.amax(f_old) * beadMaxInt
#     #  maximum  entropy regularisation - seems to work bad
#     #      f_old = f_old * rnew - 0.00001*rnew *np.log(rnew)
#     # end of iteration cycle

#     xdim = f_old.shape[1]
#     print("shape: ", xdim)
#     xstart = xdim // 4
#     xend = xstart + xdim // 2
#     hm = hm[xstart:xend, xstart:xend, xstart:xend]
#     p = p[xstart:xend, xstart:xend, xstart:xend]
#     f_old = f_old[xstart:xend, xstart:xend, xstart:xend]
#     print("End of TikhonovMillerEstimation fft")
#     return f_old


# def richardson_lucy_deconvolution_3d_tv(
#     image, psf, iterations=10, beta=0.5, tv_weight=0.1
# ):
#     """
#     Perform Richardson-Lucy deconvolution on a 3D image using a given point spread function (PSF)
#     with total variation regularization.

#     Parameters:
#     image (ndarray): The 3D image to be deconvolved.
#     psf (ndarray): The point spread function (psf) of the imaging system.
#     iterations (int): The number of iterations to perform. Defaults to 10.
#     beta (float): The step size parameter. Defaults to 0.5.
#     tv_weight (float): The weight of the total variation regularization term. Defaults to 0.1.
#     mode (str): The mode parameter passed to the convolution function. Defaults to 'nearest'.

#     Returns:
#     ndarray: The deconvolved image.
#     """
#     print("deconvolution start")
#     # Normalize the image and the PSF
#     image = image / np.max(image)
#     psf = psf / np.max(psf)

#     # Pad the PSF to match the size of the image
#     # psf = np.pad(psf, ((image.shape[0]-psf.shape[0])//2, (image.shape[1]-psf.shape[1])//2, (image.shape[2]-psf.shape[2])//2), mode='constant')

#     print("deconvolution start")
#     # Initialize the deconvolved image to the input image
#     deconvolved = np.copy(image)
#     # Pad the image to match the size of the PSF
#     pad = psf.shape
#     imagepadded = np.pad(
#         image,
#         (
#             (pad[0] // 2, pad[0] // 2),
#             (pad[1] // 2, pad[1] // 2),
#             (pad[2] // 2, pad[2] // 2),
#         ),
#         "edge",
#     )
#     deconvolved = np.pad(
#         deconvolved,
#         (
#             (pad[0] // 2, pad[0] // 2),
#             (pad[1] // 2, pad[1] // 2),
#             (pad[2] // 2, pad[2] // 2),
#         ),
#         "edge",
#     )
#     # Compute the gradient operators
#     dx = np.array([-1, 0, 1]).reshape((1, 1, 3))
#     dy = np.array([-1, 0, 1]).reshape((1, 3, 1))
#     dz = np.array([-1, 0, 1]).reshape((3, 1, 1))

#     # Iterate the Richardson-Lucy algorithm with TV regularization
#     for i in range(iterations):
#         print("iteration number:", i)
#         try:
#             # Compute the convolution of the current deconvolved image with the PSF
#             convolved = fftconvolve(deconvolved, psf, mode="same")
#             print("deconvolution start")

#             # Compute the ratio of the input image to the convolution
#             ratio = imagepadded / convolved
#             print("deconvolution start")

#             # Compute the convolution of the ratio with the PSF
#             deconvolved *= fftconvolve(ratio, psf, mode="same")
#             print("deconvolution start")

#             # Compute the TV gradient of the deconvolved image
#             grad_x = convolve(deconvolved, dx, mode="same")
#             grad_y = convolve(deconvolved, dy, mode="same")
#             grad_z = convolve(deconvolved, dz, mode="same")
#             print("deconvolution start")

#             # Compute the total variation gradient of the deconvolved image
#             tv_grad = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
#             print("deconvolution start")

#             # Compute the Laplacian operator of the TV gradient
#             laplacian = convolve(tv_grad, np.array([[-1, 2, -1]]), mode="same")
#             print("deconvolution start")

#             # Update the deconvolved image with TV regularization
#             deconvolved *= fftconvolve(
#                 ratio * np.exp(beta * laplacian), psf, mode="same"
#             )
#             print("deconvolution start")

#             # Apply Gaussian filtering to the deconvolved image
#             deconvolved = gaussian_filter(deconvolved, sigma=1)
#         except Exception as e:
#             print(e)
#     # Removing padding to keep size same as image
#     deconvolved = deconvolved[
#         pad[0] // 2 : -pad[0] // 2,
#         pad[1] // 2 : -pad[1] // 2,
#         pad[2] // 2 : -pad[2] // 2,
#     ]
#     return deconvolved

# def divergence(F):
#     """compute the divergence of n-D scalar field `F`"""
#     return reduce(np.add, np.gradient(F))
