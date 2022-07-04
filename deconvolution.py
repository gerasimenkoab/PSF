from scipy import signal
from scipy import misc
import numpy as np



def PointFunction(pt, r0, r, maxIntensity):
    """Function of sphere of radius r with center in r0. 
    Function return maxIntensity if pt in sphere and 0 if out of sphere.
    pt and r0 are np.array vectors : np.array([x,y,z])"""
    if (pt-r0).dot(pt-r0) <= r * r :
        result = maxIntensity
    else:
        result = 0
    return result


def PointFunctionAiry(pt, r0, r, maxIntensity):
    """Function of sphere of radius r with center in r0. 
    Function return Airy disk intesity within first circle if pt in sphere and 0 if out of sphere.
    pt and r0 are np.array vectors : np.array([x,y,z])"""
    distSq = (pt-r0).dot(pt-r0)
    dist = sqrt(distSq) 
    if distSq <= r * r :
        x = dist/r*4.0
        result = (2*BesselJ(1, x)/x)^2
    else:
        result = 0
    return result


def MakeIdealSphereArray(imgSize = 36, sphRadius = 5):
    """create ideall sphere array"""
    imgMidCoord = 0.5 * (imgSize)
    imgCenter = np.array([imgMidCoord,imgMidCoord,imgMidCoord])
    tiffDraw = np.ndarray([imgSize,imgSize,imgSize])
    lightIntensity = 1000
    for i in range(imgSize):
      for j in range(imgSize):
        for k in range(imgSize):
          tiffDraw[i,j,k] = PointFunction(np.array([i,j,k]), imgCenter, sphRadius, lightIntensity)
    return tiffDraw

def MaxLikelhoodEstimationFFT_3D(pImg, idSphImg, iterLimit = 20,debug_flag = True):
    """Function for test of scipy convolution 
    """
    hm = pImg
    # if there is NAN in image array(seems from source image) replace it with zeros
    hm[np.isnan(hm)] = 0
    print("starting convolution:", pImg.shape,idSphImg.shape,hm.shape)
    b_noize = (np.mean(hm[0,0,:])+np.mean(hm[0,:,0])+np.mean(hm[:,0,0]))/3
    
    if debug_flag:
        print("Debug output:")
        print( np.mean(hm[0,0,:]),np.mean(hm[0,:,0]),np.mean(hm[:,0,0]) )
        print(np.amax(hm[0,0,:]),np.amax(hm[0,:,0]),np.amax(hm[:,0,0]))
        print(hm[0,0,56],hm[0,56,0], hm[56,0,0])
#        input("debug end")
#    b_noize = 0.1
    print("Background intensity:", b_noize)
    print('max intensity value:', np.amax(hm))
    p = idSphImg
# preparing for start of iteration cycle
    f_old = hm
#    f_old = p
    Hm = np.fft.fftn(hm)
    P = np.fft.fftn(p)
    #P_hat = np.fft.fftn(np.flip(p)) # spatially inverted p
# starting iteration cycle
    for k in range(0, iterLimit):
      print("iter:",k)
       # first convolution
      e = signal.fftconvolve(f_old, p, mode='same')
      #e = np.real(e)
      e = e + b_noize
      r = hm / e
      # second convolution
      p1=np.flip(p)
      rnew = signal.fftconvolve(r, p1, mode='same')
      rnew = np.real(rnew)
#      rnew = rnew.clip(min=0)
      f_old = f_old * rnew
# applying intensity regulatisation according to Conchello(1996) 
#      constr = np.amax(f_old)/np.amax(hm)
#      f_old = (-1.0+np.sqrt(1.0 + 2.0*constr*f_old))/(constr)
#      print("result:",hm[36,36,36],f_old[36,36,36],r[36,36,36],p[36,36,36],e[36,36,36],rnew[36,36,36])

#      f_old = f_old / np.amax(f_old) 
#  maximum  entropy regularisation - seems to work bad
#      f_old = f_old * rnew - 0.00001*rnew *np.log(rnew)
# end of iteration cycle

    xdim = f_old.shape[1]
    print("shape: ",xdim)
    xstart = xdim //4
    xend = xstart + xdim // 2
    hm = hm[xstart:xend,xstart:xend,xstart:xend]
    p = p[xstart:xend,xstart:xend,xstart:xend]
    f_old = f_old[xstart:xend,xstart:xend,xstart:xend]
    print("End of MaxLikelhoodEstimation fft")
    return f_old
