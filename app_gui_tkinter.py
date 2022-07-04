from tkinter import *
from tkinter.messagebox import showerror, showinfo
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image

import os.path
from os import path
from PySimpleGUI.PySimpleGUI import Exit
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm

import file_inout as fio
import deconvolution as decon
import img_transform as imtrans

"""
TODO: 
- [x] implement bead images draw in tkinter with matplotlib subplot
- fix centering
- add bead extraction
- add full interpolation over Z(depth)-axis in extract beads module
- fix some obsolete interface buttons
"""

class MainWindowGUI(Tk):

    def __init__(self, master = None, wwidth=800, wheight = 2000):
        super().__init__()
        self.imgBeadRawLoad = FALSE

        self.title("Simple experimental PSF extractor")
        self.resizable(False,False)
        Label(self, text="").grid(row = 0, column = 0)         # blanc insert
        Label(self,text="Load avaraged bead image").grid(row=1,column = 1)

        self.beadImgPathW = Entry(self, width = 25, bg = 'white', fg = 'black')
        self.beadImgPathW.grid(row = 2, column = 1, sticky = 'w')
        Button(text = 'Select Bead Image', command = self.SelectBeadImage).grid(row=2,column=2)
        Button(text = 'Load image file', command = self.LoadBeadImageFile).grid(row=2,column=3)

        Button(text = 'Launch bead extractor').grid(row=3,column=1)
        Button(text = 'Show loaded bead image').grid(row=3,column=2)
        Button(text = 'Center image intensity',command = self.CenteringImageInt).grid(row=3,column=3)

        Label(text="Bead size(nm):").grid(row = 4,column = 1)
        self.beadSizeWgt = Entry(self, width = 15, bg = 'white', fg = 'black')
        self.beadSizeWgt.grid(row = 4, column = 2, sticky = 'w')

        Label(text="Resolution XY Z (nm/pixel):").grid(row = 5,column = 1)
        self.beadImXYResWgt = Entry(self, width = 15, bg = 'white', fg = 'black')
        self.beadImXYResWgt.grid(row = 5, column = 2, sticky = 'w')
        #Label(text="Resolution Z(nm/pixel)").grid(row = 5,column = 1)
        self.beadImZResWgt = Entry(self, width = 15, bg = 'white', fg = 'black')
        self.beadImZResWgt.grid(row = 5, column = 3, sticky = 'w')
        
        Label(text="Iteration number:").grid(row = 6,column = 1)
        self.iterNumWgt = Entry(self, width = 15, bg = 'white', fg = 'black')
        self.iterNumWgt.grid(row = 6, column = 2, sticky = 'w')

        Button(text = 'Calculate PSF', command = self.CalculatePSF).grid(row=7,column=1)

        Label(text="PSF folder").grid(row = 8,column = 1)
        self.folderPSFWgt = Entry(self, width = 15, bg = 'white', fg = 'black')
        self.folderPSFWgt.grid(row = 8, column = 2, sticky = 'w')
        Button(text = 'Browse').grid(row=9,column=3)
        
        Label(text="PSF file prefix").grid(row = 9,column = 1)
        self.filePrfxPSFWgt = Entry(self, width = 15, bg = 'white', fg = 'black')
        self.filePrfxPSFWgt.grid(row = 9, column = 2, sticky = 'w')

        Button(text = 'Save PSF multi-file',command=self.SavePSFMulti).grid(row=10, column=1)
        Button(text = 'Save PSF single-file',command=self.SavePSFSingle).grid(row=10, column=2)

        Button(text = 'EXIT!',command = self.destroy).grid(row=11,column=1)

        Label(self, text="").grid(row = 1, column = 4)         # blanc insert

        self.cnvImg = Canvas(self,  width = 150, height = 450, bg = 'white')
        self.cnvImg.grid(row = 1,column=5, rowspan=10,sticky=(N,E,S,W))
        self.cnvPSF = Canvas(self,  width = 150, height = 450, bg = 'white')
        self.cnvPSF.grid(row = 1,column=6, rowspan=10,sticky=(N,E,S,W))
        
        Label(self, text = "").grid(row = 12,column = 6) #blanc insert

    def SelectBeadImage(self):
        """Selecting bead file"""
        self.beadImgPath = askopenfilename(title = 'Load Beads Photo')
        self.beadImgPathW.insert(0,self.beadImgPath)

    def LoadBeadImageFile(self):
        """Loading raw bead photo from file at self.beadImgPath"""
        if not hasattr(self,'beadImgPath') :
            showerror("Error","Select bead image first!")
            return
        elif self.beadImgPath == "":
            showerror("Error","Bead image path empty!")
            return
        try:
#            self.imgBeadRaw = Image.open(self.beadImgPath)
#            print("Number of frames: ", self.imgBeadRaw.n_frames)
#            frameNumber = int( self.imgBeadRaw.n_frames / 2)
#            print("Frame number for output: ", frameNumber)
#            # setting imgTmp to desired number
#            self.imgBeadRaw.seek(frameNumber)
#            # preparing image for canvas from desired frame
#            self.cnvBeadImg = ImageTk.PhotoImage(self.imgBeadRaw)
            print("Open path: ",self.beadImgPath)
            self.imgBeadRaw = fio.ReadTiffStackFileTFF(self.beadImgPath)
            # creating figure with matplotlib
            fig, axs = plt.subplots(3, 1, sharex = False, figsize=(2,6))
            axs[0].pcolormesh(self.imgBeadRaw[self.imgBeadRaw.shape[0] // 2,:,:],cmap=cm.jet)
            axs[1].pcolormesh(self.imgBeadRaw[:,self.imgBeadRaw.shape[1] // 2,:],cmap=cm.jet)
            axs[2].pcolormesh(self.imgBeadRaw[:,:,self.imgBeadRaw.shape[2] // 2],cmap=cm.jet)
            # plt.show()
            # Instead of plt.show creating Tkwidget from figure
            self.figIMG_canvas_agg = FigureCanvasTkAgg(fig,self.cnvImg)
            self.figIMG_canvas_agg.get_tk_widget().grid(row = 1,column=5, rowspan=10,sticky=(N,E,S,W))

        except:
            showerror("LoadBeadImageFile: Error","Can't read file.")
            return
        # updating scrollers
        #self.cnv1.configure(scrollregion = self.cnv1.bbox('all'))  

    def LoadPSFImage(self):
        """Loading PSF from matplotlib object"""
        if not hasattr(self,'beadImgPath') :
            showerror("Error","Select bead image first!")
            return
        elif self.beadImgPath == "":
            showerror("Error","Bead image path is empty!")
            return

        try:
            self.imgBeadRaw = Image.open(self.beadImgPath)
            print("Number of frames: ", self.imgBeadRaw.n_frames)
            frameNumber = int( self.imgBeadRaw.n_frames / 2)
            print("Frame number for output: ", frameNumber)
            # setting imgTmp to desired number
            self.imgBeadRaw.seek(frameNumber)
            # preparing image for canvas from desired frame
            self.cnvBeadImg = ImageTk.PhotoImage(self.imgBeadRaw)
        except:
            showerror("Error","Can't read file.")
            return
        # replacing image on the canvas
        self.cnvImg.create_image(0, 0, image = self.cnvBeadImg, anchor = NW)
        # updating scrollers
        #self.cnv1.configure(scrollregion = self.cnv1.bbox('all'))  

    def CenteringImageInt(self):
        """Centering image array by intensity"""
        if not hasattr(self,'imgBeadRaw'):
            showerror("Error. No image loaded!")
            return
        try:
#                imgBeadRaw = imtrans.CenterImageIntensity(self.imgBeadRaw)
# FIXME: центровка работает как то странно. надо проверить и поправить.
            self.imgBeadRaw = imtrans.ShiftWithPadding(self.imgBeadRaw)
            # creating figure with matplotlib
            fig, axs = plt.subplots(3, 1, sharex = False, figsize=(2,6))
            axs[0].pcolormesh(self.imgBeadRaw[self.imgBeadRaw.shape[0] // 2,:,:],cmap=cm.gray)
            axs[1].pcolormesh(self.imgBeadRaw[:,self.imgBeadRaw.shape[1] // 2,:],cmap=cm.gray)
            axs[2].pcolormesh(self.imgBeadRaw[:,:,self.imgBeadRaw.shape[2] // 2],cmap=cm.gray)
            # plt.show()
            # Instead of plt.show creating Tkwidget from figure
            self.figIMG_canvas_agg = FigureCanvasTkAgg(fig,self.cnvImg)
            self.figIMG_canvas_agg.get_tk_widget().grid(row = 1,column=5, rowspan=10,sticky=(N,E,S,W))

        except:
            showerror("centering error")



    def CalculatePSF(self):
        txt_beadSizenm = self.beadSizeWgt.get()
        txt_resolutionXY = self.beadImXYResWgt.get()
        txt_resolutionZ = self.beadImZResWgt.get()
        txt_itNum = self.iterNumWgt.get()
        print(txt_beadSizenm,txt_resolutionXY,txt_resolutionZ)
        if not hasattr(self,'imgBeadRaw'):
            showerror("Error","No bead image loaded.")
        elif txt_beadSizenm == '' or txt_resolutionXY == ''or txt_resolutionZ == '':
            showerror("Error","Empty Bead size or resolution value.")
        elif txt_beadSizenm == '0' or txt_resolutionXY == '0' or txt_resolutionZ == '0':
            showerror("Error","Zero Bead size or resolution value.")
        elif txt_itNum == '0' or txt_itNum == '':
            txt_itNum = '10'  # default iteration number
        else:
            try:
                self.beadSizenm = float(txt_beadSizenm)
                self.resolutionXY = float(txt_resolutionXY)
                self.resolutionZ = float(txt_resolutionZ)
                self.beadSizepx = int(self.beadSizenm / self.resolutionXY / 2)
                self.itNum = int(txt_itNum)
                self.imArr1 = imtrans.PaddingImg(self.imgBeadRaw)
                print("shapes:",self.imArr1.shape[0],self.imArr1.shape[1],self.imArr1.shape[2])
                self.imgPSF = decon.MaxLikelhoodEstimationFFT_3D(self.imArr1, decon.MakeIdealSphereArray(self.imArr1.shape[0], self.beadSizepx), self.itNum)
                self.figPSF, axs = plt.subplots(3, 1, sharex = False, figsize=(2,6))
                axs[0].pcolormesh(self.imgPSF[self.imgBeadRaw.shape[0] // 2,:,:],cmap=cm.gray)
                axs[1].pcolormesh(self.imgPSF[:,self.imgBeadRaw.shape[0] // 2,:],cmap=cm.gray)
                axs[2].pcolormesh(self.imgPSF[:,:,self.imgBeadRaw.shape[0] // 2],cmap=cm.gray)
                # plt.show()
                # Instead of plt.show creating Tkwidget from figure
                self.figPSF_canvas_agg = FigureCanvasTkAgg(self.figPSF,self.cnvPSF)
                self.figPSF_canvas_agg.get_tk_widget().grid(row = 1,column=6, rowspan=10,sticky=(N,E,S,W))
            except:
                showerror("Error. Can't finish convolution properly.")
    def SavePSFMulti(self):
        """Save PSF array as single-page tiff files"""
        if hasattr(self, 'imgPSF')  :
            txt_folder = self.folderPSFWgt.get()
            txt_prefix = self.filePrfxPSFWgt.get()
            if txt_prefix == '':
                txt_prefix = "EML_psf"
            if txt_folder == '':
                dirId = -1
                while True:
                    dirId += 1
                    print(dirId)
                    txt_folder = str(os.getcwd()) + "\\"+"PSF_folder_"+str(dirId)
                    if not path.isdir(txt_folder):
                        print("creating dir")
                        os.mkdir(txt_folder)
                        break
            fio.SaveTiffFiles(self.imgPSF, txt_folder, txt_prefix)
            showinfo("PSF Files saved at:", txt_folder)

    def SavePSFSingle(self):
        """Save PSF array as multi-page tiff"""
        if hasattr(self, 'imgPSF')  :
            txt_folder = self.folderPSFWgt.get()
            txt_prefix = self.filePrfxPSFWgt.get()
            if txt_prefix == '':
                txt_prefix = "EML_psf"
            if txt_folder == '':
                dirId = -1
                while True:
                    dirId += 1
                    print(dirId)
                    txt_folder = str(os.getcwd()) + "\\"+"PSF_folder_"+str(dirId)
                    if not path.isdir(txt_folder):
                        print("creating dir")
                        os.mkdir(txt_folder)
                        break
            fio.SaveTiffStack(self.imgPSF, txt_folder, txt_prefix)
            showinfo("PSF File saved at:", txt_folder)

    def BeadExtractPlugin(self):
        self.BeadExtraction = BeadExtraction()



if __name__ == '__main__':
      rootWin = MainWindowGUI()
      rootWin.mainloop()

