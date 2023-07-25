from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilenames, asksaveasfilename
from tkinter.simpledialog import askstring
import logging


try:
    from .AuxTkPlot_class import AuxCanvasPlot as CnvPlot
except ImportError:
    from AuxTkPlot_class import AuxCanvasPlot as CnvPlot
try:
    from view.decon_view_psf import DeconvolvePSFFrame
except:
    from decon_view_psf import DeconvolvePSFFrame
try:
    from view.decon_view_image import DeconvolveImageFrame
except:
    from decon_view_image import DeconvolveImageFrame



"""   TODO:
        
        - add  bead size to tiff tag
"""


class DeconView:
    def __init__(self, master=None):
        # build ui
        self.logger = logging.getLogger('__main__.'+__name__)

        self.deconViewToplevel = tk.Tk() if master is None else tk.Toplevel(master)
        self.deconViewToplevel.configure(
            height=768,
            padx=5,
            pady=5,
            takefocus=True,
            width=1024)
        self.deconViewToplevel.geometry("950x600")
        self.deconViewToplevel.maxsize(1920, 1080)
        self.deconViewToplevel.minsize(950, 600)
        self.deconViewToplevel.resizable(True, True)
        self.deconViewToplevel.title("Deconvolution widget")

        self.deconNotebook = ttk.Notebook(self.deconViewToplevel)
        self.deconNotebook.configure(height=700, width=900)
        self.deconNotebook.pack(expand=True, fill="both", side="top")

        self.deconPsfView = DeconvolvePSFFrame(self.deconNotebook)
        self.deconNotebook.add(self.deconPsfView, text = "PSF deconvolution")

        self.deconImageView = DeconvolveImageFrame(self.deconNotebook)
        self.deconNotebook.add(self.deconImageView, text = "Image deconvolution")

        self.logOutputLabel = ttk.Label(self.deconViewToplevel)
        self.logOutStringVar = tk.StringVar(value = 'Log Output')
        self.logOutputLabel.configure(
            compound = "top",
            text = 'Log Output',
            textvariable = self.logOutStringVar)
        self.logOutputLabel.pack(fill="x", side = "bottom")

        # Main widget
        self.mainwindow = self.deconViewToplevel
        self.logger.info("Decon PSF view loaded")

# ======= Auxilary Functions ==========================

    def SetValueWidgetNormal(self, widget, value):
        widget.delete(0, END)
        widget.insert(0, str(value))

    def SetValueWidgetReadonly(self, widget, infoStr):
        widget.configure( state = "normal" )
        widget.delete(0,END)
        widget.insert( 0, infoStr )
        widget.configure( state = "readonly" )

    def GetVoxelDialog(self, widget, textInfo=""):
        """
        Create diealog and return list of values
        """
        voxelStr = askstring(parent = widget, title = "Voxel Dialog", prompt = textInfo)
        try:
            voxelList = [float(a) for a in voxelStr.split(",")]
        except:
            raise ValueError("Can not get voxel values", "bad-voxel-string")
        if voxelStr == "":
            raise ValueError("Can not get voxel values", "bad-voxel-string")
        return voxelList

    def GetFileNamesList(self, widget, titleTxt = ""):
        self.logger.debug("GetFileNamesList called")
        try:
            fNames = askopenfilenames(parent = widget, title = titleTxt)
        except:
            raise ValueError("Can not get file names","no-filenames-read")
        if fNames == "":
            raise ValueError("Can not get file names","no-filenames-read")
        return fNames

    def GetFileName(self, widget, titleTxt = ""):
        self.logger.debug("GetSaveAsFileName called")
        try:
            fName = asksaveasfilename(parent = widget, title = titleTxt)
        except:
            raise ValueError("Can not get file names","no-filenames-read")
        if fName == "":
            raise ValueError("Can not get file names","no-filenames-read")
        return fName

# ======= PSF deconvolution Widget Functions ===========
    def SetVoxelValues(self, voxelInDict):
        """Bead voxel size change"""
        if voxelInDict is None:
            raise ValueError("No voxel values recived", "voxel_is_none")
        for axisName in ["X","Y","Z"]:
            self.SetValueWidgetNormal(self.deconPsfView.voxel_entry[axisName], voxelInDict[axisName])

    def SetBeadSize(self, valueIn):
        """Bead diameter size change"""
        try:
            self.SetValueWidgetNormal(self.deconPsfView.beadSize_entry, abs(float(valueIn)))
        except:
            self.logger.info("Bead Size: Bad input")
            self.SetValueWidgetNormal(self.deconPsfView.beadSize_entry, self.deconPsfView.beadSize_entry.get())

    def SetFileInfoDeconPSF(self, infoStr:str):
        self.SetValueWidgetReadonly( self.deconPsfView.loadPsfInfo_entry, infoStr)

    def SetBeadImage(self,arrayIn):
        """Draw canvas with bead image"""
        cnv = self.deconPsfView.canvasBead
        if cnv: 
            cnv.pack_forget() # remove old canvas
        cnvTmp = CnvPlot.FigureCanvasTkFrom3DArray(arrayIn, self.deconPsfView.deconPSF_plot, plotName="Bead")
        cnvTmp.get_tk_widget().grid(column=0, padx=2, pady=2, row=1)

    def SetPSFImage(self,arrayIn):
        """Draw canvas with result of deconvolution (PSF)"""
        cnv = self.deconPsfView.canvasPSF
        if cnv: 
            cnv.pack_forget() # remove old canvas
        cnvTmp = CnvPlot.FigureCanvasTkFrom3DArray(arrayIn, self.deconPsfView.deconPSF_plot, plotName=" ")
        cnvTmp.get_tk_widget().grid(column=1, padx=2, pady=2, row=1)

    def GetPsfDeconMethod(self):
        return self.deconPsfView._deconMethodsDict[self.deconPsfView.deconMethod.get()]

    def GetPsfDeconProgressbar(self):
        return self.deconPsfView.deconPSF_pgbar
    
# ======= Image deconvolution Widget Functions ===========

    def SetFileInfoImageDeconImage(self, infoStr:str):
        self.deconImageView.imageInfoStr.set(infoStr)
        self.deconImageView.update()
       
    def SetFileInfoPsfDeconImage(self, infoStr:str):
        self.deconImageView.psfInfoStr.set(infoStr)


    def DrawDeconImage(self,arrayIn):
        """Draw input image for deconvolution."""
        CnvPlot.Draw2DArrayOnCanvasPIL(arrayIn, self.deconImageView.image_cnv)

    def DrawResultImage(self,arrayIn):
        """Draw deconvolution result image"""
        CnvPlot.Draw2DArrayOnCanvasPIL(arrayIn, self.deconImageView.result_cnv)

    
    def DrawDeconPsf(self,arrayIn):
        """Draw selected PSF on canvas at image deconvolution frame """
        cnv = self.deconImageView.psf_cnv
        cnvTmp = CnvPlot.FigureCanvasTkFrom3DArray(arrayIn, cnv, " ",150,350)
        cnvTmp.get_tk_widget().grid(column=0, row=0, sticky="n")

    def GetImageDeconMethod(self):
        return self.deconImageView.deconMethod
    
    def GetImageDeconIterationNumber(self):
        return int(self.deconImageView.deconIter_entry.get())

    def GetImageDeconRegularisation(self):
        return int(self.deconImageView.deconReg_entry.get())
    
    def SetImageDeconIterationNumber(self,value):
        try:
            value = int(value)
        except:
            raise ValueError("Wrong iteration number value", "value-not-int")
        if value <= 0:
            raise ValueError("Wrong iteration number value", "value-negative")
        self.SetValueWidgetNormal(self.deconImageView.deconIter_entry, value)

    def SetImageDeconRegularisation(self,value):
        try:
            value = float(value)
        except:
            raise ValueError("Wrong iteration number value", "value-not-float")
        if 0 < value < 1:
            self.SetValueWidgetNormal(self.deconImageView.deconReg_entry, value)
        else:
            raise ValueError("Wrong iteration number value", "not-allowed-value")

    def GetDeconImageProgressbar(self):
        return self.deconImageView.decon_progbar

    def GetImageDeconMethod(self):
        return self.deconImageView._deconMethodsDict[self.deconImageView.deconMethod.get()]
    
    def run(self):
        self.mainwindow.mainloop()



if __name__ == "__main__":
    app = DeconView()
    app.run()
