from model.decon_psf_model import DeconPsfModel
from model.decon_image_model import DeconImageModel
from view.decon_view import DeconView
import logging

# TODO: log string output
class DeconController:
    def __init__(self, master) -> None:
        super().__init__()
        # setup logger
        self.logger = logging.getLogger("__main__." + __name__)

        self._master = master

        self.modelDeconPSF = DeconPsfModel()
        self.modelDeconImage = DeconImageModel()
        self.viewDecon = DeconView(self._master)

        self.viewDecon.SetVoxelValues(self.modelDeconPSF.PSFImage.voxel)
        self.viewDecon.SetBeadSize(self.modelDeconPSF.beadDiameter)
        # binding buttons and entries events
        self._bindDeconPSF()
        self._bindDeconImage()
        self.logger.debug("PSF and Image deconvolution controller initialized")

    def _bindDeconPSF(self):
        """
        Binding events for PSF deconvolution 
        """
        self.viewDecon.deconPsfView.loadPSF_btn.bind("<1>", self.LoadBead_btn_click, add="")
        self.viewDecon.deconPsfView.beadSize_entry.bind("<Return>", self.UpdateBeadSizeValue, add="")
        self.viewDecon.deconPsfView.beadSize_entry.bind("<FocusOut>", self.UpdateBeadSizeValue, add="")
        for key in ["X","Y","Z"]:
            self.viewDecon.deconPsfView.voxel_entry[key].bind("<Return>", self.UpdateBeadVoxelValues, add="")
            self.viewDecon.deconPsfView.voxel_entry[key].bind("<FocusOut>", self.UpdateBeadVoxelValues, add="")
        self.viewDecon.deconPsfView.psfIterNum_entry.bind("<Return>", self.UpdatePsfIterlValue, add="")
        self.viewDecon.deconPsfView.psfIterNum_entry.bind("<FocusOut>", self.UpdatePsfIterlValue, add="")
        self.viewDecon.deconPsfView.psfReg_entry.bind("<Return>", self.UpdatePsfReglValue, add="")
        self.viewDecon.deconPsfView.psfReg_entry.bind("<FocusOut>", self.UpdatePsfReglValue, add="")
        self.viewDecon.deconPsfView.calcPSF_btn.bind("<1>", self.CalcPSF_btn_click, add="")
        self.viewDecon.deconPsfView.savePsf_btn.bind("<1>", self.SavePSF_btn_click, add="")

    def _bindDeconImage(self):
        """
        Binding events for Image deconvolution 
        """
        self.viewDecon.deconImageView.imageLoad_btn.bind("<1>", self.DeconLoadImage_clb, add="")
        self.viewDecon.deconImageView.psfLoad_btn.bind("<1>", self.DeconPSFLoad_clb, add="")

        self.viewDecon.deconImageView.deconIter_entry.bind("<Return>", self.UpdateImageIterlValue, add="")
        self.viewDecon.deconImageView.deconIter_entry.bind("<FocusOut>", self.UpdateImageIterlValue, add="")
        self.viewDecon.deconImageView.deconReg_entry.bind("<Return>", self.UpdateImageReglValue, add="")
        self.viewDecon.deconImageView.deconReg_entry.bind("<FocusOut>", self.UpdateImageReglValue, add="")

        self.viewDecon.deconImageView.imageLayer_spinbox.bind("<<Decrement>>", self.ImageLayerChange_clb, add="")
        self.viewDecon.deconImageView.imageLayer_spinbox.bind("<<Increment>>", self.ImageLayerChange_clb, add="")
        self.viewDecon.deconImageView.imageLayer_spinbox.bind("<Return>", self.ImageLayerChange_clb, add="")
        self.viewDecon.deconImageView.deconStart_btn.bind("<1>", self.DeconStart_clb, add="")
        self.viewDecon.deconImageView.resSave_btn.bind("<1>", self.SaveDeconImage_clb, add="")
        self.viewDecon.deconImageView.resLayer_spinbox.bind("<<Decrement>>", self.ResLayerChange_clb, add="")
        self.viewDecon.deconImageView.resLayer_spinbox.bind("<<Increment>>", self.ResLayerChange_clb, add="")
        self.viewDecon.deconImageView.imageLayer_spinbox.bind("<Return>", self.ResLayerChange_clb, add="")

    # ======= Decon PSF Callbacks ===============
    def LoadBead_btn_click(self,event):
        """Loading bead photo from file"""
        eventWgt = event.widget
        try:
            fNames = self.viewDecon.GetFileNamesList(eventWgt,"Load Bead Photo")
        except:
            return
        try:
            self.modelDeconPSF.SetPSFImage(fNames)
        except ValueError as vE:
            if vE.args[1] == "voxel_problem":
                try:
                    voxText = "Enter voxel size as z,x,y in \u03BCm"
                    self.modelDeconPSF.SetPSFImage(fNames, self.viewDecon.GetVoxelDialog(eventWgt, voxText) )
                except ValueError as vE1:
                    raise ValueError(vE1.args[0], vE1.args[1])
            elif vE.args[1] == "data_problem":
                raise ValueError(vE.args[0], vE.args[1])
            else:
                raise ValueError(vE.args[0], vE.args[1])
        # self.modelDeconPSF.PSFImage.ShowClassInfo()
        self.viewDecon.SetFileInfoDeconPSF(self.modelDeconPSF.PSFImage.GetImageInfoStr(output = "full") )
        self.viewDecon.SetBeadImage(self.modelDeconPSF.PSFImage.imArray)
        self.viewDecon.SetVoxelValues(self.modelDeconPSF.PSFImage.voxel)
        self.logger.info("Bead File Loaded: " + fNames[0])
        
    def UpdateBeadSizeValue(self, event=None):
        eventWgt = event.widget
        try:
            newValue = float( eventWgt.get() )
        except:
            self.viewDecon.SetValueWidgetNormal( eventWgt, self.modelDeconPSF.beadDiameter )
            return
        try:
            self.modelDeconPSF.beadDiameter = newValue
        except:
            self.viewDecon.SetValueWidgetNormal( eventWgt, self.modelDeconPSF.beadDiameter )
            return
        self.viewDecon.SetValueWidgetNormal( eventWgt, newValue )

    def UpdateBeadVoxelValues(self, event=None):
        eventWgt = event.widget
        axisName = eventWgt.name
        try:
            newValue = float( eventWgt.get() )
        except:
            self.viewDecon.SetValueWidgetNormal( eventWgt, self.modelDeconPSF.voxel[axisName] )
            return
        try:
            self.modelDeconPSF.SetVoxelByAxis(axisName, newValue)
        except Exception as e:
            self.viewDecon.SetValueWidgetNormal( eventWgt, self.modelDeconPSF.voxel[axisName] )
            return
        self.viewDecon.SetValueWidgetNormal( eventWgt, newValue )

        pass

    def UpdatePsfIterlValue(self, event=None):
        eventWgt = event.widget
        try:
            newValue = int( eventWgt.get() )
        except:
            self.viewDecon.SetValueWidgetNormal( eventWgt, self.modelDeconPSF.iterationNumber )
            return
        try: 
            self.modelDeconPSF.iterationNumber = newValue
        except:
            self.viewDecon.SetValueWidgetNormal( eventWgt, self.modelDeconPSF.iterationNumber )
            return
        self.viewDecon.SetValueWidgetNormal( eventWgt, newValue )

    def UpdatePsfReglValue(self, event=None):
        eventWgt = event.widget
        try:
            newValue = float( eventWgt.get() )
        except:
            self.viewDecon.SetValueWidgetNormal( eventWgt, self.modelDeconPSF.regularizationParameter )
            return
        try: 
            self.modelDeconPSF.regularizationParameter = newValue
        except:
            self.viewDecon.SetValueWidgetNormal( eventWgt, self.modelDeconPSF.regularizationParameter )
            return
        self.viewDecon.SetValueWidgetNormal( eventWgt, newValue )

    def CalcPSF_btn_click(self, event=None):
        progBar = self.viewDecon.GetPsfDeconProgressbar()
        method = self.viewDecon.GetPsfDeconMethod()
        self.logger.info("Starting bead deconvolution. Method code: " + method)
        try:
            self.modelDeconPSF.CalculatePSF( method, progBar, self.viewDecon.deconViewToplevel )
        except Exception as e:
            self.logger.info("Bead deconvolution failed with exception: " + str(e))
            return
        try:
            self.viewDecon.SetPSFImage( self.modelDeconPSF.resultImage.imArray )
        except:
            return
        self.logger.info("Bead image deconvolution finished.")


    def SavePSF_btn_click(self, event=None):

        if self.modelDeconPSF.resultImage == None:
            self.logger.info("Can not save. PSF was not created.")
            return
        try:
            fName = self.viewDecon.GetFileName(event.widget,"Select file name")
        except:
            return
        try:
            self.modelDeconPSF.resultImage.SaveAsTiff(fName)
        except ValueError as vE:
           self.logger.debug(str(vE[0]))
           return
        self.logger.info("PSF was saved as " + fName)

    # ======== Decon Image Callbacks ===============================

    def DeconLoadImage_clb(self, event=None):
        """Loading image for deconvolution from file"""
        eventWgt = event.widget
        try:
            fNames = self.viewDecon.GetFileNamesList(eventWgt,"Load Image")
        except:
            return
        try:
            self.modelDeconImage.SetDeconImage(fNames)
        except ValueError as vE:
            if vE.args[1] == "voxel_problem":
                try:
                    voxText = "Enter voxel size as z,x,y in \u03BCm"
                    try:
                        voxelRead =  self.viewDecon.GetVoxelDialog(eventWgt, voxText) 
                    except:
                        self.logger.info("Bad voxel info in dialog.")
                        return
                    self.modelDeconImage.SetDeconImage(fNames, voxelRead)
                except ValueError as vE1:
                    self.logger.info("Image load failed.")
                    return
            elif vE.args[1] == "data_problem":
                self.logger.info("Image load failed. Bad intesity array. ")
                return
            else:
                self.logger.info("Image load failed. Unknown problem.")
                return
        self.viewDecon.SetFileInfoImageDeconImage(self.modelDeconImage.deconImage.GetImageInfoStr(output = "full") )
        upLim = self.modelDeconImage.deconImage.imArray.shape[0]-1
        self.viewDecon.deconImageView.imageLayer_spinbox.configure( from_=0, to = upLim )
        layerId = int(self.viewDecon.deconImageView.imageLayer_spinbox.get())
        self.viewDecon.DrawDeconImage(self.modelDeconImage.deconImage.imArray[layerId,:,:])

        self.logger.info("Bead File Loaded: " + fNames[0])



    def DeconPSFLoad_clb(self, event=None):
        """Loading PSF for deconvolution from file"""
        eventWgt = event.widget
        try:
            fNames = self.viewDecon.GetFileNamesList(eventWgt,"Load PSF")
        except:
            return
        try:
            self.modelDeconImage.SetDeconPsf(fNames)
        except ValueError as vE:
            if vE.args[1] == "voxel_problem":
                try:
                    voxText = "Enter voxel size as z,x,y in \u03BCm"
                    self.modelDeconImage.SetDeconPsf(fNames, self.viewDecon.GetVoxelDialog(eventWgt, voxText) )
                except ValueError as vE1:
                    raise ValueError(vE1.args[0], vE1.args[1])
            elif vE.args[1] == "data_problem":
                raise ValueError(vE.args[0], vE.args[1])
            else:
                raise ValueError(vE.args[0], vE.args[1])
        self.viewDecon.SetFileInfoPsfDeconImage(self.modelDeconImage.deconPsf.GetImageInfoStr(output = "full") )
        self.viewDecon.DrawDeconPsf(self.modelDeconImage.deconPsf.imArray)
        self.logger.info("PSF File Loaded: " + fNames[0]) 


    def ImageLayerChange_clb(self,event = None):
        wgt = event.widget
        try:
            spValue = int(wgt.get())
            arr = self.modelDeconImage.deconImage.imArray[spValue,:,:]
        except:
            wgt.set("0")
            return
        self.viewDecon.DrawDeconImage(arr)       

    def ResLayerChange_clb(self,event = None):
        wgt = event.widget
        try:
            spValue = int(wgt.get())
            arr = self.modelDeconImage.deconResult.imArray[spValue,:,:]
        except:
            wgt.set("0")
            return
        self.viewDecon.DrawResultImage(arr)       

    def UpdateImageIterlValue(self, event=None):
        eventWgt = event.widget
        try:
            newValue = int( eventWgt.get() )
        except:
            self.viewDecon.SetValueWidgetNormal( eventWgt, self.modelDeconImage.iterationNumber )
            return
        try: 
            self.modelDeconImage.iterationNumber = newValue
        except:
            self.viewDecon.SetValueWidgetNormal( eventWgt, self.modelDeconImage.iterationNumber )
            return
        self.viewDecon.SetValueWidgetNormal( eventWgt, newValue )

    def UpdateImageReglValue(self, event=None):
        eventWgt = event.widget
        try:
            newValue = float( eventWgt.get() )
        except:
            self.viewDecon.SetValueWidgetNormal( eventWgt, self.modelDeconImage.regularizationParameter )
            return
        try: 
            self.modelDeconImage.regularizationParameter = newValue
        except:
            self.viewDecon.SetValueWidgetNormal( eventWgt, self.modelDeconImage.regularizationParameter )
            return
        self.viewDecon.SetValueWidgetNormal( eventWgt, newValue )




    def DeconStart_clb(self, event=None):
        progBar = self.viewDecon.GetDeconImageProgressbar()
        method = self.viewDecon.GetImageDeconMethod()
        decon_wgt = self.modelDeconImage
        self.logger.info("Starting image deconvolution. Method code: " + method)
        try:
            self.modelDeconImage.DeconvolveImage( method, progBar, self.viewDecon.deconViewToplevel )
            # self.modelDeconImage.deconResult = self.modelDeconImage.deconImage
        except:
            self.logger.info("Image deconvolution failed.")
            return


        try:
            self.viewDecon.deconImageView.resLayer_spinbox.configure( from_=0, to = decon_wgt.deconResult.imArray.shape[0]-1 )
            layerId = int(self.viewDecon.deconImageView.resLayer_spinbox.get())
            self.viewDecon.DrawResultImage(decon_wgt.deconResult.imArray[layerId,:,:])
        except Exception as e:
            return
        self.logger.info("Image deconvolution finished.")
        
    def SaveDeconImage_clb(self, event=None):
        if self.modelDeconImage.deconResult == None:
            self.logger.info("File not saved. Deconvolution result image was not created.")
            return
        try:
            fName = self.viewDecon.GetFileName(event.widget,"Select file name")
        except:
            return
        try:
            self.modelDeconImage.deconResult.SaveAsTiff(fName)
        except ValueError as vE:
           self.logger.debug(str(vE[0]))
           return
        self.logger.info("Result image was saved as " + fName)

