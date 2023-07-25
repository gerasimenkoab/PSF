import tkinter as tk
from tkinter.messagebox import askokcancel
from tkinter.filedialog import askopenfilenames, askdirectory, asksaveasfilename
from tkinter.simpledialog import askstring
import os
import logging

from model.extractor_model import ExtractorModel
from view.extractor_view import ExtractorView


class ExtractorController:
    """
    Passing actions and data from gui to model and back
    """

    def __init__(self, master=None):
        super().__init__()
        # setup logger
        self.logger = logging.getLogger("__main__." + __name__)
        self.logger.info("Initializing Bead Extractor module.")

        self._master = master
        self.model = ExtractorModel()
        self.view = ExtractorView(self._master)
        self._beadPrevNum = 0

        self.view.SetVoxelValues(self.model.mainImage.voxel)
        self.view.SetBeadSize(self.model.beadDiameter)
        self.view.SetSelectionFrameSize(2 * self.model.selectionFrameHalf)
        # binding buttons and entries events
        self._bind()

    # TODO : remove clicker test
    def buttonClickTest(self):
        print("clicked")

    def _bind(self):
        """binding all events"""
        # buttons:
        self.view.loadBeadsPhoto_btn.config(command=self.LoadsBeadPhoto)
        self.view.undoMark_btn.config(command=self.UndoMark)
        self.view.clearMarks_btn.config(command=self.ClearMarks)
        self.view.extractBeads_btn.config(command=self.ExtractBeads)
        self.view.saveExtractedBeads_btn.config(command=self.SaveExtractedBeads)
        self.view.processBeads_btn.config(command=self.ProcessBeads)
        self.view.saveAverageBead_btn.config(command=self.SaveAverageBead)
        self.view.averageSeveralBeads_btn.config(command=self.AverageSeveralBeads)
        self.view.viewBead2d_btn.config(command=self.ViewBead2D)
        self.view.viewBead3d_btn.config(command=self.ViewBead3D)
        self.view.close_btn.config(command=self.CloseExtractor)
        # entries bind at two events:
        self.view.mainPhotoCanvas.bind("<Button-3>", self.BeadMarkOnClick)
        for key in ("Z", "Y", "X"):
            self.view.voxelSizeEntries[key].bind(
                "<FocusOut>", self.UpdateMainImageVoxelValue
            )
            self.view.voxelSizeEntries[key].bind(
                "<Return>", self.UpdateMainImageVoxelValue
            )
        self.view.beadSizeEntry.bind("<FocusOut>", self.UpdateBeadSizeValue)
        self.view.beadSizeEntry.bind("<Return>", self.UpdateBeadSizeValue)
        self.view.selectSizeEntry.bind("<FocusOut>", self.UpdateSelectionSizeEntry)
        self.view.selectSizeEntry.bind("<Return>", self.UpdateSelectionSizeEntry)
        self.view.beadPrevNum.bind("<FocusOut>", self.SetBeadPrevNum)
        self.view.beadPrevNum.bind("<Return>", self.SetBeadPrevNum)
        self.logger.info("_bind: Binding buttons and entries is done.")

    def GetVoxelDialog(self, text=""):
        """
        Create diealog and return list of values
        """
        voxelStr = askstring("Voxel Dialog", text)
        return [float(a) for a in voxelStr.split(",")]

    def LoadsBeadPhoto(self, event=None):
        """Loading raw beads photo from file"""
        fNames = askopenfilenames( title="Load Beads Photo")
        if fNames is None:
            raise ValueError("No file name recieved", "filename_empty")
        try:
            self.model.SetMainImage(fNames)
        except ValueError as vE:
            if vE.args[1] == "voxel_problem":
                try:
                    tmpList = self.GetVoxelDialog(
                        "Enter voxel size as z,x,y in \u03BCm"
                    )
                    self.model.SetMainImage(fNames, tmpList)
                except ValueError as vE1:
                    raise ValueError(vE1.args[0], vE1.args[1])
            elif vE.args[1] == "data_problem":
                raise ValueError(vE.args[0], vE.args[1])
            else:
                raise ValueError(vE.args[0], vE.args[1])
        self.model.BeadCoordsClear()
        self.model.mainImage.ShowClassInfo()
        self.logger.info("LoadsBeadPhoto: file(s) loaded. ")
        try:
            try:
                os.remove("tmp.tiff")
            except:
                pass
            self.model.mainImage.SaveAsTiff(filename="tmp.tiff")
            self.view.SetMainPhotoImage("tmp.tiff")
            self.view.SetVoxelValues(self.model.mainImage.voxel)
            self.logger.info("LoadsBeadPhoto: tmp.tiff created ")
        except Exception as e:
            self.logger.error("LoadsBeadPhoto: can't create tmp.tiff " + str(e))
            raise IOError("Cant update GUI properly")

    def ClearMarks(self, event=None):
        """Clear bead marks"""
        self.view.BeadMarksClear()
        self.model.BeadCoordsClear()

    def UndoMark(self, event=None):
        """Remove the last bead mark"""
        self.view.BeadMarksRemoveLast()
        self.model.BeadCoordsRemoveLast()

    def ExtractBeads(self, event=None):
        """Extract marked beads as a list"""
        numberExtractedBeads = self.model.MarkedBeadsExtract()
        self.logger.info(
            "ExtractBeads: number of extracted beads = " + str(numberExtractedBeads)
        )

    def SaveExtractedBeads(self, event=None):
        try:
            dirPath = askdirectory()
        except:
            dirPath = ""
        self.model.ExtractedBeadsSave(dirPath)

    def ProcessBeads(self, event=None):
        self.model.BeadsArithmeticMean()
        self.model.BlurAveragedBead(self.view.blurApplyType.get())

    def SaveAverageBead(self, event=None):
        self.model.SaveAverageBead(asksaveasfilename())
        pass

    def AverageSeveralBeads(self, event=None):
        raise NotImplementedError("will do it soon")
        # TODO fix AverageManyBeads
        self.model.AverageManyBeads(
            askopenfilenames(title="Load Beads"), asksaveasfilename()
        )
        pass

    def ViewBead2D(self, event=None):
        try:
            id = int(self.view.beadPrevNum.get())
        except:
            self.beadPrevNum.delete(0, tk.END)
            self.beadPrevNum.insert(0, str(self.view._beadMarksCounter) - 1)
            raise ValueError("Wrong bead index input.")
        self.view.PlotBeadPreview2D(self.model._extractedBeads[id].imArray)

    def ViewBead3D(self, event=None):
        try:
            id = int(self.view.beadPrevNum.get())
        except:
            self.beadPrevNum.delete(0, tk.END)
            self.beadPrevNum.insert(0, str(self.view._beadMarksCounter - 1))
            raise ValueError("Wrong bead index input.")
        self.view.PlotBeadPreview3D(self.model._extractedBeads[id].imArray)

    def CloseExtractor(self, event=None):
        """Closing window and clear tmp files"""
        # Checking existance of self.imgBeadsRaw.close()
        if askokcancel("Close", "Close Bead Extractor Widget?"):
            try:
                self.view.imgBeadsRaw.close()
            except:
                pass
            tmppath = os.getcwd() + "\\tmp.tiff"
            try:
                os.remove(tmppath)
            except:
                pass
            self.view.destroy()
            self.logger.info("Bead Extractor Closed.")

    def BeadMarkOnClick(self, event=None):
        """
        Append mouse event coordinates to global list. Center is adjusted according to max intensity.
        """
        widget = event.widget
        xClick, yClick = widget.canvasx(event.x), widget.canvasy(event.y)
        if self.view.autocorrectSelection.get() == 1:
            print("corrected")
            xr, yr = self.model.LocateFrameMAxIntensity3D(xClick, yClick)
        else:
            print("no correction")
            xr, yr = xClick, yClick
        self.model.beadMarkAdd([xr, yr])
        self.view.beadMarkAdd(widget, xr, yr)

    def UpdateMainImageVoxelValue(self, event = None):
        try:
            newVoxel = [
                float(self.view.voxelSizeEntries["Z"].get()),
                float(self.view.voxelSizeEntries["Y"].get()),
                float(self.view.voxelSizeEntries["X"].get()),
            ]
            self.model.SetVoxelSize(newVoxel)
        except:
            raise ValueError("Can not update voxel values.", "cant_update_voxel")

    def UpdateBeadSizeValue(self, event=None):
        self.model.beadDiameter = float(self.view.beadSizeEntry.get())

    def UpdateSelectionSizeEntry(self, event = None):
        try:
            self.model.selectionFrameHalf = int(self.view.selectSizeEntry.get()) // 2
        except:
            self.logger.debug("Wrong selection size value.")

    def SetBeadPrevNum(self, event=None):
        self._beadPrevNum = int(self.view.beadPrevNum.get())
