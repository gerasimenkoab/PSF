import numpy as np
from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showerror
from PIL import ImageTk, Image, ImageEnhance
from .AuxTkPlot_class import AuxCanvasPlot

"""   TODO:
        - fix  AuxTkPlot_class  for all modules
       - add  bead size to tiff tag
"""


class ExtractorView(tk.Toplevel):
    """Class provides instruments for extraction of beads from microscope multilayer photo."""

    def __init__(self, parent, wwidth=600, wheight=600):
        super().__init__(parent)
        # new  class properties
        self.beadMarks = []  # rectangle pics on the canvas
        self._beadMarksCounter = 0
        self.beadCoords = []
        self.intensityFactor = 1.0  # intensity factor for beads selection widget
        self.beadsPhotoLayerID = 0  # default index of beads microscope photo

        self.voxelFields = "Z", "X", "Y"
        self.voxelSizeEntries = {}

        self.xr = 0
        self.yr = 0
        self.button_name = {
            "LoadBeadsPhoto": "Load Beads Photo",
            "UndoMark": "Undo Mark",
            "ClearAllMark": "Clear All Marks",
            "ExtractSelectedBeads": "Extract Selected Beads",
            "SaveExtractedBeads": "Save Extracted Beads",
            "ProcessExtractedBeads": "Process Extracted Beads",
            "SaveAverageBead": "Save Bead",
            "AverageSeveralBeads": "Average Several Beads",
            "Bead2D": "Show bead in 2D",
            "Bead3D": "Show bead in 3D",
            "Close": "Close Extractor",
        }
        self.button_dict = {}  # according to list of names {id_name : widget}
        self.entry_dict = {}  # according to list of names {id_name : widget}
        self.label_dict = {}  # label {id_name : string}
        # new window widgets
        self.title("Bead extraction")
        self.resizable(False, False)
        ttk.Label(
            self,
            text="Extract Beads Set from the Microscope Image",
            font="Helvetica 14 bold",
        ).grid(row=0, column=0, columnspan=2)

        f0 = Frame(self)
        # making frames to pack several fileds in one grid cell

        # -------------- image and canvas frame --------------------------
        f1 = Frame(f0)
        ttk.Label(
            f1,
            text="1. Load beads photos from the microscope and enter bead size and voxel parameters",
            font="Helvetica 10 bold",
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        f1_1 = Frame(f1)

        self.loadBeadsPhoto_btn = ttk.Button(f1_1, text="Load Beads Photo")
        self.loadBeadsPhoto_btn.pack(side=LEFT, padx=52, pady=2)
        ttk.Button(f1_1, text="-",width=3, command=self.LowerBrightnessToBeadSelectionWidget).pack(
            side=LEFT, padx=2, pady=2
        )
        ttk.Label(f1_1, text="Brightness").pack(side=LEFT, padx=2, pady=2)
        ttk.Button(f1_1, text="+", width=3,command=self.AddBrightnessToBeadSelectionWidget).pack(
            side=LEFT, padx=2, pady=2
        )
        ttk.Label(f1_1, text=" Layer:").pack(side=LEFT, padx=2, pady=2)
        ttk.Button(f1_1, text="+",width=3, command=self.ShowNextLayer).pack(
            side=LEFT, padx=2, pady=2
        )
        self.label_beadsPhotoLayerID = ttk.Label(f1_1, text=str(self.beadsPhotoLayerID))
        self.label_beadsPhotoLayerID.pack(side=LEFT, padx=2, pady=2)
        ttk.Button(f1_1, text="-",width=3, command=self.ShowPrevLayer).pack(
            side=LEFT, padx=2, pady=2
        )
        f1_1.grid(row=1, column=0, columnspan=2)
        frameBeadSize = Frame(f1)
        ttk.Label(frameBeadSize, width=20, text="Actual bead Size:", anchor="w").pack(
            side=LEFT, padx=2, pady=2
        )
        self.beadSizeEntry = ttk.Entry(frameBeadSize, width=5)
        self.beadSizeEntry.pack(side=LEFT, padx=2, pady=2)
        ttk.Label(frameBeadSize, text="\u03BCm ").pack(
            side=LEFT
        )  # mu simbol encoding - \u03BC
        frameBeadSize.grid(row=2, column=1, padx=2, pady=2, sticky="we")
        voxSizeFrame = Frame(f1)
        ttk.Label(voxSizeFrame, text="Voxel size (\u03BCm): ", anchor="w").pack(
            side=LEFT, padx=2, pady=2
        )
        for key in ("Z", "Y", "X"):
            ttk.Label(voxSizeFrame, text=key + "= ").pack(side=LEFT, padx=2, pady=2)
            self.voxelSizeEntries[key] = ttk.Entry(
                voxSizeFrame, width=5            )
            self.voxelSizeEntries[key].pack(side=LEFT, padx=2, pady=2)
        voxSizeFrame.grid(row=2, column=0, sticky="we")
        f1.pack(side=TOP)
        ttk.Separator(f0, orient="horizontal").pack(ipadx=200, pady=10)
        # ---------------------- Mark Beads Frame --------------------

        f2 = Frame(f0)
        ttk.Label(
            f2,
            text="2.Mark beads by right click on the window",
            font="Helvetica 10 bold",
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        selectSizeFrame = Frame(f2)
        ttk.Label(selectSizeFrame, width=14, text="Selection Size: ", anchor="w").pack(
            side=LEFT, padx=2, pady=2
        )
        self.selectSizeEntry = ttk.Entry(selectSizeFrame, width=5)
        self.selectSizeEntry.pack(side=LEFT, padx=2, pady=2)
        ttk.Label(selectSizeFrame, text="px").pack(side=LEFT, padx=2, pady=2)
        selectSizeFrame.grid(row=1, column=0, sticky="we")

        frameMarks = Frame(f2)
        self.autocorrectSelection = IntVar(value=1)
        ttk.Checkbutton(
            frameMarks,
            variable = self.autocorrectSelection,
            text="Auto-correction",
            onvalue=1,
            offvalue=0,
            width=15,
        ).pack(side=LEFT, padx=5, pady=2, fill=BOTH, expand=1)
        self.undoMark_btn = ttk.Button(frameMarks, text="Undo mark")
        self.undoMark_btn.pack(side=LEFT, padx=2, pady=2, fill=BOTH, expand=1)
        self.clearMarks_btn = ttk.Button(frameMarks, text="Clear All Marks")
        self.clearMarks_btn.pack(side=LEFT, padx=2, pady=2, fill=BOTH, expand=1)
        frameMarks.grid(row=1, column=1, sticky="we")

        f2.pack(side=TOP)
        ttk.Separator(f0, orient="horizontal").pack(ipadx=200, pady=10)

        # ------------------- Extract Beads Frame ------------------------
        f3 = Frame(f0)
        ttk.Label(
            f3, text="3. Extract selected beads and save set", font="Helvetica 10 bold"
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        self.extractBeads_btn = ttk.Button(f3, text="Extract Selected Beads")
        self.extractBeads_btn.grid(row=1, column=0, padx=2, pady=2, sticky="we")

        self.saveExtractedBeads_btn = ttk.Button(f3, text="Save Extracted Beads")
        self.saveExtractedBeads_btn.grid(row=1, column=1, padx=2, pady=2, sticky="we")

        self.tiffMenuBitText = ["8 bit", "16 bit", "32 bit"]
        self.tiffMenuBitDict = {
            "8 bit": "uint8",
            "16 bit": "uint16",
            "32 bit": "uint32",
        }
        self.tiffSaveBitType = StringVar()
        self.tiffSaveBitType.set(self.tiffMenuBitText[0])

        frameTiffTypeSelect = Frame(f3)
        ttk.Label(frameTiffTypeSelect, width=10, text="Tiff type ").pack(
            side=LEFT, padx=2, pady=2
        )
        self.tiffType_menu = ttk.OptionMenu(
            frameTiffTypeSelect, self.tiffSaveBitType, *self.tiffMenuBitText
        )
        self.tiffType_menu.pack(side=LEFT, padx=2, pady=2)
        frameTiffTypeSelect.grid(row=1, column=2, padx=2, pady=2, sticky="we")

        f3.pack(side=TOP)
        ttk.Separator(f0, orient="horizontal").pack(ipadx=200, pady=10)

        # --------------- Average Beads Frame --------------------------
        frameAvrageBeads = Frame(f0)
        ttk.Label(
            frameAvrageBeads,
            text="4. Calculate averaged bead with desired blur type and save it.",
            font="Helvetica 10 bold",
        ).pack(side=TOP)

        self.blurMenuTypeText = ["gauss", "none", "median"]
        self.blurApplyType = StringVar()
        self.blurApplyType.set(self.blurMenuTypeText[0])

        frameBlurTypeSelect = Frame(frameAvrageBeads)
        ttk.Label(frameBlurTypeSelect, width=10, text=" Blur type:").pack(
            side=LEFT, padx=2, pady=2
        )
        blurTypeSelect = ttk.Combobox(
            frameBlurTypeSelect,
            textvariable=self.blurApplyType,
            values=self.blurMenuTypeText,
            state="readonly",
        )
        blurTypeSelect.current(0)
        blurTypeSelect.pack(side=LEFT)
        self.doRescaleOverZ = IntVar(value=0)
        ttk.Checkbutton(
            frameBlurTypeSelect,
            variable=self.doRescaleOverZ,
            text=" equal XYZ scale",
            onvalue=1,
            offvalue=0,
        ).pack(side=LEFT, padx=2, pady=2)
        self.precessBeadPrev = IntVar(value=0)
        ttk.Checkbutton(
            frameBlurTypeSelect,
            variable=self.precessBeadPrev,
            text=" preview bead",
            onvalue=1,
            offvalue=0,
        ).pack(side=LEFT, padx=2, pady=2)

        frameBlurTypeSelect.pack(side=TOP, padx=2, pady=2)

        frameAvrageBeadsButtons = Frame(frameAvrageBeads)
        self.processBeads_btn = ttk.Button(
            frameAvrageBeadsButtons,
            text="Process Extracted Beads",
        )
        self.processBeads_btn.pack(side=LEFT, padx=2, pady=2, fill=BOTH, expand=1)
        self.saveAverageBead_btn = ttk.Button(
            frameAvrageBeadsButtons,
            text="Save Average Bead",
        )
        self.saveAverageBead_btn.pack(side=LEFT, padx=2, pady=2, fill=BOTH, expand=1)
        frameAvrageBeadsButtons.pack(side=TOP)
        frameAvrageBeads.pack(side=TOP)  # grid(row =6,column = 0,sticky='we')

        ttk.Separator(f0, orient="horizontal").pack(ipadx=200, pady=10)
        self.averageSeveralBeads_btn = ttk.Button(f0, text="Average Several Beads")
        self.averageSeveralBeads_btn.pack(side=TOP, padx=2, pady=2)

        f0.grid(row=1, column=0, sticky="NSWE")

        self.close_btn = Button(f0, text="Close")
        self.close_btn.pack(side=TOP, padx=2, pady=2)

        f0.grid(row=1, column=0, sticky="NSWE")

        # ---------------- Bead Photo Frame -----------------------------
        canvasFrame = Frame(self)
        self.mainPhotoCanvas = Canvas(
            canvasFrame, width=wwidth, height=wheight, bg="white"
        )
        self.mainPhotoCanvas.grid(row=0, column=0, sticky=(N, E, S, W))
        # main image scrollbars
        self.hScroll = ttk.Scrollbar(canvasFrame, orient="horizontal")
        self.vScroll = ttk.Scrollbar(canvasFrame, orient="vertical")
        self.hScroll.grid(row=1, column=0, columnspan=2, sticky=(E, W))
        self.vScroll.grid(row=0, column=1, sticky=(N, S))
        self.hScroll.config(command=self.mainPhotoCanvas.xview)
        self.mainPhotoCanvas.config(xscrollcommand=self.hScroll.set)
        self.vScroll.config(command=self.mainPhotoCanvas.yview)
        self.mainPhotoCanvas.config(yscrollcommand=self.vScroll.set)
        canvasFrame.grid(row=1, column=1, sticky="WENS")

        # -------------- Bead Preview Frame -----------------------------
        beadPreviewFrame = Frame(self)

        # test bead display canvas. May be removed. if implemented separate window.
        ttk.Label(beadPreviewFrame, text="Bead Preview").pack(side=TOP, padx=2, pady=2)
        self.cnvImg = Canvas(beadPreviewFrame, width=190, height=570, bg="white")
        self.cnvImg.pack(side=TOP, padx=2, pady=2)

        beadPreviewMenuFrame = ttk.Frame(beadPreviewFrame)
        self.beadPrevNum = ttk.Entry(beadPreviewMenuFrame, width=5)
        self.beadPrevNum.pack(side=LEFT)
        self.viewBead2d_btn = ttk.Button(beadPreviewMenuFrame, text="Bead 2D")
        self.viewBead2d_btn.pack(side=LEFT)
        self.viewBead3d_btn = ttk.Button(beadPreviewMenuFrame, text="Bead 3D")
        self.viewBead3d_btn.pack(side=LEFT)
        beadPreviewMenuFrame.pack(side=TOP, padx=2, pady=2)
        beadPreviewFrame.grid(row=1, column=2, sticky="NSWE")

    def UpdateBeadSelectionWidgetImage(self):
        """
        Preparing image for canvas from desired frame with setted parameters.
        """
        # brightness adjust
        enhancer = ImageEnhance.Brightness(self.imgBeadsRaw)
        imgCanvEnhaced = enhancer.enhance(self.intensityFactor)

        self.imgCnv = ImageTk.PhotoImage(imgCanvEnhaced)
        self.mainPhotoCanvas.create_image(0, 0, image=self.imgCnv, anchor=NW)
        # updating scrollers
        self.mainPhotoCanvas.configure(scrollregion=self.mainPhotoCanvas.bbox("all"))
        self.DrawAllMarks()

    def AddBrightnessToBeadSelectionWidget(self):
        """Funcion increase intensity"""
        self.intensityFactor *= 1.1
        self.UpdateBeadSelectionWidgetImage()

    def LowerBrightnessToBeadSelectionWidget(self):
        """Funcion decrease intensity"""
        self.intensityFactor *= 0.9
        self.UpdateBeadSelectionWidgetImage()

    def ShowNextLayer(self):
        """Change visible layer"""
        self.beadsPhotoLayerID += 1
        if self.beadsPhotoLayerID > self.imgBeadsRaw.n_frames - 1:
            self.beadsPhotoLayerID = self.imgBeadsRaw.n_frames - 1
        # updating label on interface
        self.label_beadsPhotoLayerID.config(text=str(self.beadsPhotoLayerID))
        self.imgBeadsRaw.seek(self.beadsPhotoLayerID)
        self.UpdateBeadSelectionWidgetImage()

    def ShowPrevLayer(self):
        """Change visible layer"""
        self.beadsPhotoLayerID += -1
        if self.beadsPhotoLayerID < 0:
            self.beadsPhotoLayerID = 0
        # updating label on interface
        self.label_beadsPhotoLayerID.config(text=str(self.beadsPhotoLayerID))
        self.imgBeadsRaw.seek(self.beadsPhotoLayerID)
        self.UpdateBeadSelectionWidgetImage()

    def SetMainPhotoImage(self, tmpFilePath=None):
        """Loading raw beads photo from file"""
        if tmpFilePath is None:
            raise FileNotFoundError("No file name provided", "no_file_path")
        try:
            try:
                self.imgBeadsRaw.close()
            except:
                pass
            self.imgBeadsRaw = Image.open(tmpFilePath)
        except:
            raise FileNotFoundError(
                "Cant set canvas image with beads photo.", "cant_read_file"
            )
        self.beadsPhotoLayerID = int((self.imgBeadsRaw.n_frames + 1) / 2)
        self.imgBeadsRaw.seek(self.beadsPhotoLayerID)
        self.BeadMarksClear()
        # updating label on interface
        self.label_beadsPhotoLayerID.config(text=str(self.beadsPhotoLayerID))
        # preparing image for canvas from desired frame
        self.imgCnv = ImageTk.PhotoImage(
            image=self.imgBeadsRaw, master=self.mainPhotoCanvas
        )
        # replacing image on the canvas
        self.mainPhotoCanvas.create_image(
            (0, 0), image=self.imgCnv, state="normal", anchor=NW
        )
        # updating scrollers
        self.mainPhotoCanvas.configure(scrollregion=self.mainPhotoCanvas.bbox("all"))

    def SetVoxelValues(self, voxelInDict):
        """Bead voxel size change"""
        if voxelInDict is None:
            raise ValueError("No voxel values recived", "voxel_is_none")
        for axisName in voxelInDict:
            self.voxelSizeEntries[axisName].delete(0, END)
            self.voxelSizeEntries[axisName].insert(0, voxelInDict[axisName])

    def SetBeadSize(self, valueIn):
        """Bead diameter size change"""
        try:
            beadDiameter = abs(float(valueIn))
            self.beadSizeEntry.delete(0, END)
            self.beadSizeEntry.insert(0, str(beadDiameter))
        except:
            showerror("Bead Size: ", "Bad input")
            self.beadSizeEntry.delete(0, END)
            self.beadSizeEntry.insert(0, str(beadDiameter))
            return

    def SetSelectionFrameSize(self, valueIn):
        """Selection Frame size change"""
        try:
            self.selectionFrameHalf = int(abs(float(valueIn)) / 2)
            self.selectSizeEntry.delete(0, END)
            self.selectSizeEntry.insert(0, str(self.selectionFrameHalf * 2))
        except:
            showerror("Selection size: ", "Bad input")
            self.selectSizeEntry.delete(0, END)
            self.selectSizeEntry.insert(0, self.selectionFrameHalf * 2)
            return

    def DrawAllMarks(self):
        """Draw marks for beads on main canvas(cnv1)"""
        cnv = self.mainPhotoCanvas
        for self.xr, self.yr in self.beadCoords:
            self.beadMarks.append(
                cnv.create_rectangle(
                    self.xr - self.selectionFrameHalf,
                    self.yr - self.selectionFrameHalf,
                    self.xr + self.selectionFrameHalf,
                    self.yr + self.selectionFrameHalf,
                    outline="chartreuse1",
                    width=2,
                )
            )

    def beadMarkAdd(self, widget, xr, yr):
        """
        Adding bead mark to canvas after mouse RBclick
        In:
            widget : canvas vidget name
            xr, yr : click event coordinates on widget
        """
        halfSide = int(self.selectSizeEntry.get()) // 2
        self.beadMarks.append(
            widget.create_rectangle(
                xr - halfSide,
                yr - halfSide,
                xr + halfSide,
                yr + halfSide,
                outline="chartreuse1",
                width=2,
            )
        )
        self.beadCoords.append([xr, yr])
        self._beadMarksCounter += 1

    def BeadMarksRemoveLast(self):
        """Removes the last bead in the list"""
        if self.beadMarks == []:
            return
        try:
            self.mainPhotoCanvas.delete(self.beadMarks[-1])
            self.beadMarks.pop()
            self.beadCoords.pop()
        except:
            ValueError("Cant delete mark.")
        self._beadMarksCounter -= 1

    def BeadMarksClear(self):
        """Clears all bead marks"""
        if self.beadMarks == []:
            return
        for sq in self.beadMarks:
            self.mainPhotoCanvas.delete(sq)
        self.beadMarks = []
        self.beadCoords = []
        self._beadMarksCounter = 0

    def PlotBeadPreview2D(self, beadArray):
        """ "Plots three bead in XYZ planes"""
        try:
            self.figIMG_canvas_agg = AuxCanvasPlot.FigureCanvasTkFrom3DArray(
                beadArray, self.cnvImg, plotName=""
            )
            self.figIMG_canvas_agg.get_tk_widget().grid(
                row=1, column=5, rowspan=10, sticky=(N, E, S, W)
            )
        except Exception as e:
            raise RuntimeError("Bead 2D plot failed" + str(e))

    def PlotBeadPreview3D(self, beadArray):
        """ "Plots three bead in 3D pointplot"""
        try:
            self.PlotBead3D(beadArray)
        except Exception as e:
            raise RuntimeError("Bead 3D plot failed" + str(e))

    def PlotBead3D(self, bead, treshold=np.exp(-1) * 255.0):
        """Plot 3D view of a given bead"""
        # popup window creation with canvas and exit button
        child_tmp = tk.Toplevel(self)
        child_tmp.title("3D Bead Preview")
        cnvPlot = tk.Canvas(child_tmp, width=300, height=300)
        cnvPlot.pack(side="top")
        Button(child_tmp, text="Close", command=child_tmp.destroy).pack(side="top")

        AuxCanvasPlot.FigureCanvasTk3DFrom3DArray(bead, cnvPlot).get_tk_widget().pack(
            side="top"
        )


if __name__ == "__main__":
    base1 = ExtractorView(Tk())
    base1.mainloop()
