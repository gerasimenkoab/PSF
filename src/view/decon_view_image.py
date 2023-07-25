#!/usr/bin/python3
import tkinter as tk
import tkinter.ttk as ttk


class DeconvolveImageFrame(ttk.Frame):
    _deconMethodsDict = {
        "Richardson-Lucy":"RL",
        "Richardson-Lucy TM Reg":"RLTMR",
        "Richardson-Lucy TV Reg":"RLTVR"
        }
    def __init__(self, master=None, **kw):
        super(DeconvolveImageFrame, self).__init__(master, **kw)
        # for i in range(0,4,2):
        #     self.grid_columnconfigure(i, weight=1,uniform=1)
        # self.grid_rowconfigure(0, weight=0,uniform=1)
        # self.grid_rowconfigure(1, weight=1,uniform=1)
# =======Image frame =================
        self.headerImageFrame = ttk.Frame(self)
        self.headerImageFrame.configure(width=200)
        self.imageFrameLbl = ttk.Label(self.headerImageFrame)
        self.imageFrameLbl.configure(font="Helvetica 10 bold", text="Input Image")
        self.imageFrameLbl.pack(side="top")
        self.imageLoad_btn = ttk.Button(self.headerImageFrame)
        self.imageLoad_btn.configure(text="Load Image", width=15)
        self.imageLoad_btn.pack(padx=2, pady=2, side="top")
        self.imageInfoStr = tk.StringVar(value='No Image Loaded')
        self.imageInfo_lbl = ttk.Label(self.headerImageFrame, textvariable=self.imageInfoStr )
        self.imageInfo_lbl.pack(padx=2, pady=2, side="top")
        self.headerImageFrame.grid(column=0, row=0,sticky=tk.N)

        self.imageFrame = ttk.Frame(self)
        self.imageFrame.configure(height=200, width=200)
        self.image_cnv = tk.Canvas(self.imageFrame)
        self.image_cnv.configure(
            background="#c0c0c0",
            height=350,
            insertbackground="#c0c0c0",
            insertborderwidth=0,
            width=350,
        )
        self.image_cnv.grid(column=0,row=0)#pack(expand=True, fill="both", padx=2, pady=2)
        self.imageFrame.grid_columnconfigure(0, weight=1)
        self.imageFrame.grid_rowconfigure(0, weight=1)

        self.imageOptionsFrame = ttk.Frame(self.imageFrame)
        self.imageOptionsFrame.configure(height=200, width=200)
        self.imageLayerSwitch = ttk.Frame(self.imageOptionsFrame)
        self.imageLayerSwitch.configure(height=200, width=200)
        self.imageLayer_lbl = ttk.Label(self.imageLayerSwitch)
        self.imageLayer_lbl.configure(state="normal", text="Z Layer")
        self.imageLayer_lbl.pack(padx=2, pady=2, side="left")
        self.imageLayer_spinbox = ttk.Spinbox(self.imageLayerSwitch)
        self.imageLayer_spinbox.configure( from_=0, to = 100, increment=1, justify="center", width=4 )
        _text_ = "0"
        self.imageLayer_spinbox.delete("0", "end")
        self.imageLayer_spinbox.insert("0", _text_)
        self.imageLayer_spinbox.pack(padx=2, pady=2, side="left")
        self.imageLayerSwitch.pack(expand=True, fill="both", padx=2, pady=2, side="top")
        self.imageOptionsFrame.grid(column=0,row=1)#pack()
        self.imageFrame.grid(column=0, row=1)
        
        separator2 = ttk.Separator(self)
        separator2.configure(orient="vertical")
        separator2.grid(column=1, ipady=200, padx=1, row=0, rowspan=2)

# ========= PSF frame ==============================
        self.headerPsfFrame = ttk.Frame(self)
        self.headerPsfFrame.configure(width=200)
        self.psfFrame_lbl = ttk.Label(self.headerPsfFrame)
        self.psfFrame_lbl.configure(font="Helvetica 10 bold", text='PSF File')
        self.psfFrame_lbl.pack(side="top")
        self.psfLoad_btn = ttk.Button(self.headerPsfFrame)
        self.psfLoad_btn.configure(text='Load PSF File', width=15)
        self.psfLoad_btn.pack(padx=2, pady=2, side="top")
        self.psfInfoStr = tk.StringVar(value='No Image Loaded')
        self.psfInfo_lbl = ttk.Label(self.headerPsfFrame,textvariable=self.psfInfoStr, wraplength=150)
        self.psfInfo_lbl.pack(padx=2, pady=2, side="top")
        self.headerPsfFrame.grid(column=2, row=0,sticky=tk.N)
        self.psfFrame = ttk.Frame(self)
        self.psfFrame.grid_columnconfigure(0, weight=1,uniform=1)
        self.psfFrame.grid_rowconfigure(0, weight=1,uniform=1)
        self.psfFrame.configure(height=200, width=200)
        self.psf_cnv = tk.Canvas(self.psfFrame)
        self.psf_cnv.configure(
            background="#c0c0c0",
            height=350,
            insertbackground="#c0c0c0",
            insertborderwidth=0,
            width=116)
        
        self.psf_cnv.grid(column=0, row=0, sticky="n")
        self.frame18 = ttk.Frame(self.psfFrame)
        self.frame18.configure(height=200, width=200)
        self.frame19 = ttk.Frame(self.frame18)
        self.frame19.configure(height=200, width=200)
        self.label12 = ttk.Label(self.frame19)
        self.label12.configure(state="normal")
        self.label12.pack(expand=True, fill="x", padx=2, pady=2, side="left")
        self.frame19.pack(expand=True, fill="both", padx=2, pady=2, side="top")
        self.frame18.grid(column=0, row=1, sticky="s")
        self.psfFrame.grid(column=2, row=1)


        separator3 = ttk.Separator(self)
        separator3.configure(orient="vertical")
        separator3.grid(column=3, ipady=200, padx=1, row=0, rowspan=2)

# ====== Result frame ==============        
        self.headerResFrame = ttk.Frame(self)
        self.headerResFrame.configure(width=200)
        self.resFrame_lbl = ttk.Label(self.headerResFrame)
        self.resFrame_lbl.configure(font="Helvetica 10 bold", text="Deconvolution")
        self.resFrame_lbl.pack(side="top")
        self.methodFrame = ttk.Frame(self.headerResFrame)
        self.methodFrame.configure(height=200, width=200)
        self.deconMethod_lbl = ttk.Label(self.methodFrame)
        self.deconMethod_lbl.configure(text="Method:", width=9)
        self.deconMethod_lbl.pack(padx=2, side="left")
        self.deconMethod = tk.StringVar()
        self.deconMethod_combobox = ttk.Combobox(
            self.methodFrame,
            textvariable = self.deconMethod,
            values = list( self._deconMethodsDict.keys() ),
            state="readonly",)
        self.deconMethod_combobox.configure(width=20)
        self.deconMethod_combobox.pack(side="left")
        self.decon_progbar = ttk.Progressbar(self.methodFrame)
        self.decon_progbar.configure(length=80, orient="horizontal")
        self.decon_progbar.pack(padx=2, pady=2, side="left")
        self.deconStart_btn = ttk.Button(self.methodFrame)
        self.deconStart_btn.configure(text="Start", width=10)
        self.deconStart_btn.pack(padx=2, pady=2, side="left")
        self.methodFrame.pack(expand=True, fill="x", side="top")
        self.settingsFrame = ttk.Frame(self.headerResFrame)
        self.settingsFrame.configure(height=200, width=200)
        self.deconIter_lbl = ttk.Label(self.settingsFrame)
        self.deconIter_lbl.configure(text="Iteration Number:")
        self.deconIter_lbl.pack(padx=2, side="left")
        self.deconIter_entry = ttk.Entry(self.settingsFrame)
        self.deconIter_entry.configure(width=5)
        _text_ = "1"
        self.deconIter_entry.delete("0", "end")
        self.deconIter_entry.insert("0", _text_)
        self.deconIter_entry.pack(padx=2, side="left")
        self.deconReg_lbl = ttk.Label(self.settingsFrame)
        self.deconReg_lbl.configure(text=" Regularization:")
        self.deconReg_lbl.pack(padx=2, side="left")
        self.deconReg_entry = ttk.Entry(self.settingsFrame)
        self.deconReg_entry.configure(width=8)
        _text_ = "0.000001"
        self.deconReg_entry.delete("0", "end")
        self.deconReg_entry.insert("0", _text_)
        self.deconReg_entry.pack(padx=2, side="left")
        self.resSave_btn = ttk.Button(self.settingsFrame)
        self.resSave_btn.configure(text="Save Result", width=10)
        self.resSave_btn.pack(expand=True, padx=2, pady=2, side="left")
        self.settingsFrame.pack(expand=True, fill="x", side="top")
        self.headerResFrame.grid(column=4, row=0,sticky=tk.N)
        self.resultFrame = ttk.Frame(self)
        self.resultFrame.configure(height=200, width=200)
        self.result_cnv = tk.Canvas(self.resultFrame)
        self.result_cnv.configure(
            background="#c0c0c0",
            height=350,
            insertbackground="#c0c0c0",
            insertborderwidth=2,
            width=350,
        )
        self.result_cnv.grid(column=0,row=0)#pack(expand=True, fill="both", padx=2, pady=2)
        self.resultFrame.grid_columnconfigure(0, weight=1)
        self.resultFrame.grid_rowconfigure(0, weight=1)#.pack(expand=True, fill="both", padx=2, pady=2, side="top")
        self.resultOptionsFrame = ttk.Frame(self.resultFrame)
        self.resultOptionsFrame.configure(height=200, width=200)
        self.resLayerSwitch = ttk.Frame(self.resultOptionsFrame)
        self.resLayerSwitch.configure(height=200, width=200)
        self.resLayer_lbl = ttk.Label(self.resLayerSwitch)
        self.resLayer_lbl.configure(state="normal", text="Z Layer")
        self.resLayer_lbl.pack(padx=2, pady=2, side="left")
        self.resLayer_spinbox = ttk.Spinbox(self.resLayerSwitch)
        self.resLayer_spinbox.configure( from_=0, to = 100, increment=1, justify="center", width=4 )
        _text_ = "0"
        self.resLayer_spinbox.delete("0", "end")
        self.resLayer_spinbox.insert("0", _text_)
        self.resLayer_spinbox.pack(padx=2, pady=2, side="top")
        self.resLayerSwitch.pack(expand=True, fill="both", padx=2, pady=2, side="top")
        self.resultOptionsFrame.grid(column=0,row=1)#.pack(side="top")
        self.resultFrame.grid(column=4, row=1)
        self.configure(height=600, width=800)
        self.grid(column=0, row=0, sticky="n")
        self.grid_anchor("center")
        self.rowconfigure(0, uniform=1)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, pad=1, uniform=1, weight=2)
        self.columnconfigure(2, pad=1, uniform=1, weight=1)
        self.columnconfigure(4, pad=1, uniform=1, weight=2)


if __name__ == "__main__":
    root = tk.Tk()
    widget = DeconvolveImageFrame(root)
    widget.pack(expand=True, fill="both")
    root.mainloop()
