#!/usr/bin/python3
import tkinter as tk
import tkinter.ttk as ttk


class DeconvolvePSFFrame(ttk.Frame):
    _deconMethodsDict = {
        "Richardson-Lucy":"RL",
        "Richardson-Lucy TM Reg":"RLTMR",
        "Richardson-Lucy TV Reg":"RLTVR"
        }
     
    def __init__(self, master=None, **kw):
        super(DeconvolvePSFFrame, self).__init__(master, **kw)
        self.master = master
        self.voxel_entry = {}
        self.deconPSF_settings = ttk.Frame(self)
        self.deconPSF_settings.configure(height=200, width=200)
        self.step1_frm = ttk.Frame(self.deconPSF_settings)
        self.step1_frm.configure(height=200, width=200)
        self.step1_lbl = ttk.Label(self.step1_frm)
        self.step1_lbl.configure(font="Helvetica 10 bold", text="Load Bead Image")
        self.step1_lbl.pack(pady=10, side="top")
        self.step1_load_frm = ttk.Frame(self.step1_frm)
        self.step1_load_frm.configure(height=200, width=200)
        self.loadPSF_btn = ttk.Button(self.step1_load_frm)
        self.loadPSF_btn.configure(text="Load Image")
        self.loadPSF_btn.pack(padx=5, side="left")
        self.loadPsfInfo_entry = ttk.Entry(self.step1_load_frm)
        self.loadPsfInfo_entry.configure(state="readonly")
        _text_ = "No Image Loaded"
        self.loadPsfInfo_entry["state"] = "normal"
        self.loadPsfInfo_entry.delete("0", "end")
        self.loadPsfInfo_entry.insert("0", _text_)
        self.loadPsfInfo_entry["state"] = "readonly"
        self.loadPsfInfo_entry.pack(expand=True, fill="x", padx=5, pady=5, side="right")
        self.step1_load_frm.pack(expand=True, fill="both", padx=5, side="top")
        self.step1_sep = ttk.Separator(self.step1_frm)
        self.step1_sep.configure(orient="horizontal")
        self.step1_sep.pack(expand=True, fill="x", padx=10, pady=10, side="top")
        self.step1_frm.pack(expand=True, fill="both", side="top")
        self.step2 = ttk.Frame(self.deconPSF_settings)
        self.step2.configure(height=200, width=200)
        self.step2_lbl = ttk.Label(self.step2)
        self.step2_lbl.configure(font="Helvetica 10 bold", text="Bead Parameters")
        self.step2_lbl.pack(side="top")
        self.beadParamFrame = ttk.Frame(self.step2)
        self.beadParamFrame.configure(height=200, width=200)
        self.psfBeadSizeFrame = ttk.Frame(self.beadParamFrame)
        self.psfBeadSizeFrame.configure(height=200, width=200)
        self.beadSize_lbl = ttk.Label(self.psfBeadSizeFrame)
        self.beadSize_lbl.configure(text="Bead Size (nm):")
        self.beadSize_lbl.pack(padx=5, pady=5, side="left")
        self.beadSize_entry = ttk.Entry(self.psfBeadSizeFrame)
        self.beadSize_entry.configure(validate="focusout", width=10)
        _text_ = "0.1"
        self.beadSize_entry.delete("0", "end")
        self.beadSize_entry.insert("0", _text_)
        self.beadSize_entry.pack(padx=5, pady=5, side="left")
        self.psfBeadSizeFrame.pack(side="top")
        self.psfVoxelSizeFrame = ttk.Frame(self.beadParamFrame)
        self.psfVoxelSizeFrame.configure(height=200, width=200)
        label8 = ttk.Label(self.psfVoxelSizeFrame)
        label8.configure(text="Voxel")
        label8.pack(padx=2, pady=2, side="left")

        for key in ["X", "Y", "Z"]:
            label11 = ttk.Label(self.psfVoxelSizeFrame)
            label11.configure(text=key + ":")
            label11.pack(padx=2, pady=2, side="left")
            self.voxel_entry[key] = ttk.Entry(self.psfVoxelSizeFrame)
            self.voxel_entry[key].name = key
            self.voxel_entry[key].configure(
                state="normal", validate="focusout", width=5
            )
            _text_ = "0.1"
            self.voxel_entry[key].delete("0", "end")
            self.voxel_entry[key].insert("0", _text_)
            self.voxel_entry[key].pack(padx=5, pady=5, side="left")
        self.psfVoxelSizeFrame.pack(side="top")
        self.beadParamFrame.pack(expand=True, fill="both", side="top")
        separator3 = ttk.Separator(self.step2)
        separator3.configure(orient="horizontal")
        separator3.pack(fill="x", padx=10, pady=10, side="top")
        self.step2.pack(expand=True, fill="both", padx=5, pady=5, side="top")
        self.psfDeconParamFrame = ttk.Frame(self.deconPSF_settings)
        self.psfDeconParamFrame.configure(height=200, width=200)
        self.step3_lbl = ttk.Label(self.psfDeconParamFrame)
        self.step3_lbl.configure(
            font="Helvetica 10 bold", text="Deconvolution parameters"
        )
        self.step3_lbl.pack(side="top")
        self.methodFrame = ttk.Frame(self.psfDeconParamFrame)
        self.methodFrame.configure(height=200, width=200)
        label17 = ttk.Label(self.methodFrame)
        label17.configure(text="Method:")
        label17.pack(padx=2, pady=5, side="left")
        self.deconMethod = tk.StringVar()
        self.deconMethod_combobox = ttk.Combobox(
            self.methodFrame,
            textvariable = self.deconMethod,
            values = list( self._deconMethodsDict.keys() ),
            state="readonly",)
        self.deconMethod_combobox.pack(padx=2, pady=5, side="left")
        self.methodFrame.pack(side="top")
        self.psfIterNumFrame = ttk.Frame(self.psfDeconParamFrame)
        self.psfIterNumFrame.configure(height=200, width=200)
        label14 = ttk.Label(self.psfIterNumFrame)
        label14.configure(text="Iteration number:")
        label14.grid(column=0, padx=2, pady=2, row=0, sticky="e")
        self.psfIterNum_entry = ttk.Entry(self.psfIterNumFrame)
        self.psfIterNum_entry.configure(width=10)
        self.psfIterNum_entry.grid(column=1, padx=2, pady=2, row=0, sticky="w")
        self.psfIterNumFrame.pack(side="top")
        self.psfIterNumFrame.grid_anchor("center")
        self.psfIterNumFrame.rowconfigure(0, uniform=0)
        self.psfIterNumFrame.columnconfigure(0, uniform=1, weight=1)
        self.psfIterNumFrame.columnconfigure(1, uniform=1, weight=1)
        frame25 = ttk.Frame(self.psfDeconParamFrame)
        frame25.configure(height=200, width=200)
        label19 = ttk.Label(frame25)
        label19.configure(text="Regularization:")
        label19.grid(column=0, padx=2, pady=2, row=0, sticky="e")
        self.psfReg_entry = ttk.Entry(frame25)
        self.psfReg_entry.configure(width=10)
        self.psfReg_entry.grid(column=1, padx=2, pady=2, row=0, sticky="w")
        frame25.pack(pady=2, side="top")
        frame25.rowconfigure(0, uniform=0)
        frame25.columnconfigure(0, uniform=1)
        frame25.columnconfigure(1, uniform=1)
        self.psfDeconBtmFrame = ttk.Frame(self.psfDeconParamFrame)
        self.psfDeconBtmFrame.configure(height=200, width=200)
        self.deconPSF_pgbar = ttk.Progressbar(self.psfDeconBtmFrame)
        self.deconPSF_pgbar.configure(orient="horizontal")
        self.deconPSF_pgbar.grid(column=0, padx=2, pady=2, row=0, sticky="w")
        self.calcPSF_btn = ttk.Button(self.psfDeconBtmFrame)
        self.calcPSF_btn.configure(text="Calculate PSF")
        self.calcPSF_btn.grid(column=1, padx=2, pady=2, row=0)
        self.psfDeconBtmFrame.pack(side="top")
        self.psfDeconBtmFrame.columnconfigure(0, uniform=1)
        self.psfDeconBtmFrame.columnconfigure(1, uniform=1)
        self.psfDeconRegFrame = ttk.Frame(self.psfDeconParamFrame)
        self.psfDeconRegFrame.configure(height=200, width=200)
        self.savePsf_btn = ttk.Button(self.psfDeconRegFrame)
        self.savePsf_btn.configure(text="Save PSF")
        self.savePsf_btn.pack(padx=2, pady=5, side="left")
        self.psfDeconRegFrame.pack(side="top")
        self.psfDeconParamFrame.pack(expand=True, fill="both", side="top")
        self.deconPSF_settings.pack(expand=True, fill="both", side="left")
        self.deconPSF_plot = ttk.Frame(self)
        self.deconPSF_plot.configure(height=200, width=600)
        self.canvasBead = tk.Canvas(self.deconPSF_plot)
        self.canvasBead.configure( width=200)
        self.canvasBead.grid(column=0, padx=2, pady=2, row=1)
        self.canvasPSF = tk.Canvas(self.deconPSF_plot)
        self.canvasPSF.configure( width=200)
        self.canvasPSF.grid(column=1, padx=2, pady=2, row=1)
        self.deconPSF_plot.pack(expand=True, fill="both", side="left")
        self.deconPSF_plot.grid_anchor("center")
        self.deconPSF_plot.rowconfigure(0, uniform=0)
        self.deconPSF_plot.columnconfigure(0, uniform=1, weight=1)
        self.deconPSF_plot.columnconfigure(1, uniform=1)
        self.configure(height=200, width=200)
        self.pack(expand=True, fill="both", side="top")


if __name__ == "__main__":
    root = tk.Tk()
    widget = DeconvolvePSFFrame(root)
    widget.pack(expand=True, fill="both")
    root.mainloop()
