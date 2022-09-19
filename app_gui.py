from tkinter import *

from cnn_deconvolution_gui import *
from app_gui import *
from psf_extractor_gui import PSFExtractorGUI

"""
TODO:
- Close all TODO in cnn_deconvloution_gui.py
- Close all TODO in psf_extractor_gui.py

- Add button "About app" with authors
"""

class MainWindowGUI(Tk):

    def __init__(self, master = None, wwidth=800, wheight = 2000):
        super().__init__()
        self.imgBeadRawLoad = FALSE

        self.title("Simple experimental PSF extractor")
        self.resizable(False,False)
        Label(self, text="").grid(row = 0, column = 0)         # blanc insert
        Label(self, text="").grid(row = 2, column = 0)         # blanc insert
        Button(text = 'Launch CNN deconvolution', command = self.LaunchCNNDeconvolution).grid(row=2,column=5)
        Label(self, text="").grid(row = 2, column = 2)         # blanc insert
        Button(text = 'Launch PSF extractor', command = self.LaunchPSFExtractor).grid(row=2,column=3)
        Label(self, text="").grid(row = 2, column = 4)         # blanc insert
        Button(text = 'Launch bead extractor', command = self.LaunchBeadExtractor).grid(row=2,column=1)
        Label(self, text="").grid(row = 2, column = 6)         # blanc insert

        Label(self, text = "").grid(row = 3,column = 1)        # blanc insert

    def LaunchCNNDeconvolution(self):
        deconvolver = CNNDeconvGUI(self)
        deconvolver.grab_set()
        return

    def LaunchPSFExtractor(self):
        extractor = PSFExtractorGUI(self)
        extractor.grab_set()
        return

    def LaunchBeadExtractor(self):
        return

if __name__ == '__main__':
      rootWin = MainWindowGUI()
      rootWin.mainloop()

