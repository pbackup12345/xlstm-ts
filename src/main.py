# src/main.py

import os
from gui.app import StockDataViewer
import tkinter as tk

if __name__ == "__main__":
    os.environ['TIINGO_API_KEY'] = 'e24d4765870dd6588f8c4706b4301dd7001aa4ba'
    root = tk.Tk()
    app = StockDataViewer(root)
    root.mainloop()
