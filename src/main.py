# src/main.py

import os
from gui.app import StockDataViewer
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    app = StockDataViewer(root)
    root.mainloop()
