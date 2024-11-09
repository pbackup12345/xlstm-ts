# src/main.py

from gui.app import StockDataViewer
import tkinter as tk

# -------------------------------------------------------------------------------------------
# This is the STARTING POINT of the APPLICATION. 
# RUN this script to LAUNCH the APP and begin its execution.
# -------------------------------------------------------------------------------------------

# Entry point of the application
if __name__ == "__main__":
    root = tk.Tk()
    app = StockDataViewer(root)
    root.mainloop()
