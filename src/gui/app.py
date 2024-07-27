# src/gui/app.py

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ml.data.download import download_data_gui, plot_data, search_ticker
from datetime import datetime
from gui.utils import validate_date, Stock
from ml.pipeline.pipeline import run_pipeline

class StockDataViewer:
    def __init__(self, root):
        self.window = root
        self.window.title("Stock Data Viewer")
        self.window.resizable(False, False)  # Disable window resizing
        
        # Define attributes
        self.ticker = ""
        self.start_date = ""
        self.freq = "daily"
        self.selected_stock_name = ""
        self.current_stock = ""
        self.df = None
        self.stock_obj = None

        # Input variables
        self.ticker_var = tk.StringVar()
        self.start_date_var = tk.StringVar()
        self.freq_var = tk.StringVar(value="daily")

        # Set up trace on variables to update plot on change
        self.ticker_var.trace_add('write', self.update_suggestions)
        self.start_date_var.trace_add('write', self.update_plot)
        self.freq_var.trace_add('write', self.update_plot)

        # Create and place the input widgets
        ttk.Label(self.window, text="Stock Ticker:").pack(pady=5)
        self.ticker_entry = ttk.Entry(self.window, textvariable=self.ticker_var)
        self.ticker_entry.pack(pady=5)

        ttk.Label(self.window, text="Start Date (DD/MM/YYYY):").pack(pady=5)
        self.start_date_entry = ttk.Entry(self.window, textvariable=self.start_date_var)
        self.start_date_entry.pack(pady=5)

        ttk.Label(self.window, text="Frequency:").pack(pady=5)
        ttk.Radiobutton(self.window, text="Daily", variable=self.freq_var, value="daily").pack(pady=5)
        ttk.Radiobutton(self.window, text="Hourly", variable=self.freq_var, value="hourly").pack(pady=5)

        # Create the Listbox for search results
        self.listbox = tk.Listbox(self.window)
        self.listbox.bind('<<ListboxSelect>>', self.on_select)

        # Create a frame for the plot
        self.plot_frame = ttk.Frame(self.window)
        self.plot_frame.pack(pady=20, fill=tk.BOTH, expand=True)

        # Load the initial image and display it
        self.logo_image = Image.open("assets/logo_complete.png")
        self.logo_image = self.logo_image.resize((800, 450), Image.LANCZOS)
        self.logo_photo = ImageTk.PhotoImage(self.logo_image)

        self.logo_label = ttk.Label(self.plot_frame, image=self.logo_photo)
        self.logo_label.image = self.logo_photo  # Keep a reference to avoid garbage collection
        self.logo_label.pack(fill=tk.BOTH, expand=True)

        # Create an initial empty plot
        self.fig, self.ax = plt.subplots(figsize=(8, 4.5))  # Ensure this matches the image dimensions
        self.ax.set_title("Stock Price Data")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)

        # Create a warning label
        self.warning_label = tk.Label(self.window, text="Hourly data is not available for dates older than 2 years from today", fg="red")
        self.warning_label.place_forget()  # Hide the warning label initially

        # Create and place the Run Pipeline button
        self.run_button = ttk.Button(self.window, text="Run Pipeline", command=self.on_run_pipeline)
        self.run_button.pack(pady=10)

        # Center the window on the screen
        self.window.eval('tk::PlaceWindow . center')

    def update_suggestions(self, *args):
        search_term = self.ticker_var.get()
        freq = self.freq_var.get()
        if search_term:
            search_results = search_ticker(search_term, freq)
            self.listbox.delete(0, tk.END)
            for result in search_results:
                self.listbox.insert(tk.END, f"{result['symbol']} - {result['name']}")
            self.listbox.place(x=self.ticker_entry.winfo_x(), y=self.ticker_entry.winfo_y() + self.ticker_entry.winfo_height())
            self.listbox.lift()
        else:
            self.listbox.place_forget()

    def on_select(self, event):
        try:
            selected_item = self.listbox.get(self.listbox.curselection())
            ticker_symbol, stock_name = selected_item.split(' - ', 1)
            self.ticker_var.set(ticker_symbol)
            self.selected_stock_name = stock_name
            self.listbox.place_forget()
            self.update_plot()
        except Exception:
            pass

    def update_plot(self, *args):
        ticker = self.ticker_var.get()
        start_date = self.get_start_date()
        freq = self.freq_var.get()
        stock = self.selected_stock_name

        self.warning_label.place_forget()  # Hide warning label initially

        current_stock_obj = Stock(ticker, stock, start_date, freq)

        if ticker and start_date:
            # Check if hourly frequency and start date is before 730 days
            if freq == "hourly" and (datetime.now() - start_date).days > 730:
                self.warning_label.place(x=200, y=120)  # Show warning label
            else:
                self.df = download_data_gui(ticker, start_date, freq=freq)
                plot_data(self.df, stock, self.ax)
                self.canvas.draw()
                self.stock_obj = current_stock_obj
                self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Show the updated plot
                self.logo_label.pack_forget()  # Hide the logo
        elif not ticker:
            self.canvas.get_tk_widget().pack_forget()  # Hide the plot
            self.logo_label.pack(fill=tk.BOTH, expand=True)  # Show the logo
        elif not current_stock_obj.equals(self.stock_obj):
            print(f"Downloading {stock} data...")
            self.df = download_data_gui(ticker, start_date, freq=freq)
            plot_data(self.df, stock, self.ax)
            self.canvas.draw()
            self.current_stock = stock
            self.stock_obj = current_stock_obj
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Show the updated plot
            self.logo_label.pack_forget()  # Hide the logo

    def get_start_date(self):
        try:
            start_date = self.start_date_var.get()
            if start_date and validate_date(start_date):
                # Convert date format from DD/MM/YYYY to YYYY-MM-DD
                start_date_obj = datetime.strptime(start_date, "%d/%m/%Y")
                return start_date_obj
        except Exception:
            pass
        return None

    def on_run_pipeline(self):
        if self.df is not None and not self.df.empty and self.ticker_var.get() and self.selected_stock_name:
            print(f"Running pipeline for {self.selected_stock_name} ({self.ticker_var.get()})...")
            run_pipeline(self.selected_stock_name, self.ticker_var.get(), self.get_start_date(), self.freq_var.get())
