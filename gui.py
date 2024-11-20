import os
import sys
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter.filedialog import asksaveasfilename
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Model.mlp import MLP
from Model.rbfn import RBFN
import math
from car import Car
import time

def str2float(strlist):
    return [round(float(i)) if float(i).is_integer() else float(i) for i in strlist]

class gui():
    def __init__(self, app_name, app_width, app_height):
        self.model = None
        self.path_result = []
        self.data = []
        self.inputs = np.array([])
        self.outputs = np.array([])
        self.ax = None
        self.car_artists = []
        self.path_artists = []

        self.epoch = 0
        self.lr = 0

        # container initialization
        self.container = tk.Tk()
        self.container.config(bg='white', padx=10, pady=10)
        self.container.maxsize(app_width, app_height)
        self.container.title(app_name)
        self.container.geometry(str(app_width) + 'x' + str(app_height))

        # components initialization
        self.graph_frame = tk.Frame(self.container, width=900, height=420, bg='green')
        self.setting_frame = tk.Frame(self.container, width=500, height=900, bg='red')
        self.result_frame = tk.Frame(self.container, width=900, height=320, bg='blue')

        self.track_graph = FigureCanvasTkAgg(master = self.graph_frame)
        self.track_graph.get_tk_widget().config(width=430, height=400)
        self.loss_graph = FigureCanvasTkAgg(master = self.graph_frame)
        self.loss_graph.get_tk_widget().config(width=430, height=400)

        self.dataset_title = tk.Label(self.setting_frame, text='Dataset', bg='white', wraplength=300)

        self.dataDropDown = ttk.Combobox(master = self.setting_frame,
                        values=['train4dAll.txt', 'train6dAll.txt'])
        self.dataDropDown.set('train6dAll.txt')
        self.dataDropDown.bind("<<ComboboxSelected>>",  lambda e: self.load(self.dataDropDown.get()))

        self.model_title = tk.Label(self.setting_frame, text='Model Selection', bg='white', wraplength=300)
        self.modelDropDown = ttk.Combobox(master = self.setting_frame,
                        values=['MLP', 'RBFN'])
        self.modelDropDown.set('MLP')
        self.modelDropDown.bind("<<ComboboxSelected>>",  lambda e: self.model_combobox_selected())
        
        self.epoch_label = tk.Label(self.setting_frame, text='Epoch:', bg='white')
        self.epoch_box = tk.Spinbox(self.setting_frame, increment=1, from_=0, width=5, bg='white', textvariable=tk.StringVar(value='2'))

        self.lrn_rate_label = tk.Label(self.setting_frame, text='Learning Rate:', bg='white')
        self.lrn_rate_box = tk.Spinbox(self.setting_frame,  format="%.2f", increment=0.01, from_=0.0,to=1, width=5, bg='white', textvariable=tk.StringVar(value='0.01'))
        self.train_btn = tk.Button(master = self.setting_frame,  
                     command = self.train, 
                     height = 2,  
                     width = 10, 
                     text = "Train Data",
                     highlightbackground='white') 
        # Add 'Run Car' button in the setting frame
        self.run_car_btn = tk.Button(
            master=self.setting_frame,  
            command=self.run_car, 
            height=2,  
            width=10, 
            text="Run Car",
            highlightbackground='white'
        )
        self.save_btn = tk.Button(master = self.setting_frame,  
                     command = self.save, 
                     height = 2,  
                     width = 16, 
                     text = "Save Track Graph",
                     highlightbackground='white')
        

        # components placing
        self.setting_frame.place(x=5, y=250)
        self.result_frame.place(x=400, y=5)
        self.graph_frame.place(x=400, y=330)
        self.track_graph.get_tk_widget().place(x=10, y=10)
        self.loss_graph.get_tk_widget().place(x=450, y=10)

        self.figure = None
        self.dataset_title.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.dataDropDown.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.model_title.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.modelDropDown.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        self.epoch_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.epoch_box.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        self.lrn_rate_label.grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.lrn_rate_box.grid(row=3, column=1, padx=5, pady=5, sticky='w')
        self.train_btn.grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.save_btn.grid(row=4, column=1, padx=5, pady=5, sticky='w')
        self.run_car_btn.grid(row=5, column=0, padx=5, pady=5, sticky='w')
        self.run_car_btn.config(state='disabled')

        self.load(self.dataDropDown.get()) # Load default data
        self.draw_car_track() # Draw track
        
    def save(self):
        if self.figure == None:
            messagebox.showerror('showerror', 'No Image to Save')
            print('No Image to Save')
            return
        filename = asksaveasfilename(initialfile = 'Untitled.png',defaultextension=".png",filetypes=[("All Files","*.*"),("Portable Graphics Format","*.png")])
        self.figure.savefig(filename)

    def load(self, option):
        try:
            if hasattr(sys, '_MEIPASS'):
                # PyInstaller uses a temporary folder named _MEIPASS to extract files
                trainPath = os.path.join(sys._MEIPASS, 'data/'+ option)
            else:
                # In development mode
                trainPath = os.path.join(os.path.abspath("."), 'data/' + option)
            
            self.read_file(trainPath)
            self.inputs = np.array([item[0] for item in self.data])
            self.outputs = np.array([item[1] for item in self.data])
        except Exception as e:
            print(f'Error loading data: {e}')
            return None
    
    def read_file(self, file):
        data = []
        try: 
            with open(file, "r") as f:
                all_lines = f.readlines() # readlines將每筆資料逐行讀取成list
                for line in all_lines:
                    line = line.strip('\n').split(' ') # strip把line break符號去掉, 然後用空格分割每筆資料
                    # Convert numpy array back to list
                    inputs = str2float(line[:-1]) # append每筆資料到input_data
                    outputs = float(line[-1]) # append每筆資料到target_class
                    data.append([inputs, outputs]) # append input 和 target as tuple
            self.data = data
            return data
        except Exception as e:
            self.data = None
            print(f'Error reading file: {e}')
            return None

    def lrn_validation(self):
        self.lr = float(self.lrn_rate_box.get())
        if self.lr > 1 or self.lr < 0:
            messagebox.showerror('showerror', 'Learning Rate must be between 0 and 1')
            self.lr = 1.00
            self.lrn_rate_box.delete(0, tk.END)
            self.lrn_rate_box.insert(0, str(self.lr))
            return False
        return True
    
    def open(self):
        self.container.mainloop()

    def train(self):
        self.clear_artists()
        if self.data == None:
            messagebox.showerror('showerror', 'No Data to Train')
            print('No Data to Train')
            return

        if self.lrn_validation() == False:
            return
        
        if self.epoch_box.get() == '0':
                messagebox.showerror('showerror', 'Epoch must be greater than 0')
                return
        
        try:
            self.dataDropDown.config(state='disabled')
            self.train_btn.config(state='disabled')
            print('Training...')
            print('Len of Inputs:', len(self.inputs))
            print('dim of Inputs:', len(self.inputs[0]))
            print('Epoch:', self.epoch_box.get())
            print('Learning Rate:', self.lrn_rate_box.get())
            self.epoch = int(self.epoch_box.get())
            
            # Initialize multi-layer perceptron(Backpropagation Network)
            self.model = MLP(len(self.inputs[0]), self.lr)
            
            self.model.train(self.inputs, self.outputs, self.lr, self.epoch)
            print('Training Done')

            # Visualize Results
            self.dataDropDown.config(state='normal')
            self.train_btn.config(state='normal')
            self.save_btn.config(state='normal')
            self.run_car_btn.config(state='normal')

        except Exception as e:
            print(f'Error training perceptron: {e}')
            return None


    def clear_artists(self):
        if len(self.car_artists) > 0:
            for artist in self.car_artists:
                if artist is not None:
                    artist.remove()
            self.car_artists = []
        if len(self.path_artists) > 0:
            for artist in self.path_artists:
                if artist is not None:
                    artist.remove()
            self.path_artists = []


    def init_all_member(self):
        self.file_name = ''
        self.data = None
        self.inputs = np.array([])
        self.outputs = np.array([])

        self.epoch = 0
        self.lr = 0
        self.dim = 0
        self.epoch_result = 1
        self.best_epoch.config(text='')
        self.best_weight.config(text='')
        self.best_acc.config(text='')
        self.train_result = []
        self.test_result = []

        self.figure = None
        self.clear_all_graph()
        self.dataset_title.config(text='Current File: No File Selected')
        self.dim_text.config(text='')
        self.sample_num.config(text='')
        self.epoch_box.delete(0, tk.END)
        self.epoch_box.insert(0, '100')
        self.lrn_rate_box.delete(0, tk.END)
        self.lrn_rate_box.insert(0, '0.1')
        self.train_acc.config(text='...')
        self.test_acc.config(text='...')
        self.weight.config(text='...')
        self.train_sample_num.config(text='')
        self.test_sample_num.config(text='')

    def draw_car_track(self):
        if hasattr(sys, '_MEIPASS'):
            trackFile = os.path.join(sys._MEIPASS, "data/track.txt")
        else:
            trackFile = os.path.join(os.path.abspath("."), "data/track.txt")

        with open(trackFile, 'r') as f:
            lines = f.readlines()
        
        # “起點座標”及“起點與水平線之的夾角”
        start_x, start_y, phi = [float(coord) for coord in lines[0].strip().split(',')]

        # “終點區域左上角座標”及“終點區域右下角座標”
        finish_top_left = [float(coord) for coord in lines[1].strip().split(',')]
        finish_bottom_right = [float(coord) for coord in lines[2].strip().split(',')]
        
        # “賽道邊界”
        boundaries = [[float(coord) for coord in line.strip().split(',')] for line in lines[3:]]
        
        # Extract x and y coordinates from boundaries
        boundary_x, boundary_y = zip(*boundaries)
        self.car = Car(0, 0, phi, boundaries)
        print('boundaries', boundaries)
        self.figure = plt.Figure(figsize=(15, 15), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(-20, 40)
        self.ax.set_ylim(-5, 55)
        self.ax.set_aspect('equal') # 讓xy軸的單位長度相等
        self.ax.set_title("Track")

        # Plot track boundary
        self.ax.plot(boundary_x, boundary_y, 'k-', linewidth=2)

        # Draw start line
        self.ax.plot([-6, 6], [0, 0], 'b-', linewidth=2, label="Start Line")

        # Draw finishing line
        self.ax.plot([18, 30], [37, 37], 'k-', linewidth=2, label="Finishing Line")
        self.ax.plot([18, 30], [40, 40], 'k-', linewidth=2)
        
        
        # Drawing the racecar-contest-like finishing line
        num_squares = 10 # Number of squares each rows
        square_width = (finish_bottom_right[0] - finish_top_left[0]) / num_squares
        square_height = (finish_bottom_right[1] - finish_top_left[1]) / 2
        
        for row in range(2):
            for i in range(num_squares):
                color = 'black' if (i + row) % 2 == 0 else 'white'
                self.ax.add_patch(plt.Rectangle((finish_top_left[0] + i * square_width, finish_top_left[1] + row * square_height),
                        square_width, square_height,
                        edgecolor=color, facecolor=color))

        # Draw starting position and direction arrow
        car, text, path = self.car.draw_car(self.ax)
        self.car_artists.append(car)
        self.car_artists.append(text)
        self.path_artists.append(path)
        self.ax.plot(start_x, start_y, 'ro', label="Start Position")
        self.ax.scatter([], [], color='darkgrey', label='Path')
        self.ax.scatter([], [], marker=r'$\rightarrow$', label=f"Front Sensor", color='red', s=100)
        self.ax.scatter([], [], marker=r'$\rightarrow$', label=f"Left Sensor", color='blue', s=100)
        self.ax.scatter([], [], marker=r'$\rightarrow$', label=f"Right Sensor", color='green', s=100)
        # Set chart properties
        self.ax.set_aspect('equal')
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title("Track")
        self.ax.legend()
        plt.grid(True)

        # Show plot
        self.track_graph.figure = self.figure
        self.track_graph.draw()

    def model_combobox_selected(self):
        if self.modelDropDown.get() == 'MLP':
            self.model = MLP(self.input_size, self.learning_rate, self.hidden_size, self.output_size)
        elif self.modelDropDown.get() == 'RBFN':
            self.model = RBFN()

    def run_car(self):
        
        if len(self.car_artists) > 0:
            for artist in self.car_artists:
                if artist is not None:
                    artist.remove()
            self.car_artists = []
        if len(self.path_artists) > 0:
            for artist in self.path_artists:
                if artist is not None:
                    artist.remove()
            self.path_artists = []
        
        # read result.txt
        if hasattr(sys, '_MEIPASS'):
            path = os.path.join(sys._MEIPASS, "data/result.txt")
        else:
            path = os.path.join(os.path.abspath("."), "data/result.txt")

        with open(path, 'r') as f:
            lines = f.readlines()
            self.car.currentX = 0  # Reset car position to start from (0, 0)
            self.car.currentY = 0
            self.car.currentPHI = 90
            self.train_btn.config(state='disabled')
            try:
                for line in lines:
                    # 移除舊的車子和感測器箭頭圖元
                    if len(self.car_artists) > 0:
                        for artist in self.car_artists:
                            if artist is not None:
                                artist.remove()
                        self.car_artists = []

                    # update car position
                    self.car.currentTHETA = float(line.strip().split(' ')[-1])

                    self.car.update_position()
                    distances = self.car.get_distances()
                    time.sleep(0.01)
                    # 繪製感測器箭頭和車子
                    self.car_artists.append(self.car.draw_sensor_distance_arrow(self.ax, 'Front', self.car.currentX, self.car.currentY, self.car.currentPHI, distances[0]))
                    self.car_artists.append(self.car.draw_sensor_distance_arrow(self.ax, 'Left', self.car.currentX, self.car.currentY, self.car.currentPHI + 45, distances[1]))
                    self.car_artists.append(self.car.draw_sensor_distance_arrow(self.ax, 'Right', self.car.currentX, self.car.currentY, self.car.currentPHI - 45, distances[2]))
                    car, text, center = self.car.draw_car(self.ax)
                    self.car_artists.append(car)
                    self.car_artists.append(text)
                    self.path_artists.append(center)

                    # 更新畫布
                    self.track_graph.get_tk_widget().update()
                    self.track_graph.draw()
                self.train_btn.config(state='normal')
            except Exception as e:
                print(f'Error running car: {e}')
                self.train_btn.config(state='normal')
                return None