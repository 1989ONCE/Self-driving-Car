from itertools import pairwise
import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon


class Car():
    def __init__(self, initX, initY, phi, track):
        self.radius = 3             # the radius of the car
        self.currentX = initX       # the current X coordinate of the car
        self.currentY = initY       # the current Y coordinate of the car
        self.currentPHI = phi        # φ: the angle between the car and the horizontal axis
        self.currentTHETA = 0       # θ: the angle that the steering wheel rotates
        
        # Sensor distances
        self.front_distance = 22.0   
        self.left_distance = 8.4853    
        self.right_distance = 8.4853  

        self.track = track

    def update_position(self):
        self.currentX = self.currentX + math.cos(math.radians(self.currentPHI + self.currentTHETA)) + (math.sin(math.radians(self.currentTHETA)) * math.sin(math.radians(self.currentPHI)))
        self.currentY = self.currentY + math.sin(math.radians(self.currentPHI + self.currentTHETA)) - (math.sin(math.radians(self.currentTHETA)) * math.cos(math.radians(self.currentPHI)))
        self.currentPHI = math.degrees(math.radians(self.currentPHI) - math.asin((2 * math.sin(math.radians(self.currentTHETA))) / (2 * self.radius)))
        
        distances = self.calculate_sensors(self.currentX, self.currentY, self.currentPHI)
        self.front_distance = distances[0]
        self.left_distance = distances[1]
        self.right_distance = distances[2]
        # print the car status
        print(f"Car position: ({self.currentX:.4f}, {self.currentY:.4f}), Car angle: {self.currentPHI:.4f}, Steering angle: {self.currentTHETA:.4f}, Front distance: {self.front_distance:.4f}, Left distance: {self.left_distance:.4f}, Right distance: {self.right_distance:.4f}")
    
    def draw_sensor_distance_arrow(self, ax, direction, center_x, center_y, phi, distance):
         # Color for the sensor lines
        if direction == 'Front':
            color = 'red'
        elif direction == 'Right':
            color = 'blue'
        else:
            color = 'green'
    
        # Convert orientation phi from degrees to radians
        phi_radians = math.radians(phi)
        distance = distance - 2
        arrow = ax.arrow(center_x, center_y, distance * math.cos(phi_radians), distance * math.sin(phi_radians), head_width=1, head_length=2, fc=color, ec=color)
        return arrow

    def draw_car(self, ax):
        car = plt.Circle((self.currentX, self.currentY), self.radius, edgecolor='black', facecolor='none')
        center_point = plt.Circle((self.currentX, self.currentY), 0.5, color='darkgrey')
        ax.add_artist(center_point)
        ax.add_artist(car)
        text = ax.text(0.95, 0.05, f'Car Center: ({self.currentX:.2f}, {self.currentY:.2f})', transform=ax.transAxes, fontsize=8, ha='right', va='bottom')
        return car, text, center_point
    
    # 計算感測器到牆壁的距離
    def calculate_sensors(self, x, y, angle):
        angle_rad = np.radians(angle)
        
        # 感測器角度(相對於車子方向)
        angles = [0, 45, -45]  # front, left, right
        distances = []
        
        for sensor_angle in angles:
            # 計算感測器絕對角度
            abs_angle = angle_rad + np.radians(sensor_angle)
            
            # 創建感測器射線
            max_distance = 50
            ray_points = np.array([
                [x, y],
                [x + max_distance * np.cos(abs_angle),
                y + max_distance * np.sin(abs_angle)]
            ])
            
            min_dist = max_distance
            
            # 檢查射線與每個牆段是否相交
            for i in range(len(self.track) - 1):
                wall = np.array([self.track[i], self.track[i + 1]])
                
                # 線段相交計算
                x1, y1 = ray_points[0]
                x2, y2 = ray_points[1]
                x3, y3 = wall[0]
                x4, y4 = wall[1]
                
                denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if denominator == 0:  # 若平行則跳過
                    continue
                    
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
                
                if 0 <= t <= 1 and 0 <= u <= 1:  # 若相交則計算距離
                    intersection_x = x1 + t * (x2 - x1)
                    intersection_y = y1 + t * (y2 - y1)
                    dist = np.sqrt((x - intersection_x)**2 + (y - intersection_y)**2)
                    min_dist = min(min_dist, dist)
            
            distances.append(min_dist)
        
        return distances

    def get_car_status(self):
        return self.currentX, self.currentY, self.currentPHI, self.currentTHETA

    def set_currentTHETA(self, theta):
        self.currentTHETA = theta
    
    def set_car_status(self, x, y, front_d, left_d, right_d, phi):
        self.currentX = x
        self.currentY = y
        self.currentPHI = phi

        self.front_distance = front_d
        self.left_distance = left_d
        self.right_distance = right_d
    
    def get_distances(self):
        return self.front_distance, self.left_distance, self.right_distance