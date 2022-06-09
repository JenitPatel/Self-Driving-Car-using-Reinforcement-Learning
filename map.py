# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing the DeepQNetwork class in ai.py
from ai import DeepQNetwork

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing a_last and b_last, used to keep the last point in memory when we draw the marker on the map
a_last = 0
b_last = 0
n_events = 0
range = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = DeepQNetwork(5, 3, 0.9)
rotationmovement = [0, 20, -20]
last_prize = 0
performance_values = []

# Initializing the map
initial_change = True


def init():
    global marker
    global target_x
    global target_y
    global initial_change
    marker = np.zeros((longitude, latitude))
    target_x = 20
    target_y = latitude - 20
    initial_change = False


# Initializing the last distance
last_distance = 0


# Creating the car class

class Car(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos
        self.signal1 = int(np.sum(marker[int(self.sensor1_x) - 10:int(self.sensor1_x) + 10,
                                  int(self.sensor1_y) - 10:int(self.sensor1_y) + 10])) / 400.
        self.signal2 = int(np.sum(marker[int(self.sensor2_x) - 10:int(self.sensor2_x) + 10,
                                  int(self.sensor2_y) - 10:int(self.sensor2_y) + 10])) / 400.
        self.signal3 = int(np.sum(marker[int(self.sensor3_x) - 10:int(self.sensor3_x) + 10,
                                  int(self.sensor3_y) - 10:int(self.sensor3_y) + 10])) / 400.
        if self.sensor1_x > longitude - 10 or self.sensor1_x < 10 or self.sensor1_y > latitude - 10 or self.sensor1_y < 10:
            self.signal1 = 1.
        if self.sensor2_x > longitude - 10 or self.sensor2_x < 10 or self.sensor2_y > latitude - 10 or self.sensor2_y < 10:
            self.signal2 = 1.
        if self.sensor3_x > longitude - 10 or self.sensor3_x < 10 or self.sensor3_y > latitude - 10 or self.sensor3_y < 10:
            self.signal3 = 1.


class Ball1(Widget):
    pass


class Ball2(Widget):
    pass


class Ball3(Widget):
    pass


# Creating the game class

class Game(Widget):
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        global brain
        global last_prize
        global performance_values
        global last_distance
        global target_x
        global target_y
        global longitude
        global latitude

        longitude = self.width
        latitude = self.height
        if initial_change:
            init()

        xx = target_x - self.car.x
        yy = target_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        action = brain.update(last_prize, last_signal)
        performance_values.append(brain.score())
        rotation = rotationmovement[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - target_x) ** 2 + (self.car.y - target_y) ** 2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if marker[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_prize = -1
        else:  # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_prize = -0.2
            if distance < last_distance:
                last_prize = 0.1

        if self.car.x < 10:
            self.car.x = 10
            last_prize = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_prize = -1
        if self.car.y < 10:
            self.car.y = 10
            last_prize = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_prize = -1

        if distance < 100:
            target_x = self.width - target_x
            target_y = self.height - target_y
        last_distance = distance


# Making GUI for car simulation

class Car_GUI(Widget):

    def on_touch_down(self, touch):
        global range, n_events, a_last, b_last
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10.
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
            a_last = int(touch.x)
            b_last = int(touch.y)
            n_events = 0
            range = 0
            marker[int(touch.x), int(touch.y)] = 1

    def on_touch_move(self, touch):
        global range, n_events, a_last, b_last
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            range += np.sqrt(max((x - a_last) ** 2 + (y - b_last) ** 2, 2))
            n_events += 1.
            density = n_events / (range)
            touch.ud['line'].width = int(20 * density + 1)
            marker[int(touch.x) - 10: int(touch.x) + 10, int(touch.y) - 10: int(touch.y) + 10] = 1
            a_last = x
            b_last = y


# Add buttons on GUI for clear, load and save operations

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = Car_GUI()
        button_clear = Button(text='clear')
        button_save = Button(text='save', pos=(parent.width, 0))
        button_load = Button(text='load', pos=(2 * parent.width, 0))
        button_clear.bind(on_release=self.clear_paint)
        button_save.bind(on_release=self.save)
        button_load.bind(on_release=self.load)
        parent.add_widget(self.painter)
        parent.add_widget(button_clear)
        parent.add_widget(button_save)
        parent.add_widget(button_load)
        return parent

    def clear_paint(self, obj):
        global marker
        self.painter.canvas.clear()
        marker = np.zeros((longitude, latitude))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(performance_values)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()


# Running the whole thing
if __name__ == '__main__':
    CarApp().run()