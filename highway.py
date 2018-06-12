import numpy as np
import random

class HighWay:
	def __init__(self, lanes, length, density):
		self.lanes = lanes
		self.length = length
		self.road = np.empty((lanes, length), dtype=object)
		self.density = density
		self.populate_road()

	def populate_road(self):
		for i in range(self.lanes):
			for j in range(self.length):
				if(random.random() < self.density):
					self.road[i,j] = Car(i,j)

	# this function returns a matrix with zeros and ones. Every one is a car, such that
	# we can plot it using imshow.
	def visualize_road(self):
		self.visualized_road = np.zeros((lanes, length))
		for i in range(self.lanes):
			for j in range(self.length):
				if self.road[i,j] != None:
					self.visualized_road[i,j] = 1

		return self.visualized_road

class Car:
	def __init__(self, xpos, ypos):
		self.xpos = xpos
		self.ypos = ypos

if __name__ == "__main__":
	highWay = HighWay(2, 10, 0.3)
	print(highWay.road)
	
