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

class Car:
	def __init__(self, xpos, ypos):
		self.xpos = xpos
		self.ypos = ypos

if __name__ == "__main__":
	highWay = HighWay(2, 10, 0.3)
	print(highWay.road)
	
