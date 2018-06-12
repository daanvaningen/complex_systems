import numpy as np
import random

class HighWay:
	def __init__(self, lanes, length, density, iterations):
		self.lanes = lanes
		self.length = length
		self.road = np.empty((lanes, length), dtype=object)
		self.density = density
		self.cars = []				# List of all cars on the road
		self.iterations = iterations	# Max iterations
		self.carsPassed = 0		# Keeps track of the number of cars passed
		self.all_cars = 0 		# Keeps track of the number of cars total.
		self.sum_of_iterations = 0 		# This is the sum of the iterations a car stays in the system. Will be divided by number of cars that passed the system later.
		# self.populate_road()

	def populate_road(self):
		for i in range(self.lanes):
			for j in range(self.length):
				if(random.random() < self.density):
					new_car = Car(i,j, 0)
					self.road[i,j] = new_car
					self.cars.append(new_car)
					self.all_cars += 1

	# this function returns a matrix with zeros and ones. Every one is a car, such that
	# we can plot it using imshow.
	def visualize_road(self):
		self.visualized_road = np.zeros((self.lanes, self.length))
		for i in range(self.lanes):
			for j in range(self.length):
				if type(self.road[i,j]) is Car:
					self.visualized_road[i,j] = 1

		return self.visualized_road

	def action(self, next_car):
		i = next_car.xpos
		j = next_car.ypos
		if j == self.length -1: # End of track
			self.road[i,j] = None
			self.cars.remove(next_car)
			self.carsPassed += 1
			self.sum_of_iterations += (self.tic - next_car.begin)

		else:
			# If place in front free, go there
			if type(self.road[i,j+1]) is not Car:
				self.road[i,j+1] = next_car
				self.road[i,j] = None
				next_car.ypos += 1
				return

			if i == 0:
				if type(self.road[i+1, j+1]) is not Car:
					self.road[i+1, j+1] = next_car
					self.road[i,j] = None
					next_car.xpos += 1
					next_car.ypos += 1
					return

			elif i == self.lanes - 1:
				if type(self.road[i-1, j+1]) is  not Car:
					self.road[i-1, j+1] = next_car
					self.road[i,j] = None
					next_car.xpos -= 1
					next_car.ypos += 1
					return
			else:
				if type(self.road[i-1, j+1]) is not Car:
					self.road[i-1, j+1] = next_car
					self.road[i,j] = None
					next_car.xpos -= 1
					next_car.ypos += 1
					return
					# Note that this will make sure a car will choose left over rightself.

				elif type(self.road[i+1, j+1]) is not Car:
					self.road[i+1, j+1] = next_car
					self.road[i,j] = None
					next_car.xpos += 1
					next_car.ypos += 1
					return
				else:
					return

	def new_flow_of_cars(self):
		for i in range(self.lanes):
			if type(self.road[i,0]) is not Car and random.random() < self.density:
				new_car = Car(i,0, self.tic)
				self.cars.append(new_car)
				self.road[i,0] = new_car
				self.all_cars += 1

	def run(self):
		for self.tic in range(self.iterations):
			self.new_flow_of_cars()
			if len(self.cars) != 0:
				next_car = random.choice(self.cars)
				self.action(next_car)



		self.visualized_road = self.visualize_road()
		return self.visualized_road

class Car:
	def __init__(self, xpos, ypos, start_iteration):
		self.xpos = xpos
		self.ypos = ypos
		self.begin = start_iteration

if __name__ == "__main__":
	highWay = HighWay(3, 10, 0.05, 1000)
	ending = highWay.run()
	Succesfullcars = highWay.carsPassed

	print("The state after all iterations was: ")
	print(ending)
	print("-----")
	print("The number of cars that passed: ", int(Succesfullcars/float(highWay.all_cars)*100.0), "percent")
	print("Average duration of passage ", int(highWay.sum_of_iterations/Succesfullcars), " Iterations")
