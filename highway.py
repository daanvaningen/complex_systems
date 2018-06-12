import numpy as np
import matplotlib.pyplot as plt
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

		self.populate_road()

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

def Analysis_of_density(lanes, length, iterations):
	density_levels = [i/100.0 for i in range(1,101)]
	average_duration = []
	for density in density_levels:
		durations = []
		for iter in range(30):
			highWay = HighWay(lanes, length, density, iterations)
			highWay.run()
			if highWay.carsPassed != 0:
				durations.append(int(highWay.sum_of_iterations/highWay.carsPassed))
			else:
				durations.append(0)

		average_duration.append(np.mean(durations))

	cars_in_system = [density*lanes*length for density in density_levels]

	return cars_in_system, average_duration

if __name__ == "__main__":
	lanes, length, density, iterations = 3, 30, 0.3, 50000

	size3 = Analysis_of_density(3, length, iterations)
	size4 = Analysis_of_density(4, length, iterations)

	plt.plot(size3[0],size3[1], label = "Highway size = 3")
	plt.plot(size4[0],size4[1], label = "Highway size = 4")
	plt.xlabel("Cars in the system")
	plt.ylabel("Average duration for passage")
	plt.legend()
	plt.title("Time to pass the highway for number of vehicles")
	plt.show()
