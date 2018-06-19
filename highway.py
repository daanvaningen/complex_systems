'''
Complex Systems 2018

Program that simulates traffic on a highway

Authors:
	Mauricio Fonseca Fernandez
	Berend Nannes
	Sam Ferwerda
	Daan van Ingen
'''

import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

class HighWay:
	def __init__(self, lanes, length, density, max_iterations):
		self.lanes = lanes
		self.length = length
		self.road = np.empty((lanes, length), dtype=object)
		self.density = density
		self.cars = []				# List of all cars on the road
		self.removed_cars = []		# list of all removed cars
		self.max_iterations = max_iterations	# Max iterations
		self.populate_road()
		self.occupied = []	 # percentage of road occupied

	def populate_road(self):
		''' Populate the road with cars, the number of initial cars depends on the
			density of the traffic
		'''
		for i in range(self.lanes):
			for j in range(self.length):
				if(random.random() < self.density):
					new_car = Car(i,j)
					self.road[i,j] = new_car
					self.cars.append(new_car)

	def visualize_road(self):
		''' prints a matrix with zeros and ones. Every one is a car, such that
			we can plot it using imshow.
		'''
		visualized_road = np.zeros((self.lanes, self.length))
		for i in range(self.lanes):
			for j in range(self.length):
				if self.road[i,j].__class__.__name__ is 'Car':
					visualized_road[i,j] = 1

		return visualized_road

	def action(self, car):
		''' Remove the car from the system if it is at the end of the track,
			otherwise try to move the car forward
		'''
		next_positions = car.get_possible_next_positions(self.road)
		if len(next_positions) == 0:
			car.blocks += 1
		else:
			next_x, next_y = next_positions[0][0], next_positions[0][1] # We might want to randomise this
			self.road[car.x, car.y] = None
			self.road[next_x, next_y] = car
			car.x, car.y = next_x, next_y

	def new_flow_of_cars(self):
		''' Try to add a new car at the beginning of the highway if the density
			of cars is too low
		'''
		# print(len(self.cars))
		# print(int(self.lanes*self.length*self.density))
		if len(self.cars) < int(self.lanes*self.length*self.density):
			for i in range(self.lanes):
				if self.road[i,0].__class__.__name__ is not 'Car':
					new_car = Car(i,0)
					self.cars.append(new_car)
					self.road[i,0] = new_car
					break

	def get_avg_time_of_passed_cars(self):
		''' Returns the average time cars of the cars that traveled through
			the system
		'''
		if len(self.removed_cars) == 0:
			return 0
		return self.get_total_time_passed_cars()/len(self.removed_cars)

	def get_total_time_passed_cars(self):
		''' Returns the total time of the cars that traveled through the system
		'''
		total = 0
		for i in range(len(self.removed_cars)):
			total += self.removed_cars[i].get_elapsed_time()

		return total

	def get_total_blocks(self):
		''' returns the total number of blocks for all cars in the system
		'''
		total = 0
		for car in self.cars:
			total += car.blocks

		return total

	def get_throughput(self):
		''' Calculate the throughput of the system by looking at all the cars in
			the system and returning the number of forward moves devided by the
			system size
		'''
		total = 0
		for car in self.cars:
			total += self.max_iterations - car.blocks

		return total

	def step(self, car):
		''' Move one car and try to add a new one to the system
		'''
		self.action(car)
		self.new_flow_of_cars()

	def run(self):
		for _ in range(self.max_iterations):
			if len(self.cars) != 0:
				for car in np.random.choice(self.cars,len(self.cars),replace=False):
					self.step(car)
			self.occupied.append(len(self.cars)/(self.length*self.lanes))
		# self.visualize_road()

class Car:
	def __init__(self, xpos, ypos):
		self.x = xpos
		self.y = ypos
		self.blocks = 0

	def get_possible_next_positions(self, highway):
		''' Returns an array with [x,y] pairs of possible next positions to move
		'''
		x, y = self.x, self.y
		if(y == highway.shape[1]-1):
			y = -1
		positions = []

		# Front
		if highway[x,y+1].__class__.__name__ is not 'Car':
			positions.append([x, y+1])

		if x == 0:
			if highway[x+1, y+1].__class__.__name__ is not 'Car':
				positions.append([x+1, y+1])

		elif x == highway.shape[0] - 1:
			if highway[x-1, y+1].__class__.__name__ is not 'Car':
				positions.append([x-1, y+1])

		else:
			if highway[x-1, y+1].__class__.__name__ is not 'Car':
				positions.append([x-1, y+1])

			elif highway[x+1, y+1].__class__.__name__ is not 'Car':
				positions.append([x+1, y+1])

		return positions

	def get_elapsed_time(self):
		''' returns the `time` a car spends in the system is the sum of the number of
			times it could not move forward and its current y position
		'''
		return self.blocks + self.y

def Analysis_of_density(lanes, length, iterations):
	density_levels = [i/100.0 for i in range(1,101)]
	average_duration = []
	for density in tqdm(density_levels):
		durations = []
		for iter in range(1):
			highWay = HighWay(lanes, length, density, iterations)
			highWay.run()
			durations.append(highWay.get_avg_time_of_passed_cars())
		average_duration.append(np.mean(durations))

	cars_in_system = [density*lanes*length for density in density_levels]

	return cars_in_system, average_duration

def analyze_blocking(lanes, length, iterations):
	# count number of times cars are blocked
	blocks = []
	densities = np.linspace(0.05, 1, 20)
	for p in tqdm(densities):
		temp = []
		highWay = HighWay(lanes, length, p, iterations)
		highWay.run()
		blocks.append(highWay.get_total_blocks())
	plt.plot(densities,blocks)
	plt.xlabel("Density")
	plt.ylabel("# Cars blocked")
	plt.show()

def analyze_throughput(lanes, length, iterations, multiple_lanes = False):
	if multiple_lanes:
		road_sizes = [lanes - 1, lanes, lanes + 1, lanes + 2]
	else:
		road_sizes = [lanes]

	for road in tqdm(road_sizes):
		throughput = []
		densities = np.linspace(0.05, 1, 20)
		for p in densities:
			temp = []
			highWay = HighWay(road, length, p, iterations)
			highWay.run()
			throughput.append(highWay.get_throughput())
		plt.plot(densities, throughput, label="Size: "+str(road))
	plt.xlabel("Density")
	plt.ylabel("Throughput")
	plt.legend()
	plt.show()

if __name__ == "__main__":
	lanes, length, density, iterations = 3, 60, 0.99, 1000
	analyze_throughput(lanes, length, iterations, multiple_lanes = True)
