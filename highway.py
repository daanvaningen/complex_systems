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
		if car.y == self.length -1: # End of track
			self.road[car.x, car.y] = None
			self.removed_cars.append(car)
			self.cars.remove(car)

		else:
			next_positions = car.get_possible_next_positions(self.road)
			if len(next_positions) == 0:
				car.blocks += 1
			else:
				next_x, next_y = next_positions[0][0], next_positions[0][1]
				self.road[car.x, car.y] = None
				self.road[next_x, next_y] = car
				car.x, car.y = next_x, next_y

	def new_flow_of_cars(self):
		for i in range(self.lanes):
			if self.road[i,0].__class__.__name__ is not 'Car' and random.random() < self.density:
				new_car = Car(i,0)
				self.cars.append(new_car)
				self.road[i,0] = new_car

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

	def run(self):
		for _ in range(self.max_iterations):
			if len(self.cars) != 0:
				for car in np.random.choice(self.cars,len(self.cars),replace=False):
					self.action(car)
				self.new_flow_of_cars()
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
		positions = []

		# Front
		if highway[self.x,self.y+1].__class__.__name__ is not 'Car':
			positions.append([self.x, self.y+1])

		if self.x == 0:
			if highway[self.x+1, self.y+1].__class__.__name__ is not 'Car':
				positions.append([self.x+1, self.y+1])

		elif self.x == highway.shape[0] - 1:
			if highway[self.x-1, self.y+1].__class__.__name__ is not 'Car':
				positions.append([self.x-1, self.y+1])

		else:
			if highway[self.x-1, self.y+1].__class__.__name__ is not 'Car':
				positions.append([self.x-1, self.y+1])

			elif highway[self.x+1, self.y+1].__class__.__name__ is not 'Car':
				positions.append([self.x+1, self.y+1])

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
	densities = np.linspace(0.01, 1, 30)
	for p in densities:
		highWay = HighWay(lanes, length, p, iterations)
		highWay.run()
		blocks.append(highWay.n_blocks)
	plt.plot(densities,blocks)
	plt.xlabel("Density")
	plt.ylabel("# Cars blocked")
	plt.show()

if __name__ == "__main__":
	lanes, length, density, iterations = 3, 60, 0.99, 10000
	
	highWay = HighWay(lanes, length, density, iterations)
	highWay.run()
	plt.plot(highWay.occupied)
	plt.ylabel('% road occupied')
	plt.xlabel('iterations')
	plt.show()

	size3 = Analysis_of_density(3, length, iterations)
	size4 = Analysis_of_density(4, length, iterations)

	plt.plot(size3[0],size3[1], label = "Highway size = 3")
	plt.plot(size4[0],size4[1], label = "Highway size = 4")
	plt.xlabel("Cars in the system")
	plt.ylabel("Average duration for passage")
	plt.legend()
	plt.title("Time to pass the highway for number of vehicles")
	plt.show()
