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
	def __init__(self, lanes, length, density, v_max):
		self.lanes = lanes
		self.length = length
		self.road = np.empty((lanes, length), dtype=object)
		self.density = density
		self.cars = []				# List of all cars on the road
		self.removed_cars = []		# list of all removed cars
		self.populate_road()
		self.occupied = []	 # percentage of road occupied
		self.v_max = v_max
		self.passes = 0


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
		# accelerate
		if car.v < self.v_max:
			car.v += 1

		# probability of slowing down
		if np.random.random_sample() < 0.3 and car.v > 0:
			car.v -= 1

		# move back a lane if follower or preceder is faster
		if ((self.v_following(car) > car.v or self.v_preceding(car) > car.v)
		and self.gap_right(car) > 0):
			move = min(self.gap_right(car),car.v)
			next_x, next_y = car.x-1, (car.y+move)%self.length

		# move left a lane if there is more room
		elif self.gap_left(car) > self.gap_front(car):
			move = min(self.gap_left(car),car.v)
			next_x, next_y = car.x+1, (car.y+move)%self.length

		# else go straight
		else:
			move = min(self.gap_front(car),car.v)
			next_x, next_y = car.x, (car.y+move)%self.length
			# print('straight')
		# print(next_x, next_y, move)
		if next_y < car.y: self.passes += 1
		self.road[car.x, car.y] = None
		self.road[next_x, next_y] = car
		car.x, car.y, car_v = next_x, next_y, move

	# def new_flow_of_cars(self):
		# for i in range(self.lanes):
			# if self.road[i,0].__class__.__name__ is not 'Car' and random.random() < self.density:
				# new_car = Car(i,0)
				# self.cars.append(new_car)
				# self.road[i,0] = new_car

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

	def v_following(self, car):
		i = 1
		while self.road[car.x,(car.y-i)%self.length] is None:
			i += 1
		follower = self.road[car.x,(car.y-i)%self.length]
		return follower.v

	def v_preceding(self, car):
		i = 1
		while self.road[car.x-1,(car.y+i)%self.length] is None:
			i += 1
		preceding = self.road[car.x-1,(car.y+i)%self.length]
		return preceding.v

	def gap_front(self, car):
		i = 1
		while self.road[car.x,(car.y+i)%self.length] is None:
			i += 1
		return i-1

	def gap_left(self, car):
		if car.x == self.lanes - 1: return 0
		i = 1
		while self.road[car.x+1,(car.y+i)%self.length] is None:
			i += 1
		return i-1

	def gap_right(self, car):
		if car.x == 0: return 0
		i = 1
		while self.road[car.x-1,(car.y+i)%self.length] is None:
			i += 1
		return i-1

	def run(self, max_iterations):
		for _ in range(max_iterations):
			for car in np.random.choice(self.cars,len(self.cars),replace=False):
				self.action(car)
			self.occupied.append(len(self.cars)/(self.length*self.lanes))

class Car:
	def __init__(self, xpos, ypos):
		self.x = xpos
		self.y = ypos
		self.v = 1
		self.blocks = 0


	def get_elapsed_time(self):
		''' returns the `time` a car spends in the system is the sum of the number of
			times it could not move forward and its current y position
		'''
		return self.blocks + self.y

def analyze_flow(lanes, length, iterations, v_max):
	flux = []
	densities = np.linspace(0.1,1,25)
	for p in tqdm(densities):
		highWay = HighWay(lanes, length, p, v_max)
		highWay.run(iterations)
		flux.append(highWay.passes/iterations)
	plt.plot(densities, flux)
	plt.ylabel('flow')
	plt.xlabel('density')
	# plt.show()

def analyze_speed(lanes, length, iterations, v_max):
	densities = np.linspace(0.1,1,25)
	speeds = []
	for p in tqdm(densities):
		highWay = HighWay(lanes, length, p, v_max)
		highWay.run(iterations)
		speeds.append(highWay.speed)
	plt.plot(densities, speeds)
	plt.ylabel('average speed')
	plt.xlabel('density')
	# plt.show()

def Analyze_diferrent_speeds(lanes, length, iterations, v_begin, v_end, precision = 10):
	all_v = np.linspace(v_begin, v_end, v_end - v_begin + 1)
	for i in tqdm(all_v):
		flows = []
		cars = np.linspace(50, lanes*length, 100)
		for c in cars:
			avg_flows = []
			for j in range(precision):
				highWay = HighWay(lanes, length, c/(lanes*length), i)
				highWay.run(iterations)
				avg_flows.append(highWay.passes/iterations)
			flows.append(np.mean(avg_flows))
		plt.plot(cars, flows, label = 'v_max = '+str(i))

	limit_function = [-0.01*j + 4 for j in cars]
	plt.plot(cars, limit_function, label = str(-0.01)+"*cars"+str("+4"))
	plt.title('Lanes, length, iterations = ' +str(lanes) + ' , ' +str(length) + ' , ' +str(iterations))
	plt.legend(loc='best')
	plt.xlabel('# cars')
	plt.ylabel('flow (cars passed per iteration)')
	plt.show()

def Analyze_different_lanes(length, iterations, vmax, lanes_begin, lanes_end, precision = 10):
	all_lanes = np.linspace(lanes_begin, lanes_end, lanes_end - lanes_begin + 1)
	for i in tqdm(all_lanes):
		flows = []
		cars = np.linspace(50, i*length, 100)
		for c in cars:
			avg_flows = []
			for j in range(precision):
				highWay = HighWay(int(i), length, c/(i*length), vmax)
				highWay.run(iterations)
				avg_flows.append(highWay.passes/iterations)
			flows.append(np.mean(avg_flows))
		plt.plot(cars, flows, label = 'Lanes = '+str(i))
	plt.title('Vmax, length, iterations = ' +str(vmax) + ' , ' +str(length) + ' , ' +str(iterations))
	plt.legend(loc='best')
	plt.xlabel('# cars')
	plt.ylabel('flow (cars passed per iteration)')
	plt.show()


if __name__ == "__main__":

	lanes, length, iterations, v_max = 2, 200, 100, 3

	'''Please note that the following functions migth take 15 minutes to run!!
	'''

	Analyze_diferrent_speeds(lanes, length, iterations, 3, 10, precision = 5)
	# Analyze_different_lanes(length, iterations, v_max, 2, 5, precision = 5)
