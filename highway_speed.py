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
import matplotlib.animation as animation
from matplotlib import colors
import random
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable


class HighWay:
	def __init__(self, lanes, length, density, v_max):
		self.lanes = lanes
		self.length = length
		self.road = np.empty((lanes, length), dtype=object)
		self.density = density
		self.new_car_probability = density
		self.cars = []				# List of all cars on the road
		self.removed_cars = []		# list of all removed cars
		# self.populate_road()
		new_car = Car(0,0)
		self.road[0,0] = new_car
		self.cars.append(new_car)
		self.occupied = []	 # percentage of road occupied
		self.v_max = v_max
		self.passes = 0
		self.on_ramps = [int(self.length*(1/2))]


	def populate_road(self):
		''' Populate the road with cars, the number of initial cars depends on the
			density of the traffic
		'''

		for i in range(self.lanes):
			for j in range(self.length):
				if(random.random() < self.new_car_probability):
					new_car = Car(i,j)
					self.road[i,j] = new_car
					self.cars.append(new_car)


	def visualize_road(self):
		''' returns a matrix with zeros and ones. Every one is a car, such that
			we can plot it using imshow.
		'''
		visualized_road = np.zeros((self.lanes, self.length))
		for i in range(self.lanes):
			for j in range(self.length):
				if self.road[i,j].__class__.__name__ is 'Car':
					visualized_road[i,j] = 1

		return visualized_road

	def compute_local_density(self):

		density = np.zeros((self.lanes,self.length))
		for i in range(self.lanes):
			for j in range(self.length):
				a = [self.road[i,j+k] for k in range(-5,5)
					if j+k >= 0 and j+k < self.length ]
				count = 0
				for c in a:
					if c != None:
						count += 1.
					density[i,j] = count/len(a)
		return density

	def compute_local_velocity_average(self):

		velocity_av = np.zeros((self.lanes,self.length))
		flow = np.zeros(self.length)
		for i in range(self.lanes):
			for j in range(self.length):
				a = [self.road[i,j+k] for k in range(-5,5)
					if j+k >= 0 and j+k < self.length ]
				count = 0.
				for c in a:
					if c != None:
						count += c.v
				velocity_av[i,j] = count/len(a)
		for j in range(self.length):
			for i in range(self.lanes):
				flow[j] += velocity_av[i,j]
		flow = flow/self.lanes
		return velocity_av, flow

	def get_avg_speed_per_meter(self):
		avg = np.zeros(self.length)
		for i in range(self.length):
			cross = []
			for j in range(self.lanes):
				elem = self.road[j,i]
				if type(elem) is Car:
					cross.append(elem.v)

			avg[i] = np.mean(cross)
		return avg

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
		if ((self.v_following(car) > car.v or self.v_preceding(car) >= car.v)
		and self.gap_right(car) > 0):
			move = min(self.gap_right(car),car.v)
			next_x, next_y = car.x-1, (car.y+move)

		# move left a lane if there is more room
		elif self.gap_left(car) > self.gap_front(car):
			move = min(self.gap_left(car),car.v)
			next_x, next_y = car.x+1, (car.y+move)

		# else go straight
		else:
			move = min(self.gap_front(car),car.v)
			next_x, next_y = car.x, (car.y+move)

		if next_y < self.length:
			self.road[car.x, car.y] = None
			self.road[next_x, next_y] = car
			car.x, car.y, car.v = next_x, next_y, move
		else:
			self.passes += 1
			self.road[car.x, car.y] = None
			self.cars.remove(car)

	def new_flow_of_cars(self):
		for i in range(self.lanes):
			if self.road[i,0].__class__.__name__ is not 'Car' and random.random() < self.new_car_probability:
				new_car = Car(i,0)
				new_car.v = max(1, int(self.v_max/2))
				self.cars.append(new_car)
				self.road[i,0] = new_car
		for i in self.on_ramps:
			if self.road[0,i].__class__.__name__ is not 'Car' and random.random() < self.new_car_probability:
				new_car = Car(0,i)
				new_car.v = max(1, int(self.v_max/4))
				self.cars.append(new_car)
				self.road[0,i] = new_car

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
		if car.y-i < 0: return 0
		while self.road[car.x,(car.y-i)] is None:
			i += 1
			if car.y-i < 0: return 0
		follower = self.road[car.x,(car.y-i)]
		return follower.v

	def v_preceding(self, car):
		i = 1
		if car.y+i >= self.length: return 0
		while self.road[car.x-1,(car.y+i)] is None:
			i += 1
			if car.y+i >= self.length: return car.v
		preceding = self.road[car.x-1,(car.y+i)]
		return preceding.v

	def gap_front(self, car):
		i = 1
		if car.y+i >= self.length: return car.v
		while self.road[car.x,(car.y+i)] is None:
			i += 1
			if car.y+i >= self.length: return car.v
		return i-1

	def gap_left(self, car):
		if car.x == self.lanes - 1: return 0
		i = 1
		if car.y+i >= self.length: return car.v
		while self.road[car.x+1,(car.y+i)] is None:
			i += 1
			if car.y+i >= self.length: return car.v
		return i-1

	def gap_right(self, car):
		if car.x == 0: return 0
		i = 1
		if car.y+i >= self.length: return car.v
		while self.road[car.x-1,(car.y+i)] is None:
			i += 1
			if car.y+i >= self.length: return car.v
		return i-1

	def run(self, max_iterations):
		for _ in range(max_iterations):
			self.step()

	def step(self):
		for car in np.random.choice(self.cars,len(self.cars),replace=False):
			self.action(car)
		self.new_flow_of_cars()
		self.occupied.append(len(self.cars)/(self.length*self.lanes))

	def get_speeds(self):
		m = np.empty((self.lanes, self.length))
		for i in range(self.lanes):
			for j in range(self.length):
				temp = self.road[i,j]
				if(temp is not None):
					m[i,j] = self.road[i,j].v
				else:
					m[i,j] = None

		return m

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
	plt.xlabel('new_car_probability')
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
	plt.xlabel('new_car_probability')
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

def evolution_visualization(highWay):
	highWay.run(500)
	total_time = 1000
	jam_evolution = np.zeros((total_time,length))
	velocity_average_ev = np.zeros((total_time,lanes,length))
	for t in tqdm(range(total_time)):
		highWay.run(1)
		local_density_lane2 = highWay.compute_local_density()[-1,:]
		velocity_average_ev[t,:,:], flow = highWay.compute_local_velocity_average()
		for j in range(length):
			#if local_density_lane2[j] > 0.75 and flow[j] < 2:
			if local_density_lane2[j] > 0.75:
				jam_evolution[t,j] = 1
		if t > 6:
			for j in range(length):
				count_k = 0
				for k in range(1,6):
					if np.max(velocity_average_ev[k,:,j])-np.min(velocity_average_ev[k,:,j]) <= 1:
						count_k += 1
				if count_k == 5 and jam_evolution[t,j] == 1:
					jam_evolution[t,j] == 2

	plt.imshow(np.transpose(jam_evolution))
	plt.xlabel('time')
	plt.ylabel('position')
	plt.show()

def animate_simulation(lanes, length, density, v_max):
	highWay = HighWay(lanes, length, density, v_max)
	highWay.run(10)
	speed_matrix = highWay.get_speeds()

	fig = plt.figure(figsize=(20,12))
	ax = fig.add_subplot(111)

	div = make_axes_locatable(ax)
	cax = div.append_axes('right', '5%', '5%')

	# make a color map of fixed colors
	cmap = colors.ListedColormap(['red', 'yellow', 'blue', 'green'])
	bounds=[0,1,2,3,4]
	norm = colors.BoundaryNorm(bounds, cmap.N)

	im = ax.imshow(speed_matrix, cmap=cmap, norm=norm, origin='lower', animated=True)
	ax.set_xticks(np.arange(-.5, length, 1), minor=True);
	ax.set_yticks(np.arange(-.5, lanes, 1), minor=True);
	ax.grid(b=True, which='minor', color='black', linestyle='-', linewidth=5)
	cb = fig.colorbar(im, cax=cax)
	labels = np.arange(0,4,1)
	loc    = labels + .5
	cb.set_ticks(loc)
	cb.set_ticklabels(labels)
	tx = ax.set_title('Highway with cars moving at different speeds')

	def animate(i):
		highWay.step()
		arr = highWay.get_speeds()
		print(arr)
		vmax     = np.nanmax(arr)
		vmin     = np.nanmin(arr)
		im.set_data(arr)
		im.set_clim(vmin, vmax)
		tx.set_text('Highway with cars moving at different speeds')

	ani = animation.FuncAnimation(fig, animate, interval=1000)

	plt.show()
	# cur_cmap = plt.cm.get_cmap('jet', highWay.v_max)
	# cur_cmap.set_bad(color='white')
	# ax.set_xticks(np.arange(-.5, length, 1), minor=True);
	# ax.set_yticks(np.arange(-.5, lanes, 1), minor=True);
	# ax.grid(b=True, which='minor', color='w', linestyle='-', linewidth=2)
	# im = plt.imshow(speed_matrix, animated=True, cmap=cur_cmap)
	# cb = fig.colorbar(im)
	# tx = ax.set_title('Frame 0')

	# def updatefig(i):
	# 	highWay.step()
	# 	speeds = highWay.get_speeds()
	# 	# vmax = np.max(speeds)
	# 	# vmin = np.min(speeds)
	# 	im.set_array(speeds)
	# 	# im.set_data(speeds)
	# 	# im.set_clim(vmin, vmax)
	# 	# tx.set_text('Frame {0}'.format(i))
	# 	return im,
	#
	# ani = animation.FuncAnimation(fig, updatefig, interval=200, blit=True)
	#
	# plt.show()

def analyze_phases(lanes, length, new_car_probability, v_max):
	highWay = HighWay(lanes, length, new_car_probability, v_max)
	total_time = 500
	highWay.run(500)
	jam_evolution = []
	for t in range(total_time):
		highWay.run(1)
		avg_speeds = highWay.get_avg_speed_per_meter()
		local_clusters = find_local_clusters(avg_speeds, 0.5, 1.5, 2)

		jam_evolution.append(local_clusters)
	plt.imshow(np.transpose(jam_evolution))
	plt.xlabel('time')
	plt.ylabel('position')
	plt.title('Synchronized flow, $p = $' + str(new_car_probability))
	plt.show()

def find_local_clusters(speeds, min, max, res_val):
	res = []
	length = len(speeds)
	for i in range(length):
		if(i < 2 or i > length-3):
			res.append(0)
		else:
			avg = np.mean([speeds[i-2], speeds[i-1], speeds[i], speeds[i+1], speeds[i+2]])
			if(avg > min and avg < max):
				res.append(res_val)
			else:
				res.append(0)
	return res

if __name__ == "__main__":

	#lanes, length, iterations, density, v_max = 2, 40, 100, 0.75, 2
	#animate_simulation(lanes, length, density, v_max)

	lanes, length, iterations, new_car_probability, v_max = 3, 500, 100, 0.55, 5
	highWay = HighWay(lanes, length, new_car_probability, v_max)
	analyze_phases(lanes, length,  new_car_probability, v_max)
	# evolution_visualization(highWay)
	'''Please note that the following functions migth take 15 minutes to run!!
	'''


	# Analyze_diferrent_speeds(lanes, length, iterations, 3, 10, precision = 5)
	# Analyze_different_lanes(length, iterations, v_max, 2, 5, precision = 5)
