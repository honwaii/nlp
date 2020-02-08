import json
import time
from functools import reduce

import math
import plotly.graph_objects as go
import requests
import networkx as nx
import matplotlib.pyplot as plt
from pypinyin import pinyin, Style
from collections import defaultdict

city_codes = {"北京": 1100, '上海': 3100, '广州 ': 4401, '深圳': 4403, '香港': 8100, '南京': 3201, '长沙': 4301, '成都': 5101,
			  '重庆': 5000, '天津': 1200, '沈阳': 2101, '杭州': 3301, '武汉': 4201, '苏州': 3205, '大连': 2102, '西安': 6101,
			  '长春': 2201, '昆明': 5301, '佛山': 4406, '郑州': 4101, '宁波': 3302, '无锡': 3202, '哈尔滨': 2301}


def get_subway_routes(city_code, city):
	timestamp = str(round(time.time() * 1000))
	url = 'http://map.amap.com/service/subway?_' + timestamp + '&srhdata=' + str(city_code) + '_drw_' + city + '.json'
	print('获取城市地铁路线信息url=' + url)
	response = requests.get(url)
	response.encoding = 'utf-8'
	return response.text


def parse_stations(subway_routes):
	routes = {}
	data = json.loads(subway_routes)
	route = list(data["l"])
	for each in route:
		route_name = each['ln']
		route_stations = {}
		for st in each['st']:
			st_name = st['n']
			st_lng_lat = str(st['sl']).split(",")
			route_stations[st_name] = (st_lng_lat[0], st_lng_lat[1])
		routes[route_name] = route_stations
	return routes


# 采用plotly绘制地图
def plot_stations(routes):
	lon = []
	lat = []
	for route in routes:
		stations = routes[route]
		temp_lon = []
		temp_lat = []
		for st in stations:
			coor_wgs = gcj_2_wgs(float(stations[st][0]), float(stations[st][1]))
			temp_lon.append(coor_wgs[0])
			temp_lat.append(coor_wgs[1])
		lon.append(temp_lon)
		lat.append(temp_lat)

	# 在地图上显示各点
	fig = go.Figure(go.Scattermapbox(
		mode="markers+lines",
		marker={'size': 10}))

	fig.add_traces([go.Scattermapbox(
		mode="markers+lines",
		lon=lo,
		lat=la,
		marker={'size': 10}) for lo, la in zip(lon, lat)])

	fig.update_layout(
		margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
		mapbox={
			'center': {'lon': 120.234391, 'lat': 30.167585},
			'style': "open-street-map",
			'zoom': 1})
	fig.show()
	return (lon, lat)


#  将从搞得地图获取的gcj02坐标转换为wgs84坐标
def gcj_2_wgs(lon, lat):
	a = 6378245.0  # 克拉索夫斯基椭球参数长半轴a
	ee = 0.00669342162296594323  # 克拉索夫斯基椭球参数第一偏心率平方
	PI = 3.14159265358979324
	x = lon - 105.0
	y = lat - 35.0
	# 经度
	dLon = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
	dLon += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0
	dLon += (20.0 * math.sin(x * PI) + 40.0 * math.sin(x / 3.0 * PI)) * 2.0 / 3.0
	dLon += (150.0 * math.sin(x / 12.0 * PI) + 300.0 * math.sin(x / 30.0 * PI)) * 2.0 / 3.0
	# 纬度
	dLat = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
	dLat += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0
	dLat += (20.0 * math.sin(y * PI) + 40.0 * math.sin(y / 3.0 * PI)) * 2.0 / 3.0
	dLat += (160.0 * math.sin(y / 12.0 * PI) + 320 * math.sin(y * PI / 30.0)) * 2.0 / 3.0
	radLat = lat / 180.0 * PI
	magic = math.sin(radLat)
	magic = 1 - ee * magic * magic
	sqrtMagic = math.sqrt(magic)
	dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * PI)
	dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * PI)
	wgsLon = (lon - dLon)
	wgsLat = (lat - dLat)
	return wgsLon, wgsLat


def draw_network(routes):
	stations_info = get_stations_info(routes)
	print(stations_info)
	subway_graph = nx.Graph()
	subway_graph.add_nodes_from(list(stations_info.keys()))
	nx.draw(subway_graph, stations_info, with_labels=True, node_size=5, font_size=5)
	plt.rcParams['font.sans-serif'] = ['SimHei']
	plt.rcParams['axes.unicode_minus'] = False
	plt.savefig('stations.png', bbox_inches='tight')
	plt.show()
	return


def get_stations_info(routes):
	stations_info = {}
	for route in routes:
		for station in routes[route]:
			stations_info[station] = gcj_2_wgs(float(routes[route][station][0]), float(routes[route][station][1]))
	return stations_info


def build_connection(routes):
	station_connections = defaultdict(list)
	for route in routes:
		stations = list(routes[route].keys())
		last_one = None
		last_two = None
		for station in stations:
			if last_one is not None:
				if station not in station_connections[last_one]:
					station_connections[last_one].append(station)
				if last_one not in station_connections[station]:
					station_connections[station].append(last_one)
			if last_two is not None:
				if last_two not in station_connections[last_one]:
					station_connections[last_one].append(last_two)
			last_two = last_one
			last_one = station
	return station_connections


def draw_subway_lines(routes):
	plt.figure()
	plt.subplot(111)
	stations_info = get_stations_info(routes)
	cities_connection_graph = nx.Graph(build_connection(routes))
	nx.draw(cities_connection_graph, stations_info, with_labels=True, node_size=5, font_size=5)
	plt.rcParams['font.sans-serif'] = ['SimHei']
	plt.rcParams['axes.unicode_minus'] = False
	plt.savefig('./subway_lines.png', bbox_inches='tight')
	plt.show()


def search_with_bfs(graph, start, destination):
	pathes = [[start]]  # list 用来存储待搜索路径
	visited = set()  # set用来存储已搜索的节点
	while pathes:
		path = pathes.pop(0)  # 提取第一条路径
		frontier = path[-1]  # 提取即将要探索的节点
		if frontier in visited: continue  # 检查如果该点已经探索过 则不用再探索
		successors = graph[frontier]
		for city in successors:  # 遍历子节点
			if city in path: continue  # check loop #检查会不会形成环
			new_path = path + [city]
			pathes.append(new_path)  # bfs     #将新路径加到list里面
			# pathes = [new_path] + pathes #dfs
			if city == destination:  # 检查目的地是不是已经搜索到了
				return new_path
		visited.add(frontier)


def search(start, end, city):
	city_code = city_codes.get(city)
	city_name = reduce(lambda x, y: x[0] + y[0], pinyin(city, style=Style.NORMAL))
	subways = get_subway_routes(city_code, city_name)
	routes = parse_stations(subways)
	draw_subway_lines(routes)
	connections = build_connection(routes)
	route_stations = search_with_bfs(connections, start, end)
	if route_stations is None:
		print('没有找到匹配的路线，请检查城市或站台的名称是否正确.')
		return
	print("第二题: 根据查询直接返回查到的第一条线路:")
	route = reduce(lambda x, y: x + '->' + y, route_stations)
	print('从 ' + start + ' 到 ' + end + ' 的地铁路线是: \n' + route + '\n')

	print("第三题: 查询判断最合适的一条路线:")
	all_paths = search_all_routes(connections, start, end)
	most_suitable_path = get_suitable_path(all_paths, get_stations_info(routes))
	route = reduce(lambda x, y: x + '->' + y, most_suitable_path)
	print('从 ' + start + ' 到 ' + end + ' 的地铁最合适的路线是: \n' + route + '\n')
	return


def compute_distance(origin, destination):
	lat1, lon1 = origin
	lat2, lon2 = destination
	radius = 6371  # km
	dlat = math.radians(lat2 - lat1)
	dlon = math.radians(lon2 - lon1)
	a = (math.sin(dlat / 2) * math.sin(dlat / 2)
		 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
		 * math.sin(dlon / 2) * math.sin(dlon / 2))
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
	d = radius * c
	return d


def search_with_dfs(graph, start, destination):
	paths = [[start]]  # list 用来存储待搜索路径
	visited = set()  # set用来存储已搜索的节点
	while paths:
		path = paths.pop(0)  # 提取第一条路径
		frontier = path[-1]  # 提取即将要探索的节点
		if frontier in visited: continue  # 检查如果该点已经探索过 则不用再探索
		successors = graph[frontier]
		for city in successors:  # 遍历子节点
			if city in path: continue  # check loop #检查会不会形成环
			new_path = path + [city]
			paths.append(new_path)  # bfs     #将新路径加到list里面
			# pathes = [new_path] + pathes #dfs
			if city == destination:  # 检查目的地是不是已经搜索到了
				return new_path
		visited.add(frontier)


def search_all_routes(graph, start, destination):
	paths = []
	search_list = [[start]]
	# visited = set()
	while search_list:
		path = search_list.pop(0)
		frontier = path[-1]
		# if frontier in visited: continue  # 检查如果该点已经探索过 则不用再探索
		if frontier == destination:
			paths.append(path)
			continue
		successors = graph[frontier]
		for city in successors:
			if city in path: continue  # check loop
			new_path = path + [city]
			search_list.append(new_path)  # bfs
		# visited.add(frontier)
		if len(paths) > 5:
			break
	print("已至少找到 " + str(len(paths)) + " 条符合的线路.")
	return paths


def get_suitable_path(paths, station_info):
	min_distance_path = None
	min_distance = 0
	for path in paths:
		last_station = None
		distance = 0
		for station in path:
			if last_station:
				distance = distance + compute_distance(station_info[last_station], station_info[station])
			last_station = station
		if min_distance <= 0:
			min_distance = distance
			min_distance_path = path
		if distance < min_distance:
			min_distance = distance
			min_distance_path = path
	paths.sort(key=lambda x: len(x))
	if len(min_distance_path) < len(paths[0]) - 3:
		return min_distance_path
	return paths[0]


search('奥体中心', '天安门东', '北京')
search('近江', '火车东站', '杭州')
