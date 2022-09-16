import numpy as np
import pandas as pd
import random
import argparse
import datetime
import time
from matplotlib import pyplot as plt
import math
import copy

import os, sys, random
#from agent_ypp import use_agent
# from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from environment import HVAC_env


class HVAC_Building(object):
    def __init__(self, type, area, height, layers, setted_temperature):
        # international standard parameter
        self.c_water = 4.2          #水的比热容c(kJ/kg·℃)
        self.c_air = 1.005          #室内气体比热容c(kJ/kg·℃)
        self.density_air = 1.205    #室内气体密度ρ(kg/m³)

        # building parameter
        self.type = type
        self.area = area
        self.height = height
        self.layers = layers
        self.volume = self.area * self.height * self.layers
        self.surface_area = 2*self.area + 4*(math.sqrt(self.area) * self.height * self.layers)
        self.setted_temperature = setted_temperature
        self.U = 0.0036             #墙体传热系数U (kW/㎡·℃)
        self.COP = 5.0              #HVAC能效比
        self.UA = self.U * self.surface_area
        self.CPV = self.c_air * self.density_air * self.volume

        # PID parameter
        self.PID_P_water_mass = 1 * self.volume / 10000
        self.PID_I_water_mass = 0.1 * self.volume / 10000
        self.return_water_aim = 12
        self.PID_P_wind = 0.5 * self.volume / 1000000
        self.PID_I_wind = 0.05 * self.volume / 1000000

        # HVAC parameter
        self.supply_water = 3
        self.water_mass_max = 4 * self.volume / 10000
        self.water_mass_min = 0.1 * self.water_mass_max
        self.wind_mass_max = 150 * self.volume / 10000
        self.wind_mass_min = 0.1 * self.wind_mass_max
        
        # HVAC variables, need to update, just default value is 0.0
        self.current_inside_temperature = 0.0
        self.return_water = 0.0
        self.supply_wind = 0.0
        self.return_wind = 0.0
        self.water_mass = 0.0
        self.power = 0.0
        self.wind_mass = 0.0
        

    def building_initialization(self, current_outside_temperature, current_people_flow):
        # HVAC variables, need to update
        self.current_inside_temperature = copy.deepcopy(self.setted_temperature)
        self.return_water = 12
        self.supply_wind = (self.supply_water + self.return_water) / 2
        self.return_wind = copy.deepcopy(self.current_inside_temperature)
        
        delta_temperature = current_outside_temperature - self.current_inside_temperature
        Q_loss = self.UA * delta_temperature + self.CPV * delta_temperature * current_people_flow
        self.wind_mass = Q_loss / (self.c_air * (self.return_wind-self.supply_wind))
        self.water_mass = Q_loss / (self.c_water * (self.return_water-self.supply_water))
        self.power = self.c_water * self.water_mass * (self.return_water - self.supply_water)
        
        
    def normal_mode(self, current_outside_temperature, current_people_flow):
        delta_temperature = current_outside_temperature - self.current_inside_temperature
        Q_loss = self.UA * delta_temperature + self.CPV * delta_temperature * current_people_flow
        Q_gain = self.c_air * self.wind_mass * (self.return_wind - self.supply_wind)
        inside_temperature_temp = self.current_inside_temperature
        self.current_inside_temperature += 60 * (Q_loss - Q_gain) / self.CPV
        self.return_wind = self.current_inside_temperature    
        # update wind_mass
        self.wind_mass += self.PID_P_wind * (self.current_inside_temperature - inside_temperature_temp) + self.PID_I_wind * (self.current_inside_temperature - self.setted_temperature)
        self.wind_mass = self.wind_mass.clip(0, self.wind_mass_max)     
        #update return_water temperature
        return_water_temp = copy.deepcopy(self.return_water)
        self.return_water = (self.c_air * self.wind_mass * (self.return_wind - self.supply_wind)) / (self.c_water * self.water_mass) + self.supply_water  
        self.water_mass += self.PID_P_water_mass * (self.return_water - return_water_temp) + self.PID_I_water_mass * (self.return_water - self.return_water_aim)       
        self.water_mass = self.water_mass.clip(self.water_mass_min, self.water_mass_max)
        
        self.supply_wind = self.return_wind - self.c_water * self.water_mass * (self.return_water - self.supply_water) / (self.c_air * self.wind_mass)
        self.power = self.c_water * self.water_mass * (self.return_water - self.supply_water)
        

    def control_mode(self, current_outside_temperature, current_people_flow, signal, capacity, power_base):
        delta_power = signal * capacity + power_base - self.power
        # print(delta_power)
        # print(signal * capacity + power_base, self.power, delta_power, self.return_water)
        self.water_mass += delta_power / (self.c_water * (self.return_water - self.supply_water))
        self.water_mass = self.water_mass.clip(self.water_mass_min, self.water_mass_max)
        # print(self.supply_wind)
        self.supply_wind = self.return_wind - self.c_water * self.water_mass * (self.return_water - self.supply_water) / (self.c_air * self.wind_mass)
        # print(self.supply_wind)
        
        delta_temperature = current_outside_temperature - self.current_inside_temperature
        Q_loss = self.UA * delta_temperature + self.CPV * delta_temperature * current_people_flow
        Q_gain = self.c_air * self.wind_mass * (self.return_wind - self.supply_wind)
        inside_temperature_temp = self.current_inside_temperature
        # print(self.current_inside_temperature)
        self.current_inside_temperature += 60 * (Q_loss - Q_gain) / self.CPV
        # print(self.current_inside_temperature)
        # input()
        self.return_wind = self.current_inside_temperature
        # update wind_mass
        self.wind_mass += self.PID_P_wind * (self.current_inside_temperature - inside_temperature_temp) + self.PID_I_wind * (self.current_inside_temperature - self.setted_temperature)
        self.wind_mass = self.wind_mass.clip(0, self.wind_mass_max)
        #update return_water temperature
        self.return_water = (self.c_air * self.wind_mass * (self.return_wind - self.supply_wind)) / (self.c_water * self.water_mass) + self.supply_water
        # print(self.return_water)
        # update power
        self.power = self.c_water * self.water_mass * (self.return_water - self.supply_water)

def get_today_people_flow(seed):
    np.random.seed(seed)
    
    hourly_people_flow = (pd.read_excel("people_flow.xlsx")).iloc[:, 0]

    miniute_people_flow = np.array([[hourly_people_flow[i]]*60 for i in range(len(hourly_people_flow))]).flatten()
    miniute_people_flow = miniute_people_flow * (1+np.random.uniform(-0.2, 0.2, len(miniute_people_flow)))
    return miniute_people_flow / 3600


if __name__ == '__main__':
    
    # load data
    ypp_house = HVAC_Building('office', random.randrange(3000,4001), 4, random.randrange(60,100), 10)
    outside_weather = pd.read_excel("environ_T.xlsx", sheet_name=0)
    outside_weather = outside_weather[outside_weather.index % 2 == 0].reset_index(drop=True)
    outside_temperature = outside_weather['Temperature'].map(lambda x: (int(x[0:2])-32)/1.8)
    # people_flow = (random_ee(2000, [1]) / 3600).flatten()
    people_flow = get_today_people_flow(2000)
    
    # load signals
    signals = pd.read_excel("signal.xlsx", sheet_name=1)
    dayily_signals = signals.iloc[:,1]
    signal = dayily_signals[(dayily_signals.index % 30) == 0].reset_index(drop=True)
    signal.drop([1440], inplace=True)

    current_inside_temperature = []
    return_water = []
    wind_mass = []
    water_mass = []
    power = []
    
    ypp_house.building_initialization(outside_temperature[0], people_flow[0])
    # initialization
    current_inside_temperature.append(ypp_house.current_inside_temperature)
    return_water.append(ypp_house.return_water)
    wind_mass.append(ypp_house.wind_mass)
    water_mass.append(ypp_house.water_mass)
    power.append(ypp_house.power)
    
    wzy_house = copy.deepcopy(ypp_house)

    for i in range(24):
        for t in range(60):
            if not (i==0 and t==0):
                ypp_house.normal_mode(outside_temperature[i], people_flow[t+i*60])
                # record the value of variables
                current_inside_temperature.append(ypp_house.current_inside_temperature)
                return_water.append(ypp_house.return_water)
                wind_mass.append(ypp_house.wind_mass)
                water_mass.append(ypp_house.water_mass)
                power.append(ypp_house.power)
    
    # fig_, axes = plt.subplots(2, 2)
    # axes[0,0].plot(np.array(current_inside_temperature) - ypp_house.setted_temperature)
    # axes[0,0].set_title('Delta_tempreature')
    # axes[0,1].plot(power)
    # axes[0,1].set_title('Power')
    # axes[1,0].plot(wind_mass)
    # axes[1,0].set_title('Wind_mass')
    # axes[1,1].plot(current_inside_temperature)
    # axes[1,1].set_title('Inside_temperature')
    # plt.show()
    print(len(power))
    print(power[:5])

    base_power = [np.mean(power[i*60:i*60+60]) for i in range(24)]
    
    # # new era
    current_inside_temperature = []
    return_water = []
    wind_mass = []
    water_mass = []
    power = []

    current_inside_temperature.append(wzy_house.current_inside_temperature)
    return_water.append(wzy_house.return_water)
    wind_mass.append(wzy_house.wind_mass)
    water_mass.append(wzy_house.water_mass)
    power.append(wzy_house.power)

    for i in range(24):
        if i == 15:
            # current_inside_temperature = []
            # return_water = []
            # wind_mass = []
            # water_mass = []
            # power = []
            for t in range(60):
                wzy_house.control_mode(outside_temperature[i], people_flow[t+i*60], signal[t+i*60], 0.3*base_power[i], base_power[i])
                current_inside_temperature.append(wzy_house.current_inside_temperature)
                return_water.append(wzy_house.return_water)
                wind_mass.append(wzy_house.wind_mass)
                water_mass.append(wzy_house.water_mass)
                power.append(wzy_house.power)
            
            # fig_, axes = plt.subplots(2, 2)
            # axes[0,0].plot(np.array(current_inside_temperature) - ypp_house.setted_temperature)
            # axes[0,0].set_title('Delta_tempreature')
            # axes[0,1].plot(power, c='b')
            # base = np.array([base_power[i]] *60).flatten()
            # print(base)
            # print(power)
            # print((base+signal[i*60:i*60+60]*(0.1*base)).tolist())
            # axes[0,1].plot((base+signal[i*60:i*60+60]*(0.1*base)).tolist(), c='r')
            # axes[0,1].set_title('Power')
            # axes[1,0].plot(wind_mass)
            # axes[1,0].set_title('Wind_mass')
            # axes[1,1].plot(return_water)
            # axes[1,1].set_title('Return_water')
            # plt.show()

        else:
            for t in range(60):
                wzy_house.normal_mode(outside_temperature[i], people_flow[t+i*60])
                current_inside_temperature.append(wzy_house.current_inside_temperature)
                return_water.append(wzy_house.return_water)
                wind_mass.append(wzy_house.wind_mass)
                water_mass.append(wzy_house.water_mass)
                power.append(wzy_house.power)

    fig_, axes = plt.subplots(2, 2)
    axes[0,0].plot(np.array(current_inside_temperature) - ypp_house.setted_temperature)
    axes[0,0].set_title('Delta_tempreature')
    axes[0,1].plot(power, c='b')
    axes[0,1].plot(np.array( [ [base_power[i]] *60 for i in range(24) ] ).flatten()+signal*(0.3*np.array( [ [base_power[i]] *60 for i in range(24) ] ).flatten()), c='r')
    axes[0,1].set_title('Power')
    axes[1,0].plot(wind_mass)
    axes[1,0].set_title('Wind_mass')
    axes[1,1].plot(current_inside_temperature)
    axes[1,1].set_title('Inside_temperature')
    plt.show()
    
    