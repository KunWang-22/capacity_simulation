import pandas as pd
import numpy as np
import random
import copy
import datetime
from matplotlib import pyplot as plt


class HVAC_env(object):
    def __init__(self, seed, building):
        self.seed = seed
        
        self.state = []
        self.reward = 0
        self.done = False

        self.state_dim = 6
        self.action_dim = 1
        self.action_max = 10000

        self.today_temperature = []
        self.today_people_flow = []
        self.today_signals = []

        self.building = building
        self.power_base = []
        self.capacity = 0

        self.comfort = 2
        self.flag = True
        self.score_standard = 0.9
        self.weight_1 = 0.2
        self.weight_2 = 0.1
        self.time = 0

    def generate_reset(self, month, num):
        # np.random.seed(self.seed)
        self.time = 0
        # day_num = np.random.randint(0,30)
        # get temperature data
        data_file = '../dataset/weather/weather_2021/weather-'+str(month)+'.xlsx'
        day_num = num
        print(month, num)
        today_weekday = datetime.datetime(2021, month, num+1).weekday()

        # get weather data
        outside_weather = pd.read_excel(data_file, sheet_name=day_num, header=None)
        outside_weather = outside_weather[outside_weather.index % 2 == 0].reset_index(drop=True)
        outside_temperature = outside_weather[1].map(lambda x: int(x[:2]))
        self.today_temperature = outside_temperature

        # get people flow data
        self.today_people_flow = self.get_today_people_flow(today_weekday)

        # get signal data
        self.today_signals = np.ones(1440)

        self.building.building_initialization(self.today_temperature[self.time], self.today_people_flow[self.time])

        self.power_base = self.get_today_base_power()
        
        self.done = False
        self.state = self.get_current_state()
        
        print('generating day: ' + str(month).zfill(2) + ' - ' + str(day_num).zfill(2))
        
        return self.state


    def reset(self):
        # np.random.seed(self.seed)
        self.time = 0
        day_num = np.random.randint(0,153)
        # day_num = 12
        # print(day_num)
        self.today_temperature = self.get_today_temperature(day_num)
        if day_num<31:
            month = 10
            num = day_num
        elif day_num<61:
            month = 6
            num = day_num-31
        elif day_num<92:
            month = 7
            num = day_num-61
        elif day_num<123:
            month = 8
            num = day_num-92
        elif day_num<153:
            month = 9
            num = day_num-123
        
        today_weekday = datetime.datetime(2021, month, num+1).weekday()
        self.today_people_flow = self.get_today_people_flow(today_weekday)
        # self.today_signals = self.get_today_signals(day_num)
        self.today_signals = np.ones(1440)

        self.building.building_initialization(self.today_temperature[self.time], self.today_people_flow[self.time])
        # print("------------------------")
        # print(self.building.power)
        # print("------------------------")
        self.power_base = self.get_today_base_power()
        
        self.done = False
        self.state = self.get_current_state()
        
        print('training day: '+str(month)+'-'+str(num+1))
        
        return self.state


    def step(self, action):
        self.capacity = action[0]
        self.flag = True
        for t in range(60):
            self.building.normal_mode(self.today_temperature[self.time], self.today_people_flow[self.time*60+t])
        self.time += 1
        try:
            self.state = self.get_current_state()
        except IndexError:
            pass
        
        # print("------------------------")
        # print(self.building.power)
        # print("------------------------")

        temp_building = copy.deepcopy(self.building)
        power_gap = []
        delta_temperature = np.abs(temp_building.current_inside_temperature-temp_building.setted_temperature)
        for t in range(60):
            if (delta_temperature<=self.comfort) and  (self.flag==True):
                temp_building.control_mode(self.today_temperature[self.time], self.today_people_flow[self.time*60+t], self.today_signals[self.time*60+t], self.capacity, self.power_base[self.time])
            else:
                temp_building.normal_mode(self.today_temperature[self.time], self.today_people_flow[self.time*60+t])
                self.flag =False
            
            delta_temperature = np.abs(temp_building.current_inside_temperature-temp_building.setted_temperature)
            if (self.flag==False) and (delta_temperature<=0.5):
                self.flag = True

            gap = (self.today_signals[self.time*60+t] * self.capacity) + self.power_base[self.time] - temp_building.power
            power_gap.append(gap)
        score_1 = 1 - np.mean(np.abs(power_gap)/(self.capacity+0.1))

        # print("==========================================")
        # print(self.capacity)
        # print(self.power_base[self.time])
        # print(temp_building.power)
        # print("==========================================")

        temp_building = copy.deepcopy(self.building)
        power_gap = []
        delta_temperature = np.abs(temp_building.current_inside_temperature-temp_building.setted_temperature)
        for t in range(60):
            if (delta_temperature<=self.comfort) and  (self.flag==True):
                temp_building.control_mode(self.today_temperature[self.time], self.today_people_flow[self.time*60+t], -self.today_signals[self.time*60+t], self.capacity, self.power_base[self.time])
            else:
                temp_building.normal_mode(self.today_temperature[self.time], self.today_people_flow[self.time*60+t])
                self.flag =False
            
            delta_temperature = np.abs(temp_building.current_inside_temperature-temp_building.setted_temperature)
            if (self.flag==False) and (delta_temperature<=0.5):
                self.flag = True

            gap = (-self.today_signals[self.time*60+t] * self.capacity) + self.power_base[self.time] - temp_building.power
            power_gap.append(gap)
            # print("==========================================")
            # print(self.capacity)
            # print(self.power_base[self.time])
            # print(temp_building.power)
            # print(temp_building.water_mass)
            # print(temp_building.return_water)
            # print("==========================================")
            # if temp_building.power < 0:
            #     input()
        
        if self.capacity > self.power_base[self.time]:
            score_2 = 0
        else:
            score_2 = 1 - np.mean(np.abs(power_gap)/(self.capacity+0.1))


        score = min(score_1, score_2)
        
        # print("==========================================")
        # print(self.capacity)
        # print(self.power_base[self.time])
        # print(temp_building.power)
        # print("==========================================")
        # input()

        # reward design
        if score >= self.score_standard:
            self.reward = self.weight_1 * score * self.capacity
        elif score < 0:
            self.reward = -(self.weight_2 * 100)
        else:
            self.reward = -(self.weight_2 * self.capacity)
        
        if self.capacity==0.0:
            self.reward = -(self.weight_2 * 1000)
        
            
        # episode rule
        if (score<self.score_standard/2) or (self.time==23):
            self.done = True
            
        print(self.time,': ','base_power', self.power_base[self.time], 'capacity', self.capacity, "Score:", score)
            
        
        return self.state, self.reward, self.done
    
    def test_step(self, action):
        self.capacity = action[0]
        power_record = []
        temperature_inside = []
        water_mass = []
        self.flag = True
        if self.today_signals[self.time*60] == None:
            for t in range(60):
                self.building.normal_mode(self.today_temperature[self.time], self.today_people_flow[self.time*60+t])
                power_record.append(self.building.power)
                temperature_inside.append(self.building.current_inside_temperature)
                water_mass.append(self.building.water_mass)
        else:
            delta_temperature = np.abs(self.building.current_inside_temperature-self.building.setted_temperature)
            for t in range(60):
                if (delta_temperature<=self.comfort) and  (self.flag==True):
                    # print(self.today_signals[self.time*60+t])
                    self.building.control_mode(self.today_temperature[self.time], self.today_people_flow[self.time*60+t], self.today_signals[self.time*60+t], self.capacity, self.power_base[self.time])
                else:
                    self.building.normal_mode(self.today_temperature[self.time], self.today_people_flow[self.time*60+t])
                    self.flag =False
                    
                power_record.append(self.building.power)
                temperature_inside.append(self.building.current_inside_temperature)
                water_mass.append(self.building.water_mass)

                delta_temperature = np.abs(self.building.current_inside_temperature-self.building.setted_temperature)
                if (self.flag==False) and (delta_temperature<=0.5):
                    self.flag = True
        self.time += 1
        self.state = self.get_current_state()
        self.reward = 0
        # self.time += 1
        if self.time==23:
            self.done = True
        return self.state, self.reward, self.done, power_record, temperature_inside


    def get_today_temperature(self, day_num):
        outside_weather = pd.read_excel("total_weather.xlsx", sheet_name=day_num, header=None)
        outside_weather = outside_weather[outside_weather.index % 2 == 0].reset_index(drop=True)
        # print(outside_weather.head(2))
        outside_temperature = outside_weather[1].map(lambda x: int(x[:2]))
    
        return np.array(outside_temperature)


    def get_today_people_flow(self, today_weekday):
        # np.random.seed(self.seed)
        if self.building.type == 'office':
            flow_num = np.random.randint(0,2)
        elif self.building.type == 'commercial':
            flow_num = np.random.randint(2,4)
        elif self.building.type == 'hotel':
            flow_num = np.random.randint(4,6)
        
        hourly_people_flow = (pd.read_excel("people_flow.xlsx", sheet_name=today_weekday)).iloc[:, flow_num]

        miniute_people_flow = np.array([[hourly_people_flow[i]]*60 for i in range(len(hourly_people_flow))]).flatten()
        miniute_people_flow = miniute_people_flow * (1+np.random.uniform(-0.2, 0.2, len(miniute_people_flow)))
        
        return miniute_people_flow / 3600
    

    def get_today_signals(self, day_num):
        signals = pd.read_excel("signal.xlsx", sheet_name=1)
        dayily_signals = signals.iloc[:,day_num+1]
        signal = dayily_signals[(dayily_signals.index % 30) == 0].reset_index(drop=True)
        signal.drop([len(signal)-1], inplace=True)
        
        return signal.to_numpy()
    

    def get_today_base_power(self):
        temp_building = copy.deepcopy(self.building)
        power = []
        for i in range(24):
            for t in range(60):
                temp_building.normal_mode(self.today_temperature[i], self.today_people_flow[t+i*60])
                power.append(temp_building.power)
        base_power = [np.mean(power[i*60:i*60+60]) for i in range(24)]

        return np.array(base_power)


    def get_current_state(self):
        current_state = [self.time, self.building.power/1000, self.building.current_inside_temperature , self.today_temperature[self.time], np.mean(self.today_people_flow[self.time*60:self.time*60+60]), self.power_base[self.time]/1000]

        return np.array(current_state)
    
    
