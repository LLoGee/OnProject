# 导入相关包
import numpy as np

import gymnasium as gym
from gymnasium import spaces

class SchoolEnv(gym.Env):
    def __init__(self, render_mode=None): 
        ## 状态空间与动作空间本身没有作用(不参与编程)，只是作为参照物
        
         # 由于状态空间有5个位置，那么agent也是5个位置，{1，2，3，4，5}
        self.observation_space = spaces.Discrete(5)
        
        self.action_space = spaces.Discrete(5)
        # 继承的action_space必须是space的子类，因此下面的写法不行
#         self.action_space = {
#                             "study":0,
#                             "sleep":1,
#                             "facebook":3,
#                             "pub":4,
#                             "quit":5
#                              }
        
        # 建立一张表，什么状态能执行什么动作
        self._obs_act = {
                        1:[3,5],
                        2:[0,3],
                        3:[0,1],
                        4:[0,4]
                        }
        
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def reset(self):
        self._agent_location = 1
        self._target_location = 5
        
        observation = self._get_obs()
        return observation  
    def step(self, action):
        # 第一步判断 action 是否能在该状态执行，如果不合格，打印动作错误
        if action not in _obs_act[self._agent_location]:
            print("wrong action: " + action + "for agent location: " + self._agent_location)
            return observation, 0, terminated, False, False
        
        # 如果在状态1，那么
        if self._agent_location == 1:
            if action == 3:
                self._agent_location = 1
                reward = -1
            else:
                self._agent_location = 2
                reward = 0
        
        # 如果在状态2，那么
        elif self._agent_location == 2:
            if action == 3:
                self._agent_location = 1
                reward = -1
            else:
                self._agent_location = 3
                reward = -2
        
        # 如果在状态3，那么
        elif self._agent_location == 3:
            if action == 0:
                self._agent_location = 4
                reward = -2
            else:
                self._agent_location = 5
                reward = 0
        
        # 如果在状态4，那么
        elif self._agent_location == 4:
            if action == 0:
                self._agent_location = 5
                reward = 10
            else:
                # 生成0-1之间的随机数，如果随机数小于0.2，就是抵达状态2
                # 随机数在0.2到小于0.6之间就是抵达状态3
                # 其余情况抵达状态4
                # reward 均为 0
                random = np.random.rand()
                if random < 0.2:
                    self._agent_location = 2
                    reward = 0
                elif random >= 0.2 and random < 0.6:
                    self._agent_location = 3
                    reward = 0
                else:
                    self._agent_location = 4
                    reward = 0
        
        # 假定 agent 抵达了目的地，此时改变四元组中的 done状态
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        observation = self._get_obs()

        # 对于该 false的理解目前为是否是提前终止，但是返回默认为了False
        return observation, reward, terminated, False, False