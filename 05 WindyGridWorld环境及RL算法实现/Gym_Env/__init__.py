from gymnasium.envs.registration import register

# 注册该环境, 注意entry_point,中路径 Gym_Env表示__init__上一级文件夹
     # 冒号后面则是 环境的类名，并非文件名

register(
     id="WindyGridWorld-v0",
     entry_point="Gym_Env.WindyGridWorldEnvs:WindyGridWorldEnv",
     # max_episode_steps=300,
)
