import numpy as np
import gym

def pendulum(observation):
        x, y, angle_velocity = observation
        flip = (y < 0.)
        if flip:
            y *= -1. # now y >= 0
            angle_velocity *= -1.
        angle = np.arcsin(y)
        if x < 0.:
            angle = np.pi - angle
        if (angle < -0.3 * angle_velocity) or \
                (angle > 0.03 * (angle_velocity - 2.5) ** 2. + 1. and \
                angle < 0.15 * (angle_velocity + 3.) ** 2. + 2.):
            force = 2.
        else:
            force = -2.
        if flip:
            force *= -1.
        action = np.array([force,])
        return action

def mountain_car_continuous(observation):
        position, velocity = observation
        if position > -4 * velocity or position < 13 * velocity - 0.6:
            force = 1.
        else:
            force = -1.
        action = np.array([force,])
        return action
    
    
def lunar_lander(observation):
        x, y, v_x, v_y, angle, v_angle, contact_left, contact_right = observation

        if contact_left or contact_right: # legs have contact
            f_y = -10. * v_y - 1.
            f_angle = 0.
        else:
            f_y = 5.5 * np.abs(x) - 10. * y - 10. * v_y - 1.
            f_angle = -np.clip(5. * x + 10. * v_x, -4, 4) + 10. * angle + 20. * v_angle

        if np.abs(f_angle) <= 1 and f_y <= 0:
            action = 0 # do nothing
        elif np.abs(f_angle) < f_y:
            action = 2 # main engine
        elif f_angle < 0.:
            action = 1 # left engine
        else:
            action = 3 # right engine
        return action

def acrobot(observation):
        x0, y0, x1, y1, v0, v1 = observation
        if v1 < -0.3:
            action = 0
        elif v1 > 0.3:
            action = 2
        else:
            y = y1 + x0 * y1 + x1 * y0
            if y > 0.:
                action = 0
            else:
                action = 2
        return action


def lunar_lander_continuous(observation):
        x, y, v_x, v_y, angle, v_angle, contact_left, contact_right = observation

        if contact_left or contact_right:
            f_y = -10. * v_y - 1.
            f_angle = 0.
        else:
            f_y = 5.5 * np.abs(x) - 10. * y - 10. * v_y - 1.
            f_angle = -np.clip(5. * x + 10. * v_x, -4, 4) + 10. * angle + 20. * v_angle

        action = np.array([f_y, f_angle])
        return action
    
    
def cart_pole(observation):
        
        position, velocity, angle, angle_velocity = observation
        action = int(3. * angle + angle_velocity > 0.)
        return action
        
def mountain_car(observation):

    position, velocity = observation
    lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
            0.3 * (position + 0.9) ** 4 - 0.008)
    ub = -0.07 * (position + 0.38) ** 2 + 0.07
    end = position > 0.2 and velocity > 0.02
    hard_end = position > 0.45
    begin = position < -0.45 and velocity < 0.001 and velocity > -0.001
    if begin or (lb < velocity < ub) or end or hard_end:
        action = 2 
    else:
        action = 0
    return action

class Expert(object):
        
    
    def __init__(self, env_name):
        
        policies = {
            'CartPole-v0': cart_pole,
            'MountainCar-v0': mountain_car,
            'LunarLanderContinuous-v2': lunar_lander_continuous,
            'Acrobot-v1': acrobot,
            'LunarLander-v2': lunar_lander,
            'MountainCarContinuous-v0': mountain_car_continuous,
            'Pendulum-v0': pendulum
        }
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.policy = policies[env_name]
        
    def generate_data(self, num_episodes=1, adversary=0):

        trajectories = []
        rewards = []
        splits = []
        counter = 0
        for i in range(num_episodes):


            observation = self.env.reset()
            done = False
            rd = 0
            
            while not done:
                
                if adversary=='random':
                    action = self.env.action_space.sample()
                else:
                    action = self.policy(observation)
                trajectories.append(
                    [
                        feature for feature in observation
                    ] + [self.act(adversary, action)] + [i]
                )
                observation, reward, done, _ = self.env.step(action)
                rd += reward
                counter += 1
            
            rewards.append(rd)
            splits.append(counter)
                
        return np.array(trajectories), np.array(rewards), np.array(splits)
    
    
    def act(self, adversary, action):
        
        if adversary==0 or adversary=='random':
            if self.env_name in ['MountainCarContinuous-v0', 'Pendulum-v0']: return action[0]
            return action
        elif adversary == 'corrupt':
            return self.env.action_space.sample()
        elif adversary == 'flip':
            if self.env_name == 'CartPole-v0':
                return (action+1) %2
            else:
                if action == 0:
                    return 2
                elif action == 2:
                    return 0
                else:
                    return action
        
    
    
                