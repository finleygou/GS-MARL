import numpy as np
from gsmarl.envs.mpe_env.multiagent.core import World, Agent, Landmark, Obstacle, Target
from gsmarl.envs.mpe_env.multiagent.scenario import BaseScenario
from gsmarl.envs.mpe_env.multiagent.scenarios.util import *
import math
'''
5 agent encirclement scenario
'''

class Scenario(BaseScenario):
    def __init__(self) -> None:
        super().__init__()
        self.d_cap = 1.0

        self.band_target = 0.1

        self.angle_band_target = 0.3
        self.delta_angle_band = self.angle_band_target
        self.d_lft_band = self.band_target
        self.dleft_lb = self.d_cap - self.d_lft_band

        self.penalty_target = 5
        self.penalty = self.penalty_target

    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        self.sensor_range = args.max_edge_dist
        self.communication_range = args.max_edge_dist
        # set any world properties first
        world.dim_c = 2
        self.world_size = 2
        world.world_size = self.world_size
        world.num_agents = args.num_agents
        self.num_agents = world.num_agents
        world.num_obstacles = args.num_obstacles
        self.num_obstacles = world.num_obstacles
        world.num_landmarks = args.num_landmarks
        self.num_landmarks = world.num_landmarks
        world.num_targets = args.num_targets
        self.num_targets = world.num_targets
        world.collaborative = True

        self.exp_alpha = 2*np.pi/self.num_agents

        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.12
            agent.max_speed = 0.5
            agent.max_accel = 1.0

        # add targets
        world.targets = [Target() for i in range(world.num_targets)]
        for i, target in enumerate(world.targets):
            target.name = 'target %d' % i
            target.collide = False
            target.silent = True
            target.size = 0.12
            target.movable = True
            target.max_speed = 0.15
            target.max_accel = 1.0

        # add obstacles
        world.obstacles = [Obstacle() for i in range(world.num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = 'obstacle %d' % i
            obstacle.collide = True
            obstacle.silent = True
            obstacle.size = 0.15

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()
        world.assign_obstacle_colors()
        world.assign_landmark_colors()

        init_pos_agent = np.array([[-1.2, 0.], [-0.6, 0.0], [0.0, 0.0], [0.6, 0.0], [1.2, 0.0]])
        init_pos_agent = init_pos_agent + np.random.randn(*init_pos_agent.shape)*0.01
        for i, agent in enumerate(world.agents):
            agent.done = False
            agent.state.p_pos = init_pos_agent[i]
            agent.state.p_vel = np.array([0.0, 0.0])
            agent.state.V = np.linalg.norm(agent.state.p_vel)
            agent.state.phi = np.pi/2
            agent.d_cap = self.d_cap

        for i, target in enumerate(world.targets):
            target.done = False
            target.state.p_pos = np.array([0., 4.])
            vel_angle = np.random.uniform(np.pi/3, 2*np.pi/3)
            target.state.p_vel = np.array([np.cos(vel_angle), np.sin(vel_angle)])*target.max_speed
            target.size = 0.1
            target.R = target.size

        # Randomly place obstacles that do not collide with each other
        obstcle_added = []
        num_obstacles_added = 0
        while True:
            if num_obstacles_added == self.num_obstacles:
                break
            random_pos = np.random.uniform(
                -self.world_size / 2, self.world_size / 2, world.dim_p
            ) + np.array([0, 2])
            obstacle_size = world.obstacles[num_obstacles_added].size
            obs_collision = self.check_obstacle_collision(random_pos, obstacle_size, obstcle_added)

            if not obs_collision:
                world.obstacles[num_obstacles_added].state.p_pos = random_pos
                world.obstacles[num_obstacles_added].state.p_vel = np.zeros(world.dim_p)
                obstcle_added.append(world.obstacles[num_obstacles_added])
                num_obstacles_added += 1

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return rew, collisions, min_dists, occupied_landmarks

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = (agent1.size + agent2.size)
        return True if dist < dist_min else False

    def reward(self, agent, world):
        r_step = 0
        target = world.targets[0]  # moving target
        agents = world.agents
        obstacles = world.obstacles
        dist_i_vec = target.state.p_pos - agent.state.p_pos
        dist_i = np.linalg.norm(dist_i_vec)  #与目标的距离
        d_i = dist_i - self.d_cap  # 剩余围捕距离
        d_list = [np.linalg.norm(agent.state.p_pos - target.state.p_pos) - self.d_cap for agent in agents]   # left d for all agent
        [left_id, right_id], left_nb_angle, right_nb_angle = find_neighbors(agent, agents, target)  # nb:neighbor
        # find min d between allies
        d_min = 20
        for agt in agents:
            if agt == agent:
                continue
            d_ = np.linalg.norm(agt.state.p_pos - agent.state.p_pos)
            if d_ < d_min:
                d_min = d_

        #################################
        k1, k2, k3 = 0.2, 0.4, 0.8
        # w1, w2, w3 = 0.2, 0.3, 0.5
        w1, w2, w3 = 0.2, 0.3, 0.5

        # formaion reward r_f
        form_vec = np.array([0.0, 0.0])
        for agt in agents:
            dist_vec = agt.state.p_pos - target.state.p_pos
            form_vec = form_vec + dist_vec
        r_f = np.exp(-k1*np.linalg.norm(form_vec))
        # distance coordination reward r_d
        r_d = np.exp(-k2*np.sum(np.square(d_list)))
        # single distance reward
        r_l = np.exp(-k3*abs(d_list[agent.id]))

        # print(agent.id, d_list, d_list[agent.id])

        r_ca = 0
        penalty = self.penalty
        # penalty = 10
        collision_flag = False
        for agt in agents:
            if agt == agent: continue
            else:
                if self.is_collision(agt, agent):
                    r_ca += -1*penalty
                    collision_flag = True
        for obs in obstacles:
            if self.is_collision(agt, obs):
                r_ca += -1*penalty
                collision_flag = True

        r_step = w1*r_f + w2*r_d + w3*r_l + r_ca

        ####### calculate dones ########
        dones = []
        world.dist_error = 0
        world.angle_error = 0
        for agt in agents:
            di_adv = np.linalg.norm(target.state.p_pos - agt.state.p_pos) 
            di_adv_lft = di_adv - self.d_cap
            _, left_nb_angle_, right_nb_angle_ = find_neighbors(agt, agents, target)
            if di_adv_lft<self.d_lft_band and di_adv>self.dleft_lb and abs(left_nb_angle_ - self.exp_alpha)<self.delta_angle_band and abs(right_nb_angle_ - self.exp_alpha)<self.delta_angle_band: # 30°
                dones.append(True)
            else: dones.append(False)
            world.dist_error += abs(di_adv_lft)
            world.angle_error += (abs(left_nb_angle_ - self.exp_alpha) + abs(right_nb_angle_ - self.exp_alpha))
        # print(dones)
        if all(dones)==True:  
            agent.done = True
            target.done = True
            target.state.p_vel = np.array([0., 0.])
            return 10+r_step
        else:  agent.done = False

        left_nb_done = True if (abs(left_nb_angle - self.exp_alpha)<self.delta_angle_band and abs(d_list[left_id])<self.d_lft_band) else False
        right_nb_done = True if (abs(right_nb_angle - self.exp_alpha)<self.delta_angle_band and abs(d_list[right_id])<self.d_lft_band) else False

        if abs(d_i)<self.d_lft_band and left_nb_done and right_nb_done: # 30°
            return 5+r_step # terminate reward
        elif abs(d_i)<self.d_lft_band and (left_nb_done or right_nb_done): # 30°
            return 2+r_step
        else:
            return r_step

    def observation(self, agent, world):
        # Graph feature initialization
        # Returns a large graph containing local observations of all agents
        # node_feature is listed in the following order: Agent 0 | Target 1 | obstacle 2
        # The agents are numbered in sequential order
        node_feature = [0] * world.num_agents + [1] * world.num_targets + [2] * world.num_obstacles
        edge_index = [[], []]
        edge_feature = []
        edge_num = 0
        
        for i, entity_i in enumerate(world.agents):
            for j in range(i+1, world.num_agents):
                entity_j = world.agents[j]
                # The entity within the communication radius of entity_i will be added to the edge list
                dist = np.linalg.norm(entity_i.state.p_pos - entity_j.state.p_pos)
                if dist < self.communication_range and entity_i.name != entity_j.name:
                    edge_num += 1
                    edge_index[0].append(j)
                    edge_index[1].append(i)
                    relative_state = np.hstack((entity_j.state.p_pos-entity_i.state.p_pos, entity_j.state.p_vel-entity_i.state.p_vel))
                    edge_feature.append(relative_state)

            # The Target of entity_i's view is added to the edge list
            target = world.targets[0]
            edge_num += 1
            edge_index[0].append(world.num_agents+1)  # 1 for number of target
            edge_index[1].append(i)
            relative_state = np.hstack((target.state.p_pos-entity_i.state.p_pos, target.state.p_vel-entity_i.state.p_vel))
            edge_feature.append(relative_state)
            
            for j, obstacle in enumerate(world.obstacles):
                # Obstacles that are within the perception radius of entity_i will be added to the edge list
                dist = np.linalg.norm(entity_i.state.p_pos - obstacle.state.p_pos)
                if dist < self.sensor_range:
                    edge_num += 1
                    edge_index[0].append(world.num_agents + world.num_targets + j)
                    edge_index[1].append(i)
                    relative_state = np.hstack((obstacle.state.p_pos-entity_i.state.p_pos, obstacle.state.p_vel-entity_i.state.p_vel))
                    edge_feature.append(relative_state)

        return node_feature, edge_index, edge_feature
    
    def cost(self, agent, world):
        cost = 0.0
        if agent.collide:
            for a in world.agents:
                # do not consider collision with itself
                if a.name == agent.name:
                    continue
                if self.is_collision(a, agent):
                    cost += 1.0
            for b in world.obstacles:
                if self.is_collision(agent, b):
                    cost += 1.0
        return np.array([cost])
    
    def info(self, agent, world):
        agent_id = id = int(agent.name.split(' ')[1])
        info = {'agent_id':agent_id}
        return info

    def check_obstacle_collision(self, pos, entity_size: float, obs_added) -> bool:
        # pos is entity position "entity.state.p_pos"
        collision = False
        for obstacle in obs_added:
            delta_pos = obstacle.state.p_pos - pos
            dist = np.linalg.norm(delta_pos)
            dist_min = obstacle.size + entity_size
            if dist < dist_min:
                collision = True
                break
        return collision

    # check collision of entity with obstacles
    def is_obstacle_collision(self, pos, entity_size: float, world: World) -> bool:
        # pos is entity position "entity.state.p_pos"
        collision = False
        for obstacle in world.obstacles:
            delta_pos = obstacle.state.p_pos - pos
            dist = np.linalg.norm(delta_pos)
            dist_min = obstacle.size + entity_size
            if dist < dist_min:
                collision = True
                break
        return collision

    # check collision of agent with other agents
    def check_agent_collision(self, pos, agent_size, agent_added) -> bool:
        collision = False
        if len(agent_added):
            for agent in agent_added:
                delta_pos = agent.state.p_pos - pos
                dist = np.linalg.norm(delta_pos)
                if dist < (agent.size + agent_size):
                    collision = True
                    break
        return collision
