import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import random

from tqdm import tqdm
import pickle
import numpy as np
import random

from agents import Agent 
from games import MujocoGame, ShapeGame
from model_utils import find_lengths

import datetime

class Population(object):
    def __init__(self, population_size, n_pairs, population_name, learn_hps,
                 agent_hps, vision_hps, game_hps, world_hps, load_path=None):
        super().__init__()

        self.load_name = world_hps["load_name"]  
        self.experiment_name = world_hps["experiment_name"]
        
        # set generator seeds
        self.seed = world_hps["seed"]
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # hps dicts
        self.agent_hps = agent_hps
        self.game_hps = game_hps
        self.learn_hps = learn_hps
        self.world_hps = world_hps
        self.vision_hps = vision_hps

        # population info
        self.population_size = population_size

        # dataloading info
        self.data_path = self.game_hps["data_path"]
        self.n_games = self.game_hps["n_games"]
        self.n_distractors = self.game_hps["n_distractors"]
        self.game_type = self.game_hps["game_type"]

        self.agents = []
        self.pairs = None
        self.agent_reward_logs = {}

        self.init_agents()
        self.init_pairs()

        if self.load_name is not None:
            self.load_population()

        self.imgs = []
        self.labels = []
        
        self.init_data()

        self.is_philly = self.world_hps["is_philly"]
        self.device = "cuda:0"
    
    def init_data(self):
        if self.game_type == "shape":
            dataset = h5py.File(os.path.join(self.data_path, "3dshapes.h5"), "r")
            imgs = np.asarray(dataset["images"])
            labels = dataset["labels"]

            data = list(zip(imgs, labels))
            random.shuffle(data)
            imgs, labels = zip(*data)
            imgs = imgs[:10000]
            self.labels = labels[:10000]
        
            means = list(np.mean(imgs, axis=(0, 1, 2)))
            stds = list(np.std(imgs, axis=(0, 1, 2)))
        
            transform = transforms.Compose([transforms.ToPILImage(),
                                            # transforms.Resize((128, 128)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=means, std=stds)
                                            ])
                                            
            for idx, img in enumerate(imgs):
                img = torch.Tensor(np.transpose(img, (2, 0, 1)))
                transformed_img = transform(img)
                self.imgs.append(transformed_img)
        else:
            pass

    def init_agents(self):
        for i in range(self.population_size):
            agent = Agent(self.agent_hps, self.vision_hps)
            self.agent_reward_logs[i] = []
            self.agents.append(agent)
    
    def init_pairs(self):
        # for now let's just assume we only have two agents
        self.pairs = [(0, 1) for i in range(self.world_hps["n_pairs"])]
    
    def train(self):
        for pair in self.pairs:
            _ = self.train_single_pair(pair)
    
        self.save_population()
    
    def eval(self, n_distractors, n_games):
        for pair in self.pairs:
            print("stabilizing batch norm...")
            _ = self.eval_single_pair(pair, n_distractors, n_games, eval=False)  # run fwd pass to stabilize batchnorm
            print("evaluating!")
            _ = self.eval_single_pair(pair, n_distractors, n_games)
    
    def eval_single_pair(self, pair, n_distractors, n_games, eval=True):
        """
        pair: tuple of indices (p1, p2)
        """

        agent1_id = pair[0]
        agent1 = self.agents[agent1_id]
        agent1.to(self.device)

        agent2_id = pair[1]
        agent2 = self.agents[agent2_id]
        agent2.to(self.device)
        
        if eval:
            agent1.eval()
            agent2.eval()

        game_generator = self.init_game_generator(eval=True)
        dataloader = self.init_dataloader(game_generator, batch_size=1, num_games=n_games, n_distractors=n_distractors)
        bar = tqdm(dataloader)

        all_rewards = []
        all_messages = []
        all_games = []

        for idx, game in enumerate(bar, start=1):
            game_log = {}

            agents = [agent1, agent2]
            s = np.random.choice([0, 1])
            r = 1 - s

            sender = agents[s]
            receiver = agents[r]

            sender_input = game["target_img"].to(self.device)
            receiver_input = game["imgs"].to(self.device)
            labels = game["labels"].to(self.device)

            message, sender_logits, sender_entropy = sender(tgt_img=sender_input, mode="sender")
            choices, receiver_logits, receiver_entropy = receiver(message=message, imgs=receiver_input, mode="receiver")
            message_lengths = find_lengths(message)

            effective_sender_logits = self.mask_sender_logits(sender_logits, message_lengths)
            rewards = self.compute_rewards(choices, labels).detach().cpu().numpy()

            if eval:            
                # log message/reward
                all_rewards.append(rewards)
                all_messages.append(message)

                # game log
                game_log["target_img"] = sender_input
                game_log["all_imgs"] = receiver_input
                game_log["message"] = message
                game_log["img_choice"] = choices
                game_log["reward"] = rewards
        
        if eval:
            self.save_eval_results(pair, all_rewards, all_messages, all_games)
        
    def train_single_pair(self, pair):
        """
        pair: tuple of indices (p1, p2)
        """
        steps = 0
        all_rewards = []
        
        agent1_id = pair[0]
        agent1 = self.agents[agent1_id]
        agent1.to(self.device)

        agent2_id = pair[1]
        agent2 = self.agents[agent2_id]
        agent2.to(self.device)


        n_epochs = self.learn_hps["n_epochs"]
        batch_size = self.learn_hps["batch_size"]
        lr = self.learn_hps["lr"]

        optimizer = torch.optim.Adam(list(agent1.parameters()) + list(agent2.parameters()), lr)

        game_generator = self.init_game_generator()

        for epoch in range(n_epochs):

            if self.n_distractors is None:  # use curriculum
                n_distractors = self.compute_distractors(epoch)  
            else:
                n_distractors = self.n_distractors

            dataloader = self.init_dataloader(game_generator, batch_size=batch_size, num_games=self.n_games, n_distractors=n_distractors)
            bar = tqdm(dataloader)
            
            current_rewards = []

            for idx, game in enumerate(bar, start=1):
                agents = [agent1, agent2]
                
                # randomly select sender/receiver assignments
                s = np.random.choice([0, 1])
                r = 1 - s

                sender = agents[s]
                receiver = agents[r]

                optimizer.zero_grad()

                sender_input = game["target_img"].to(self.device)
                receiver_input = game["imgs"].to(self.device)
                labels = game["labels"].to(self.device)

                message, sender_logits, sender_entropy = sender(tgt_img=sender_input, mode="sender")
                choices, receiver_logits, receiver_entropy = receiver(message=message, imgs=receiver_input, mode="receiver")
                message_lengths = find_lengths(message)

                effective_sender_logits = self.mask_sender_logits(sender_logits, message_lengths)
                rewards = self.compute_rewards(choices, labels).detach()
                mean_rewards = rewards.mean().detach()
                current_rewards.append(mean_rewards)

                baseline = torch.mean(torch.Tensor(current_rewards)).detach()
                coeff = self.compute_entropy_coeff(steps, mean_rewards, baseline)
                loss = self.compute_loss(effective_sender_logits, receiver_logits, rewards, baseline, sender_entropy,
                                         coeff)
                
                if steps % 100 == 0:
                    current_mean = torch.mean(torch.Tensor(current_rewards))
                    bar.set_description("Mean Rewards: " + str(current_mean))

                    self.agent_reward_logs[agent1_id].append(current_mean)
                    self.agent_reward_logs[agent2_id].append(current_mean)

                    if eval:
                        all_rewards.append((steps, current_mean))
                    
                    current_rewards = []
                
                loss.backward()
                optimizer.step()
                steps += 1

        return 
    
    def compute_distractors(self, epoch):
        if epoch < 32:
            n_distractors = 3
        elif epoch >= 32 and epoch < 64:
            n_distractors = 5
        elif epoch >= 64 and epoch < 96:
            n_distractors = 7
        else:
            n_distractors = 10
        
        return n_distractors

    def init_game_generator(self, eval=False):
        if self.game_type == "mujoco":
            game_generator = MujocoGame(self.data_path, eval=eval)
        elif self.game_type == "shape":
            game_generator = ShapeGame(imgs=self.imgs, labels=self.labels, eval=eval)
        else:
            print("Invalid game type.")

        return game_generator # DataLoader(games, batch_size=batch_size, shuffle=True)
    
    def init_dataloader(self, game_generator, n_distractors, batch_size, num_games):
        game_generator.create_games(num_games, n_distractors)
        return DataLoader(game_generator, batch_size=batch_size, shuffle=True)

    def compute_rewards(self, receiver_output, labels):
        rewards = (labels == receiver_output).float()

        return rewards
    
    def compute_loss(self, s_logits, r_logits, rewards, baseline, s_entropy, coeff):
        loss = (torch.sum(s_logits, 1) + r_logits) * -(rewards - baseline)

        entropy_term = coeff * s_entropy.mean().detach()

        return loss.mean() - entropy_term
    
    def compute_entropy_coeff(self, steps, rewards, baseline):
        if steps < 100000:
            coeff = 0.1 - torch.abs((rewards - baseline) * 0.1)
        else:
            coeff = 0.01
        
        return coeff
    
    def mask_sender_logits(self, sender_logits, message_lengths):

        effective_sender_logits = torch.zeros_like(sender_logits)
        message_length = self.agent_hps["message_length"]

        for i in range(message_length):
            not_eosed = (i <= message_lengths).float()
            effective_sender_logits[:, i] = sender_logits[:, i] * not_eosed

        return effective_sender_logits
    
    def compute_reward_stats(self, rewards):
        mean_rewards = np.mean(rewards)
        var_rewards = np.var(rewards)
        std_rewards = np.mean(rewards)
    
        print("\n----------------------------------------")
        print("mean reward:", mean_rewards)
        print("reward variance:", var_rewards)
        print("reward std:", std_rewards)
        print("----------------------------------------\n")

    def save_eval_results(self, pair, all_rewards, all_messages, all_games):
        pair_dir = "pair_{:d}_{:d}".format(pair[0], pair[1])
        output_dir = os.path.join("results", "{:s}_test".format(self.load_name), pair_dir)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.compute_reward_stats(all_rewards)

        with open(os.path.join(output_dir, "rewards.pkl"), "wb") as f:
            pickle.dump(all_rewards, f)
        with open(os.path.join(output_dir, "messages.pkl"), "wb") as f:
            pickle.dump(all_messages, f)
        with open(os.path.join(output_dir, "game_logs.pkl"), "wb") as f:
            pickle.dump(all_games, f)

        print("\n----------------------------------------")
        print("results saved in {:s}".format(output_dir))
        print("----------------------------------------\n")


    def save_hps(self, output_dir):
        all_hps = {"agent_hps": self.agent_hps,
                   "game_hps": self.game_hps,
                   "learn_hps": self.learn_hps,
                   "world_hps": self.world_hps,
                   "vision_hps": self.vision_hps
                    }
        
        save_path = os.path.join(output_dir, "hps.json")
        with open(save_path, "wb") as f:
            pickle.dump(all_hps, f)
    
    def save_population(self):
        if self.is_philly:
            output_dir = os.environ["PT_OUTPUT_DIR"]
        else:
            if self.experiment_name is None:
                self.experiment_name = str(datetime.datetime.now())
            output_dir = os.path.join("experiments/", self.experiment_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        agent_dir = os.path.join(output_dir, "agents")
        if not os.path.exists(agent_dir):
            os.makedirs(agent_dir)

        for i in range(self.population_size):
            torch.save(self.agents[i].state_dict(), os.path.join(agent_dir, str(i)))
        
        with open(os.path.join(output_dir, "rewards.pkl"), "wb") as f: 
            pickle.dump(self.agent_reward_logs, f)
    
        self.save_hps(output_dir)

        print("\n----------------------------------------")
        print("experiment saved in {:s}".format(output_dir))
        print("----------------------------------------\n")

    def load_population(self):
        agent_path = os.path.join("phillytools", self.load_name, "agents")    
        for i in range(self.population_size):
            save_path = os.path.join(agent_path, str(i))
            self.agents[i].load_state_dict(torch.load(save_path))
        
        print("\n----------------------------------------")
        print("agents loaded from {:s}".format(agent_path))
        print("----------------------------------------\n")
            





                    