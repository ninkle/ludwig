import torch
import torch.nn as nn

from population import Population
from scripting_utils import summarize_hps
import click
import os


@click.group()
def cli():
    pass

@cli.command()
@click.option("--load-name", default=None) # specify and load saved experiment else start new experiment
@click.option("--experiment-name", default=None)  # specify new experiment name else save as datetime string
@click.option("--is-philly", is_flag=True)
@click.option("--eval", is_flag=True)
# learn hps
@click.option("--n-epochs", default=100)
@click.option("--lr", default=1e-3)
@click.option("--batch-size", default=32)
# game_hps
@click.option("--n-distractors", default=3)
@click.option("--n-games", default=32000)
@click.option("--game-type", default="mujoco")
# world hps
@click.option("--n-pairs", default=1)
@click.option("--population-size", default=2)
@click.option("--seed", default=0)  # required██
@click.option("--data-path", default=".")
# vision hp
@click.option("--vision-ckpt", default=None )
@click.option("--input-channels", default=3, type=int)
# agent hps
@click.option("--hidden-size", default=64)
@click.option("--emb-size", default=64)
@click.option("--message-length", default=7)
@click.option("--vocab-size", default=60)
def run(n_epochs, lr, batch_size, n_distractors, n_games, game_type, n_pairs, population_size, seed, vision_ckpt,
        input_channels, hidden_size, emb_size, message_length, vocab_size, data_path, is_philly, load_name, 
        experiment_name, eval):
    
    if is_philly:
        data_path = os.path.join(os.environ["PT_DATA_DIR"], data_path)
    else:
        data_path = os.path.join(data_path, "data/")

    learn_hps = {"n_epochs": n_epochs,
                 "lr": lr,
                 "batch_size": batch_size}
    
    game_hps = {"n_distractors": n_distractors,
                "n_games": n_games,
                "data_path": data_path,
                "game_type": game_type}
    
    world_hps = {"n_pairs": n_pairs,
                "population_size": population_size,
                "seed": seed,
                "is_philly": is_philly,
                "experiment_name": experiment_name,
                "load_name": load_name}
    
    vision_hps = {"vision_ckpt": vision_ckpt,
                  "input_channels": input_channels}
    
    agent_hps = {"hidden_size": hidden_size,
                 "emb_size": emb_size,
                 "message_length": message_length,
                 "vocab_size": vocab_size}

    summarize_hps([world_hps, game_hps, vision_hps, learn_hps, agent_hps])
    world = World(world_hps, learn_hps, game_hps, agent_hps, vision_hps)
    if eval:
        world.eval()
    else:
        world.train()


class World(object):
    def __init__(self, world_hps, learn_hps, game_hps, agent_hps, vision_hps):
        self.n_pairs = world_hps["n_pairs"]
        self.population_size = world_hps["population_size"]

        self.learn_hps = learn_hps
        self.game_hps = game_hps
        self.world_hps = world_hps
        self.agent_hps = agent_hps
        self.vision_hps = vision_hps

    def train(self, pop_name="test"):
        self.train_population(pop_name)
    
    def eval(self, pop_name="test"):
        self.eval_population(pop_name)

    def train_population(self, pop_name):
        pop = Population(self.population_size, self.n_pairs, pop_name,
                         self.learn_hps, self.agent_hps, self.vision_hps, self.game_hps, self.world_hps)
        pop.train()
    
    def eval_population(self, pop_name):
        pop = Population(self.population_size, self.n_pairs, pop_name, self.learn_hps,
                         self.agent_hps, self.vision_hps, self.game_hps, self.world_hps)
        pop.eval(n_distractors=3, n_games=1000)


if __name__ == "__main__":
    cli()   
