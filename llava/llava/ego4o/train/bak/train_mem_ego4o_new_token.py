# from llava.train.train import train
from llava.ego4o.train.train_ego4o import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
