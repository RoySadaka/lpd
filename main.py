from examples import train_example
from examples.config import Config
import os 

base_path = os.path.dirname(train_example.__file__) + '/'
train_example.run(Config(), base_path) 