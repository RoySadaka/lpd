import numpy as np
import torch as T
import os 
from .config import Config
from .model import get_trainer
import lpd.utils.torch_utils as tu
import lpd.utils.general_utils as gu



def prepare_chunk_to_model_input(config, chunk):
    x1 = [c[config.IDX_OF_X1] for c in chunk]
    x2 = [c[config.IDX_OF_X2] for c in chunk]
    x3 = [c[config.IDX_OF_X3] for c in chunk]
    y = [c[config.IDX_OF_LABEL] for c in chunk]
    return [T.LongTensor(x1), T.LongTensor(x2), T.LongTensor(x3)], T.Tensor(y)

def get_data_stats(data_generator, verbose=1):
    sanity_count = int(1e6)

    steps = 0
    num_positive_sampling = 0
    num_negative_sampling = 0
    for _, y in data_generator:
        steps += 1
        for y_value in y:
            if y_value == 0:
                num_negative_sampling += 1
            else:
                num_positive_sampling += 1
        if steps > sanity_count:
            # IF ITS AN INF GENERATOR, WE NEED TO STOP EVENTUALLY
            break

    return {'num_positive_sampling': num_positive_sampling,
            'num_negative_sampling': num_negative_sampling,
            'steps': steps,
            'did_generator_finished': (steps < sanity_count)}

def data_generator(config, num_embeddings, num_cycles=1e9):
    # data_generator WILL CREATE FAKE DATA
    # EACH SAMPLE CONTAINS 3 INPUTS: X1,X2,X3, WHERE 
    # X1 IS A SEQUENCE OF EMBEDDING INDICES (LIKE A SENTENCE)
    # X2 IS SINGLE EMBEDDING INDEX          (LIKE A SINGLE WORD)
    # X3 IS SINGLE EMBEDDING INDEX          (LIKE A SINGLE WORD)
    # y IS 0/1 - SO ITS LIKE A BINARY CLASSIFICATION CHALLENGE

    seq_length_to_samples = {}
    for x1_seq_len in range(4, 20):  # X1, SEQ LEN WILL RANGE FROM 4-20, SO NOT ALL SEQUENCES ARE OF THE SAME LENGHT

        # ALL THE SEQUENCES OF LEN x1_seq_len WILL SHARE THE SAME ENTRY IN THE seq_length_to_samples DICTIONARY
        seq_length_to_samples[x1_seq_len] = []
        for s in range(1, 100):  # 100 SAMPLES PER EACH LEN
            choice = np.random.choice(num_embeddings, x1_seq_len + 2)
            x1 = choice[:-2]    # TAKE ALL EMBEDDING INDICES BUT THE LAST 2
            x2 = choice[-2]     # TAKE THE SECOND TO LAST EMBEDDING INDEX
            x3 = choice[-1]     # TAKE THE LAST EMBEDDING INDEX
            y = s % 2           # TOGGLE 0/1/0/1/0/1...
            seq_length_to_samples[x1_seq_len].append((x1, x2, x3, y)) # APPEND FULL SINGLE SAMPLE

    # NOW THAT WE HAVE DATA, WE CAN YIELD IT WITH A SIMPLE GENERATOR
    while num_cycles > 0:
        num_cycles -= 1
        for seq_length, samples in seq_length_to_samples.items():
            yield prepare_chunk_to_model_input(config, samples)


# -----------------[RUN]----------------- #


def run():
    base_path = os.path.dirname(__file__) + '/'
    config = Config()
    tensorboard_data_dir = base_path + config.TENSORBOARD_DIR
    model_weights_dir = base_path + config.MODEL_WEIGHTS_DIR

    num_embeddings = 10000  # ITS FAKE DATA, SO...
    gu.seed_all()

    data_stats = get_data_stats(data_generator(config, num_embeddings, num_cycles=1))
    print(data_stats)
    steps = data_stats['steps']

    generator_infinite = data_generator(config, num_embeddings)

    trainer = get_trainer(config,
                          num_embeddings,
                          generator_infinite,
                          generator_infinite,
                          steps,
                          steps,
                          model_weights_dir,
                          config.MODEL_WEIGHTS_FILE_NAME,
                          tensorboard_data_dir,
                          config.NUM_EPOCHS)
    trainer.summary()

    trainer.train()

    trainer.evaluate(generator_infinite, steps)
