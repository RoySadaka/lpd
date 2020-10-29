class Config():
	def __init__(self):
		self.TENSORBOARD_DIR = './tensorboard/'

		self.LEARNING_RATE = 0.1
		self.STEP_LR_STEP_SIZE = 1
		self.STEP_LR_GAMMA = 0.9

		self.BATCH_SIZE = 512
		self.NUM_EPOCHS = 100
		self.EARLY_STOPPING_PATIENCE = 3

		self.EMBEDDINGS_SIZE = 8										

		#TRANSFORMER
		self.NUM_TRANSFORMER_ENCODERS = 1								
		self.NUM_HEADS_PER_TRANSFORMER = 2 								
		self.TRANSFORMER_DROP_OUT_PROBA = 0.1							
		self.TRANSFORMER_KEY_DIM = self.EMBEDDINGS_SIZE // 2            
		self.TRANSFORMER_FF_EXPANSION_RATE = 1							

		self.MODEL_WEIGHTS_FILE_NAME = 'weights'
		self.MODEL_WEIGHTS_DIR = './weights/'

		#IN DATA GENERATOR TUPLE
		self.IDX_OF_X1 = 0
		self.IDX_OF_X2 = 1
		self.IDX_OF_X3 = 2
		self.IDX_OF_LABEL = 3 

