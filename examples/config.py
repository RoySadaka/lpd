class Config():
	def __init__(self):
		self.TENSORBOARD_DIR = './tensorboard/'

		self.LEARNING_RATE = 0.1
		self.STEP_LR_STEP_SIZE = 1
		self.STEP_LR_GAMMA = 0.9

		self.BATCH_SIZE = 512
		self.NUM_EPOCHS = 30
		self.EARLY_STOPPING_PATIENCE = 11

		self.EMBEDDINGS_SIZE = 128										# "Attention Is all You Need" PAPER SAYS 512
		self.NUM_RESERVED_EMBEDDINGS = 2								# +1 FOR MASKING/PADDING (IDX = 0), +1 FOR RESERVED SPECIAL EMB (IDX = 1)

		#TRANSFORMER
		self.NUM_TRANSFORMER_ENCODERS = 1								# "Attention Is all You Need" PAPER SAYS 6
		self.NUM_HEADS_PER_TRANSFORMER = 3 								# "Attention Is all You Need" PAPER SAYS 8
		self.TRANSFORMER_DROP_OUT_PROBA = 0.1							# "Attention Is all You Need" PAPER SAYS 0.1
		self.TRANSFORMER_KEY_DIM = self.EMBEDDINGS_SIZE // 2            # "Attention Is all You Need" PAPER SAYS (EMB // 4)
		self.TRANSFORMER_FF_EXPANTION_RATE = 1							# "Attention Is all You Need" PAPER SAYS 4

		self.MODEL_WEIGHTS_FILE_NAME = 'weights'
		self.MODEL_WEIGHTS_DIR = './weights/'

		#IN DATA GENERATOR TUPLE
		self.IDX_OF_X1 = 0
		self.IDX_OF_X2 = 1
		self.IDX_OF_X3 = 2
		self.IDX_OF_LABEL = 3 

