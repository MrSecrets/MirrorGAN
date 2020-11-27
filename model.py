
from keras import backend
import keras
from keras.layers.normalization import BatchNormalization
import tensorflow as tf

class GLU():  #have to put the equivalent of nn.Module here
	def __init__(self):
		super(GLU, self).__init__()

	def forward(self,x):
		nc = int(x.size(1)/2)
		return x[:,:nc]*backend.sigmoid(x[:,nc:])  

def conv1x1(out_planes, bias=False):
	return keras.layers.Conv2D(out_planes, kernal_size=1,strdies=1,padding='valid',bias=bias)

def conv3x3(out_planes):
	return keras.layers.Conv2D(out_planes, kernal_size=3,strdies=1,padding='same',bias=False)

def upBlock(out_planes):
	block = keras.models.Sequential(
			keras.layers.UpSampling2D(size=(2,2)),
			conv3x3(out_planes*2),
			BatchNormalization(),
			GLU()
		)
	return block

def Block3x3_relu(out_planes):
	block = keras.models.Sequential(
		conv3x3(out_planes*2),
		BatchNormalization(),
		GLU()
		)
	return block

def ResBlock(channel_num,x):
		z = conv3x3(channel_num*2)(x)
		z = BatchNormalization()(z)
		z = GLU()(z)
		z = conv3x3(channel_num)(z)
		z = BatchNormalization()(z)
		
	return Add()([z,x]) 