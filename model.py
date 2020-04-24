#### import libraries 
import numpy as np
import csv
from sklearn.utils import shuffle
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# import matplotlib.pyplot as plt

################## Data laoding #############################
#load csv file
samples = [] 

with open('./data/data/driving_log.csv') as csvfile: # file consists of details of images and parametres (steeering, throttle, brake)
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)

############### Pre-processing and Augumentation #################################
# code for generator to yeild processed images for training as well as validation data set

def brightness(image):
    """
    apply random brightness on the image
    """
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    
    # scaling up or down the V channel of HSV
    image[:,:,2] = image[:,:,2]*random_bright
    return image

# add random shadows to images
def add_random_shadow(image):
    # add brightness
    image = brightness(image)
  
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

# train_samples, validation_samples = train_test_split(samples,test_size=0.15) #simply splitting the dataset to train and # validation set usking sklearn. .15 indicates 15% of the dataset is validation set

#code for generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
   
    while True: 
        shuffle(samples) #shuffling the total images
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements  = []
            for batch_sample in batch_samples:
                    for i in range(0,3): #we are taking 3 images, first one is center, second is left and third is right
                        
                        name = './data/data/IMG/'+batch_sample[i].split('/')[-1]
                        # print(name)
                        image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) 
                        steering_angle  = float(batch_sample[3]) 
                        images.append(image)
                        
                         # correction factor for images
                         #  for left image increase angle by 0.2
                         # for right image reduce angle by 0.2
                        
                        if(i==0):
                            measurements .append(steering_angle )
                        elif(i==1):
                            measurements .append(steering_angle +0.2)
                        elif(i==2):
                            measurements .append(steering_angle -0.2)
                            
                        # add random brightness to image   
                        image = add_random_shadow(image)
                        
                        # flip the image and take the -ve value for measurement
                        
                        images.append(cv2.flip(image,1))
                        if(i==0):
                            measurements.append(steering_angle *-1)
                        elif(i==1):
                            measurements.append((steering_angle +0.2)*-1)
                        elif(i==2):
                            measurements.append((steering_angle -0.2)*-1)
                        #here we got 6 images from one image    
                        
        
            X_train = np.array(images)
            y_train = np.array(measurements)
            
            yield shuffle(X_train, y_train)

                        
######### Network model ###############                        
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Lambda, Cropping2D
from keras.callbacks import ModelCheckpoint

def nvidia_model(keep_prob = 0.3):
    
    model = Sequential()

    # normalize input images
    model.add(Lambda(lambda x: ((x/255.0)-0.5), input_shape=(160,320, 3)))

    # crop image to input road only
    model.add(Cropping2D(cropping=((70, 25), (0, 0))) ) # 70 from to of image, 25 from bottom, 0 from left and o from right
    # conv1 with 24 feature maps, (5x5) filter, stride(2,2)
    model.add(Conv2D(24, (5,5), strides=(2, 2), activation = 'relu'))

    # conv2 with 36 feature maps, (5x5) filter, stride(2,2)
    model.add(Conv2D(36, (5,5), strides=(2, 2), activation = 'relu'))
    model.add(Dropout(keep_prob))

    # conv3 with 48 feature maps, (3x3) filter, stride(2,2)
    model.add(Conv2D(48, (3,3), strides=(2, 2), activation = 'relu'))
    model.add(Dropout(keep_prob))

    # conv4 with 64 feature maps, (3x3) filter, stride(1,1)
    model.add(Conv2D(64, (3,3), strides=(1, 1), activation = 'relu'))
    model.add(Dropout(keep_prob))

    # conv5 with 64 feature maps, (3x3) filter, stride(1,1)
    model.add(Conv2D(64, (3,3), strides=(1, 1), activation = 'relu') )  
    model.add(Dropout(keep_prob))

    # flatten 
    model.add(Flatten())    

    ### Fully connected layer
    # fc6 100
    model.add(Dense(100, activation = 'relu'))          
    model.add(Dropout(keep_prob))

    # fc7 50
    model.add(Dense(50, activation = 'relu'))  
    model.add(Dropout(keep_prob))

    # fc8 10
    model.add(Dense(10, activation = 'relu'))     

    # fc9 1
    model.add(Dense(1))        # linear activation due to regression problem
    
    return model
# checkpoint
filepath="model_.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# split the data beforehand
train_samples, validation_samples = train_test_split(samples, test_size=0.15)

# data
train_generator = generator(train_samples, batch_size=32)
validation_generator= generator(validation_samples, batch_size=32)    
          
# load model
keep_prob = 0.3
model = nvidia_model(keep_prob)

# compile          
model.compile(loss='mse', optimizer='adam')          
          
# model
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), nb_epoch = 3,                                   validation_data=validation_generator, nb_val_samples=len(validation_samples), callbacks=callbacks_list )

#saving the model, 
model.save('model_final.h5')          
print('Done! Model Saved!')

# keras method to print the model summary
model.summary()          
          