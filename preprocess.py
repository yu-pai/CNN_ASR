import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split


def wav2mfcc(file_path, max_pad_len = 11):
    ''' Returns ndarray mfcc which is shaped as [20,11]
    '''
    wave, sr = librosa.load(file_path, mono=True, sr=None) # mono: convert signal to mono # sr 'None' uses the native sampling rate
    wave = wave[::3] #??
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
#    print(mfcc)
    return mfcc

#file_path_test = './data/test/0a7c2a8d_nohash_0.wav'
#mfcc_test = wav2mfcc(file_path_test)



DATA_PATH = "./data/"
        
def get_labels(path=DATA_PATH):
    '''Input: data path
    Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
    '''
    labels = [f for f in os.listdir(path) if not f.startswith('.')]
    label_indices = np.arange(0,len(labels))
    return labels, label_indices
#, to_categorical(label_indices)

#t1, t2 = get_labels()


def save_data_to_array(path=DATA_PATH, max_pad_len=11):
    '''reads all audio files from each directory and 
    save the vectors in a .npy file 
    which is named after the name of the label.
    '''
    labels, _, _ = get_labels(path)
    
    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []
        
        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in wavfiles:
            mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)


def get_train_test(split_ratio=0.6, random_state=42):
#    labels, indices, _ = get_labels(DATA_PATH)
    labels, indices = get_labels(DATA_PATH)    
    # get first arrays
    X = np.load(labels[0] + '.npy') # (sahpe: (17xx, 20, 11))
    y = np.zeros(X.shape[0]) # get 17xx WHY??
    
    # append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load((label + '.npy'))
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i+1)))
        
    assert X.shape[0] == len(y)
    
    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)


#X_train, X_test, y_train, y_test = get_train_test()
#print(X_train.shape) # (3112, 20, 11) # 3122 shows the number audio samples; for each audio file we get (20,11) matrix


# reshape it to give it a depth like an image with a single channel
#X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
#print(X_train.shape) # (3112, 20, 11, 1)


# one-hot encoding with Keras
#y_train_hot = to_categorical(y_train)
#y_test_hot = to_categorical(y_test)

# now we are ready to feed it into CNN