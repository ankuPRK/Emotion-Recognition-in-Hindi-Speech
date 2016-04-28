#This code is being used to extract features from audio files and save those features as an nxm array
#Speech features are extracted using library from: https://github.com/jameslyons/python_speech_features

from features.base import mfcc, fbank, logfbank, ssc
import scipy.io.wavfile as wav
import numpy as np

cepCount=13 #no of MFCC coefficients
nfeatures = 7 #features per coefficient
elcount = 6

def audioread(datafs,gender_flag):
	(data, fs) = wav.read(datafs)
	ceps = mfcc(fs,numcep=cepCount)
	feat2 = ssc(fs,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97)
	ls = []
	for i in range(ceps.shape[1]):
		temp = ceps[:,i]
		dtemp = np.gradient(temp)
		lfeatures  = [np.mean(temp), np.var(temp), np.amax(temp), np.amin(temp), 
		np.var(dtemp), np.mean(temp[0:temp.shape[0]/2]), np.mean(temp[temp.shape[0]/2:temp.shape[0]])]
		temp2 = np.array(lfeatures)
		ls.append(temp2)
	ls2 = []
	for i in range(feat2.shape[1]):
		temp = feat2[:,i]
		dtemp = np.gradient(temp)
		lfeatures = [np.mean(temp), np.var(temp), np.amax(temp), np.amin(temp), 
		np.var(dtemp), np.mean(temp[0:temp.shape[0]/2]), np.mean(temp[temp.shape[0]/2:temp.shape[0]])]
		temp2 = np.array(lfeatures)
		ls2.append(temp2)
	source = np.array(ls).flatten()
	source = np.append(source, np.array(ls2).flatten())
	return source

def load_data():
	emotions = ['anger','disgust','fear','happy','neutral','sadness','sarcastic','surprise']
	male_path = '/home/ankuprk/Emotion-Recognition/IITKGP-SEHSC/3/session'
	female_path = '/home/ankuprk/Emotion-Recognition/IITKGP-SEHSC/4/session'

	max_len_male=max_len_female=0
	X_male=np.empty(shape=(1200,(cepCount + 26)*nfeatures ))
	X_female=np.empty(shape=(1200,(cepCount + 26)*nfeatures ))
	y_male=np.empty(1200)
	y_female=np.empty(1200)
	mcount=fcount=0
	print "Loop Started...."
#	filename style: 3.2.anger-01.wav
	for j in xrange(1,16):
		if(j<=9):
			jstring = '0' + str(j)
		else:
			jstring = str(j)
		for i in xrange(1,11):
			for emo in emotions:
				x = male_path+str(i) + '/' + emo + '/' + '3.' + str(i) + '.' +emo +'-' + str(jstring) + '.wav'
				X_male[mcount]=audioread(x,gender_flag='male')
				y_male[mcount]=emotions.index(emo)
				mcount+=1
			for emo in emotions:
				x = female_path+str(i) + '/' + emo + '/' + '4.' + str(i) + '.' +emo +'-' + str(jstring) + '.wav'
				X_female[fcount]=audioread(x,gender_flag='female')
				y_female[fcount]=emotions.index(emo)
				fcount+=1 					
	return X_male,X_female ,y_male,y_female

if __name__ == '__main__':
	print "Start loading..."
	X_male,X_female,y_male,y_female=load_data()
	print "Start dumping..."
	X_male.dump("X_male.dat")
	X_female.dump("X_female.dat")
	y_male.dump("y_male.dat")
	y_female.dump("y_female.dat")
	print "Done."
