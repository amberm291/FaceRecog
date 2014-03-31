import sys
import numpy as np
from scipy.misc import imread, imresize, imsave
import math
import re
import glob
import scipy
import cPickle as pickle
import matplotlib.pyplot as plt

#function to sort images according to their name
digits = re.compile(r'(\d+)')
def tokenize(filename):
    return tuple(int(token) if match else token
                 for token, match in
                 ((fragment, digits.search(fragment))
                  for fragment in digits.split(filename)))

#function to read pgm files
def read_pgm(filename, byteorder='>'):
	with open(filename, 'rb') as f:
		buffer = f.read()
	try:
		header, width, height, maxval = re.search(
			b"(^P5\s(?:\s*#.*[\r\n])*"
			b"(\d+)\s(?:\s*#.*[\r\n])*"
			b"(\d+)\s(?:\s*#.*[\r\n])*"
			b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
	except AttributeError:
		raise ValueError("Not a raw PGM file: '%s'" % filename)
	return np.frombuffer(buffer,
		        	dtype='u1' if int(maxval) < 256 else byteorder+'u2',
				count=int(width)*int(height),
				offset=len(header)
			   	 ).reshape((int(height), int(width)))

#Function to reconstruct faces
def reconFace(eFaces, AvgImage, Im, N):
        Im = Im.ravel()
        Im = Im - AvgImage
	temp = np.zeros((1))
	I = np.zeros((10304))
	I = AvgImage
	for i in xrange(N+1):
        	temp = temp + np.dot(eFaces[i,:]/np.linalg.norm(eFaces[i,:]),Im)
        	I = I + temp*eFaces[i,:]/np.linalg.norm(eFaces[i,:])
		temp = 0
	Im = Im + AvgImage
	e = 0
	for i in xrange(10304):	
		e = e + pow((I[i]-Im[i]),2)	
	e = e/10304
	# To get reconstructed images, uncomment this subcode
	"""		
	if N in (0,3,14,149,199):
		I = I.reshape(112,92) 
        	scipy.misc.imsave('ReconFace1{:>03}.jpg'.format(N+1),I)
	"""
	return e


#function for task2	
def classify(eFaces, AvgImage, Im, N):
	tFiles = glob.glob("Probe/*/*.pgm")  #Change Testing folder here for some other location
	tFiles.sort(key = tokenize)
	Label = np.zeros((200,2))
	c = 0
	Im1 = np.zeros((200,10304))
	for i in xrange(len(tFiles)):
		Label[i,0] = i/5 + 1
	for k in xrange(len(tFiles)):
		Ima = read_pgm(tFiles[k],byteorder = '>')
		Ima = Ima.ravel()
		Ima = Ima - AvgImage
		Im1[k,:] = Ima
	redData = np.zeros((len(tFiles),N))
	redTest = np.zeros((len(tFiles),N))
	redData = np.dot(Im,eFaces[0:N,:].T)
	redTest = np.dot(Im1,eFaces[0:N,:].T)
	d = np.zeros((len(tFiles)))
	for i in xrange(len(tFiles)):
		temp = redTest[i,:]
		for j in xrange(len(tFiles)):
			error = redData[j,:] - temp
		d[i] = np.linalg.norm(error)
		Label[i,1] = (np.argmin(d)-1)/5 + 1
		if Label[i,0]==Label[i,1]:
			c = c+1
	return c/2


def main():
	files = glob.glob("Gallery/*/*.pgm")  #Change Training folder here for some other location
	files.sort(key = tokenize)
	I = read_pgm(files[0],byteorder = '>')
	h = I.shape[0]
	w = I.shape[1]
	#Calculating average image
	AvgImage = np.zeros(h*w,np.float)
	Im = np.zeros((len(files),h*w))
	for i in xrange(len(files)):
		I = read_pgm(files[i],byteorder = '>')
		I.astype(float)
		I = I.ravel()
		Im[i,:] = I
	
	for i in xrange(w*h):
		AvgImage[i] = AvgImage[i] + np.sum(Im[:,i])/len(files)
	"""
	# To get average image, uncomment this subcode
	AvgImage = AvgImage.reshape(112,92)
	scipy.misc.imsave('average.jpg',AvgImage)
	AvgImage = AvgImage.ravel()	
	"""	

	A = np.zeros((h*w,len(files)),np.float)
	for i in xrange(len(files)):	
		temp = Im[i,:] - AvgImage
		A[:,i] = temp
	
	L = np.zeros((len(files),len(files)),np.float)
	L = np.dot(A.T, A)/len(files)

	eValues, eVectors = np.linalg.eig(L)
	idx = eValues.argsort()[::-1]   
	eValues = eValues[idx]
	eVectors = eVectors[:,idx]
	
	eFaces = np.zeros((len(files),w*h),np.float)
	eFaces = np.dot(A,eVectors)
	eFaces = eFaces.T
	#Calculating eigenfaces
	for i in xrange(5):
		temp = eFaces[i,:].reshape(112,92)  
		scipy.misc.imsave('eigenFaces{:>01}.jpg'.format(i),temp)	

	dRet = np.zeros(len(eValues))
	dRet[0] = eValues[0]
	for i in range(1,len(eValues)):
		dRet[i] = dRet[i-1] + eValues[i]
	dRet = dRet/sum(eValues)*100
	Y = np.arange(1,201,1)	
	plt.plot(Y, dRet)
	plt.savefig('part2.png')
	plt.clf()
	for i in xrange(len(dRet)):
		if dRet[i]>85:
			print "The no of dimensions for capturing more than 85% of original data is ",i
			break
	
	for i in xrange(len(dRet)):
		if dRet[i]>95:
			print "The no of dimensions for capturing more than 95% of original data is ",i
			break

	#Calculating Mean Squared Error for each reconstruction
	MSError1 = np.zeros((len(files)))
	MSError2 = np.zeros((len(files)))
	Im1 = read_pgm('face_input_1.pgm',byteorder = '<')
	for i in xrange(len(files)):
		MSError1[i] = reconFace(eFaces, AvgImage, Im1, i)
		if i+1 in (1,4,15,150,200):
			print "MSERROR for first image",MSError1[i]
	
	Im1 = read_pgm('face_input_2.pgm',byteorder = '>')
	for i in xrange(len(files)):
		MSError2[i] = reconFace(eFaces, AvgImage, Im1, i)
		if i+1 in (1,4,15,150,200):
                        print "MSERROR for second image",MSError2[i]
	plt.plot(Y,MSError1)
	plt.savefig('image1MSE.png')
	plt.clf()
	plt.plot(Y,MSError2)
	plt.savefig('image2MSE.png')
	plt.clf()
	
	#Classifying images
	for i in xrange(len(files)):
		a = classify(eFaces, AvgImage, Im, i)
		print a
	
if __name__ == "__main__":
	main()
