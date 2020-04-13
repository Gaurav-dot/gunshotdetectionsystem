
#from matplotlib import pyplot 
import scipy.io.wavfile
import matplotlib.pyplot as plt


'''for i in range(1 , 1089):
	#plt.ylim(-60000, 60000)
	rate, data =scipy.io.wavfile.read('Audio_Gunshots/'+str(i)+'.wav')
	plt.plot(data)
	plt.savefig('Training_datasets/gunshot/gunshot'+str(i)+'.jpg')
	plt.clf()'''
'''for i in range(1070 , 1089):
	rate, data =scipy.io.wavfile.read('Audio_not_Gunshots/'+str(i)+'.wav')
	plt.plot(data)
	plt.savefig('Training_datasets/not_gunshot/notgunshot'+str(i)+'.jpg')
	plt.clf()
'''


rate, data =scipy.io.wavfile.read('output.wav')
plt.plot(data)
plt.savefig('result.jpg')
