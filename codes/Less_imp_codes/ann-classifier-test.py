#based on PyBrain tutorial: http://pybrain.org/docs/tutorial/fnn.html 

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.structure           import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, TanhLayer, SoftmaxLayer, GaussianLayer
from pybrain.supervised.trainers import BackpropTrainer
from numpy.random                import multivariate_normal
from sklearn.cross_validation    import KFold
from scipy                       import diag, arange, meshgrid, where
from sklearn.preprocessing       import StandardScaler

import matplotlib.pyplot as plt

import numpy as np

def mlpClassifier(X,y,train_indices, test_indices, mom=0.1,weightd=0.01, epo=5):
    X_train, y_train, X_test, y_test = X[train_indices],y[train_indices], X[test_indices], y[test_indices]

    #Converting the data into a dataset which is easily understood by PyBrain. 
    tstdata = ClassificationDataSet(X.shape[1],target=1,nb_classes=8)
    trndata = ClassificationDataSet(X.shape[1],target=1,nb_classes=8)
 #   print "shape of X_train & y_train: " + str(X_train.shape) + str(y_train.shape)
    for i in range(y_train.shape[0]):
        trndata.addSample(X_train[i,:], y_train[i])
    for i in range(y_test.shape[0]):
        tstdata.addSample(X_test[i,:], y_test[i])
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()

    #printing the specs of data
#    print "Number of training patterns: ", len(trndata)
#    print "Input and output dimensions: ", trndata.indim, trndata.outdim
#    print "First sample (input, target, class):"
#    print trndata['input'][0], trndata['target'][0], trndata['class'][0]

    #The neural-network used
 #   print "Building Network..."
    #input layer, hidden layer of size 10(very small), output layer
    ANNc = FeedForwardNetwork()
    inLayer = LinearLayer(trndata.indim, name="ip")
    hLayer1 = TanhLayer(100, name = "h1")
    hLayer2 = SigmoidLayer(100, name = "h2")
    outLayer = SoftmaxLayer(trndata.outdim, name = "op")

    ANNc.addInputModule(inLayer)
    ANNc.addModule(hLayer1)
    ANNc.addModule(hLayer2)
    ANNc.addOutputModule(outLayer)

    ip_to_h1 = FullConnection(inLayer, hLayer1, name = "ip->h1")
    h1_to_h2 = FullConnection(hLayer1, hLayer2, name = "h1->h2")
    h2_to_op = FullConnection(hLayer2, outLayer, name = "h2->op")

    ANNc.addConnection(ip_to_h1)
    ANNc.addConnection(h1_to_h2)
    ANNc.addConnection(h2_to_op)
    ANNc.sortModules()

#    print "Done. Training the network."

    #The trainer used, in our case Back-propagation trainer
    trainer = BackpropTrainer( ANNc, dataset=trndata, momentum=mom, verbose=True, weightdecay=weightd)
    trainer.trainEpochs( epo )

    #The error
    trnresult = percentError( trainer.testOnClassData(dataset=trndata), trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )
 #   print "Done."
    return ANNc, trainer.totalepochs, (100 - trnresult), (100 - tstresult) 

if __name__ == '__main__':
    print "Start loading..."
    X_male_us = np.load("X_male26.dat")
    print "X_male shape: " + str(X_male_us.shape)
    y_male = np.load("y_male.dat")

    X_female_us = np.load("X_female26.dat")
    print "X_female shape: " + str(X_female_us.shape)
    y_female = np.load("y_female.dat")
    print "Done."

    scaler_m = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_male = scaler_m.fit_transform(X_male_us)

    scaler_f = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_female = scaler_f.fit_transform(X_female_us)

    mom = [0.001, 0.01, 0.1, 1, 10, 100]
    epo = [30, 40, 50, 60, 70, 80, 90]
    weightd = []
    accu_m_tst = []
    accu_m_trn = []
    accu_f_tst = []
    accu_f_trn = []


    for item in epo:
        kf=KFold(len(y_male),n_folds=15)
        print item
        totaltstaccuf = totaltrnaccuf = totaltstaccum = totaltrnaccum = 0
        turnno = 0
        for train_indices,test_indices in kf:
            turnno+=1
#            print "\n" + str(turnno) + "th turn:->"
            mlpc, epoch, trnaccu, tstaccu = mlpClassifier(X_male, y_male,train_indices, test_indices, epo=item)
            totaltrnaccum+=trnaccu
            totaltstaccum+=tstaccu
            mlpc, epoch, trnaccu, tstaccu = mlpClassifier(X_female, y_female,train_indices, test_indices, epo=item)
            totaltrnaccuf+=trnaccu
            totaltstaccuf+=tstaccu
#            print "epoch: %4d" % epoch,"  train accuracy: %5.2f%%" % trnaccu, "  test accuracy: %5.2f%%" % tstaccu
        accu_m_tst.append(totaltstaccum/15)
        accu_m_trn.append(totaltrnaccum/15)
        accu_f_tst.append(totaltstaccuf/15)
        accu_f_trn.append(totaltrnaccuf/15)

#    print "Result: train accuracy-" + str(totaltrnaccu) + "\t\ttest accuracy-" + str(totaltstaccu)
        #The network type and the parameters of Back-Propagation Trainer can be changed by modifying the mlpClassifier function

print max(accu_m_tst)
print max(accu_f_tst)
print max(accu_m_trn)
print max(accu_f_trn)

plt.plot(epo,accu_m_tst,label='Male Test')
plt.plot(epo,accu_f_tst,label='Female Test')
plt.plot(epo,accu_m_trn,label='Male Train')
plt.plot(epo,accu_f_trn,label='Female Train')
plt.legend(loc='best')
#plt.xscale('log')
plt.title('Artificial Neural Network')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.axis([1,100,0,100])
plt.grid(True)
plt.savefig("figures/mfcc_ann_epoch2.png", bbox_inches='tight')
plt.show()

