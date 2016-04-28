#based on PyBrain tutorial: http://pybrain.org/docs/tutorial/fnn.html 

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from numpy.random                import multivariate_normal
from sklearn.cross_validation    import KFold
from scipy                       import diag, arange, meshgrid, where
from sklearn.preprocessing       import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,confusion_matrix
from itertools import izip


import numpy as np

def mlpClassifier(X,y,train_indices, test_indices, mom=0.1,weightd=0.01, epo=5):
    X_train, y_train, X_test, y_test = X[train_indices],y[train_indices], X[test_indices], y[test_indices]

    #Converting the data into a dataset which is easily understood by PyBrain. 
    tstdata = ClassificationDataSet(X.shape[1],target=1,nb_classes=8)
    trndata = ClassificationDataSet(X.shape[1],target=1,nb_classes=8)
    for i in range(y_train.shape[0]):
        trndata.addSample(X_train[i,:], y_train[i])
    for i in range(y_test.shape[0]):
        tstdata.addSample(X_test[i,:], y_test[i])
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()
    mlpc = buildNetwork( trndata.indim, 100, trndata.outdim, outclass=SoftmaxLayer )
    trainer = BackpropTrainer( mlpc, dataset=trndata, momentum=mom, verbose=True, weightdecay=weightd)
    trainer.trainEpochs( epo )
    y_pred = trainer.testOnClassData(dataset=tstdata )
    print "Done. Accu: " + "%.2f"%accuracy_score(y_test, y_pred)
    return y_test, y_pred

if __name__ == '__main__':
    print "Start loading..."
    X_male_us = np.load("X_male.dat")
    print "X_male shape: " + str(X_male_us.shape)
    y_male = np.load("y_male.dat")

    X_female_us = np.load("X_female.dat")
    print "X_female shape: " + str(X_female_us.shape)
    y_female = np.load("y_female.dat")
    print "Done."

    scaler_m = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_male = scaler_m.fit_transform(X_male_us)

    scaler_f = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_female = scaler_f.fit_transform(X_female_us)

    kf=KFold(len(y_male),n_folds=15)
    list_test_male=[]
    list_test_female=[]
    list_pred_male=[]
    list_pred_female=[]
    total=0
    y_test_male=[]
    y_pred_male=[]
    y_test_female=[]
    y_pred_female=[]
    recall_male=[0,0,0,0,0,0,0,0]
    precision_male=[0,0,0,0,0,0,0,0]
    f1_male=[0,0,0,0,0,0,0,0]
    recall_female=[0,0,0,0,0,0,0,0]
    precision_female=[0,0,0,0,0,0,0,0]
    f1_female=[0,0,0,0,0,0,0,0] 
    accuracy_male_avg=0
    recall_male_avg=0
    precision_male_avg=0
    accuracy_female_avg=0
    recall_female_avg=0
    precision_female_avg=0  
    f1_male_avg=0
    f1_female_avg=0
    label_list=['anger','disgust','fear','happy','neutral','sadness','sarcastic','surprise']
    
    # for i in range(1,205,5):
    efficiency_male=efficiency_female=0
    total=0
    for train_inds,test_inds in kf:
        # print("TRAIN:",train_inds[0],"-",train_inds[-1],"TEST:",test_inds[0],"-",test_inds[-1])
        y_test_male,y_pred_male=mlpClassifier(X_male, y_male, train_inds, test_inds, epo=50)
        y_test_female,y_pred_female=mlpClassifier(X_female, y_female,train_inds, test_inds,  epo=50)

        accuracy_male_avg+=(accuracy_score(y_test_male,y_pred_male)*100.0)
        accuracy_female_avg+=(accuracy_score(y_test_female,y_pred_female)*100.0)
        recall_male_avg+=(recall_score(y_test_male,y_pred_male,average='macro')*100.0)
        recall_female_avg+=(recall_score(y_test_female,y_pred_female,average='macro')*100.0)
        precision_male_avg+=(precision_score(y_test_male,y_pred_male,average='macro')*100.0)
        precision_female_avg+=(precision_score(y_test_female,y_pred_female,average='macro')*100.0)
        f1_male_avg+=(f1_score(y_test_male,y_pred_male,average='macro')*100.0)
        f1_female_avg+=(f1_score(y_test_female,y_pred_female,average='macro')*100.0)
        temp=recall_score(y_test_male,y_pred_male,average=None)
        # print temp
        temp2=[sum(x) for x in izip(recall_male,temp)]
        recall_male=temp2[:]
        temp=precision_score(y_test_male,y_pred_male,average=None)
        temp2=[sum(x) for x in izip(precision_male,temp)]
        precision_male=temp2[:]
        temp=f1_score(y_test_male,y_pred_male,average=None)
        temp2=[sum(x) for x in izip(f1_male,temp)]
        f1_male=temp2[:]
        temp=recall_score(y_test_female,y_pred_female,average=None)
        temp2=[sum(x) for x in izip(recall_female,temp)]
        recall_female=temp2[:]
        temp=precision_score(y_test_female,y_pred_female,average=None)
        temp2=[sum(x) for x in izip(precision_female,temp)]
        precision_female=temp2[:]
        temp=f1_score(y_test_female,y_pred_female,average=None)
        temp2=[sum(x) for x in izip(f1_female,temp)]
        f1_female=temp2[:]
        list_test_male.extend(y_test_male)
        list_pred_male.extend(y_pred_male)
        list_test_female.extend(y_test_female)
        list_pred_female.extend(y_pred_female)
        total+=1
    # print i
    print("Male")
    print("Accuracy avg : "+str((accuracy_male_avg/total)))
    print("Recall avg : "+str((recall_male_avg/total)))
    print("Precision avg : "+str((precision_male_avg/total)))
    print("F1 avg : "+str((f1_male_avg/total)))
    print("Female")
    print("Accuracy avg : "+str((accuracy_female_avg/total)))
    print("Recall avg : "+str((recall_female_avg/total)))
    print("Precision avg : "+str((precision_female_avg/total)))
    print("F1 avg : "+str((f1_female_avg/total)))
    recall_male[:]=[x/total for x in recall_male]
    precision_male[:]=[x/total for x in precision_male]
    f1_male[:]=[x/total for x in f1_male]
    recall_female[:]=[x/total for x in recall_female]
    precision_female[:]=[x/total for x in precision_female]
    f1_female[:]=[x/total for x in f1_female]
    cm_male=confusion_matrix(list_test_male,list_pred_male)
    cm_female=confusion_matrix(list_test_female,list_pred_female)
    cm_norm_male = cm_male.astype('float') / cm_male.sum(axis=1)[:, np.newaxis]
    cm_norm_female = cm_female.astype('float') / cm_female.sum(axis=1)[:, np.newaxis]
    # print label_list
    print("Male confusion matrix")
    print(cm_male)
    print('Female confusion matrix')
    print(cm_female)
    print("for male")   
    print "\trecall\tprecision\tf1"
    for x in label_list:
        print(x+"\t%.2f"%recall_male[label_list.index(x)]+"\t%.2f"%precision_male[label_list.index(x)]+"\t%.2f"%f1_male[label_list.index(x)])
    print("for female") 
    for x in label_list:
        print(x+"\t%.2f"%recall_female[label_list.index(x)]+"\t%.2f"%precision_female[label_list.index(x)]+"\t%.2f"%f1_female[label_list.index(x)])
    plt.figure(1)
    plt.imshow(cm_norm_male, interpolation='nearest', cmap=plt.cm.Greys)
    plt.title("Confusion matrix for male data(MLPC)")
    plt.colorbar()
    tick_marks = np.arange(len(label_list))
    plt.xticks(tick_marks, label_list, rotation=45)
    plt.yticks(tick_marks, label_list)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("./figures/confusion_MLPC_male.png", bbox_inches='tight')
    plt.show()

    plt.figure(2)
    plt.imshow(cm_norm_female, interpolation='nearest', cmap=plt.cm.Greys)
    plt.title("Confusion matrix for female data(Random Forest)")
    plt.colorbar()
    tick_marks = np.arange(len(label_list))
    plt.xticks(tick_marks, label_list, rotation=45)
    plt.yticks(tick_marks, label_list)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("./figures/confusion_MLPC_female.png", bbox_inches='tight')
    plt.show()
