import numpy as np

print "Start loading..."
X_male_us = np.load("X_male26.dat")
print X_male_us.shape
X_female_us = np.load("X_female26.dat")
print X_male_us.shape
#y_male = np.load("y_male.dat")
#y_female = np.load("y_female.dat")
print "Loaded."

print "Start Dumping..."
X_super = np.empty((X_male_us.shape[0]+X_female_us.shape[0],X_male_us.shape[1]))
#y_super = np.empty((y_male.shape[0]+y_female.shape[0]))

X_super[::2,:] = X_male_us
X_super[1::2,:] = X_female_us
#y_super[::2] = y_male
#y_super[1::2] = y_female

print X_super.shape
#print y_super.shape


print "Start dumping..."
X_super.dump("X_super26.dat")
#y_super.dump("y_super.dat")
print "Dumped."
