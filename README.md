# Emotion-Recognition-in-Hindi-Speech
Classifying utterances in Hindi speech in one of the 
8 emotional states (anger, fear, disgust, neutral, sad, happy, surprise, sarcastic) in spoken speech in Hindi

To understand what we have done, refer in sequence:
1) Poster
2) Report

Please note that code we uploaded here is not very significant. 'Codes' folder contains 
1) simple code for feature extraction into array,
2) using Sklearn, Pybrain Classifiers
3) using MatplotLib to make Confusion Matrices.

Quoting the abstract from our report:

"In this project, simulated Hindi emotional speech database has been borrowed
from a subset of IITKGP-SEHSC dataset(2 out of 10 speakers). Emotional
classification is attempted on the corpus using spectral features. The
spectral features used are Mel Frequency Cepstral Coefficients(MFCCs) and
Subband Spectral Coefficents(SSCs) The feature vector in use has 273 features,
obtained from 7 individual features of 13 banks of MFCCs and 26 SSCs computed
over the dataset. This dataset is trained on multiple classifiers, wherein
with each classifier, related learning and error penalty rate parameters have
been varied to find the best set of classifiers. The lists of accuracies, precisions,
and f1-scores are compared. Our methods show that Support Vector
Machines with Radial Basis Function kernel provides the best accuracy rates,
with accuracy for male dataset being 89.08% and for female dataset being
83.16%. The results are on par with the results obtained by training on full
IITKGP-SEHSC dataset."

Our main work was to use extracted MFCC & SSC features in such a way that can help the
Classifiers to classify the emotion expressed by the speech utterances.

But again, I request you to read the poster (and the report if you are patient enough) for exact details.
