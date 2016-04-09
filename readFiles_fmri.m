function[] = readFiles_fmri()
trX = load('fmri/data/Xtrain_transpose.txt');
trY = load('fmri/data/Ytrain_transpose');
tstX = load('fmri/data/Xtest_transpose');
tstY = load('fmri/data/Ytest_transpose');

fmri_shooting(trX, trY, tstX, tstY);