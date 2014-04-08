% Short script to load data when testing on command line
clear all;
datatype = 'nonlinearlySeparable';
[trainX, trainT] = importd(datatype, 'train');
[valX, valT] = importd(datatype, 'val');
[testX, testT] = importd(datatype, 'test');
inputs = [trainX valX testX];
targets = [trainT valT testT];
