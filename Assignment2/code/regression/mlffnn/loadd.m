% Short script to load data when testing on command line
type = 'nonlinearlySeparable';
clear all;
[trainX, trainT] = importd(type, 'train');
[valX, valT] = importd(type, 'val');
[testX, testT] = importd(type, 'test');
inputs = [trainX valX testX];
targets = [trainT valT testT];
