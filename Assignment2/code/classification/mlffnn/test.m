%   in the workspace 'imp' is the data imput (7 variables for each input) and
  % 'targ' is the data target
  clear all;
  loadd;
  imp = inputs; targ = targets;
  sz = size (imp);
  % numbers of the coloumns for separation between train set, validation set, test set.
  d1=round(sz(2)/2); % the half of the dataset
  d2=round((sz(2)-d1)/2)+d1; % the half of the remaining part i.e. the quart
  d3=sz(2);  % the last quart
  % network with 1 input layer (size of imput is 7) , one hidden layer of 5
  % neurons, and one unique neurone in the output layer
  net = patternnet(5, 'traingdm');
  net = configure(net, imp, targ);
  % actual separation for train test and validation set
  imp1=imp(:,1:d1); % imput for training
  targ1=targ(:,1:d1); % target for training
  VV.P=imp(:,d1+1:d2); % validation set
  VV.T=targ(:,d1+1:d2);
  VT.P=imp(:,d2+1:d3); % test set
  VT.T=targ(:,d2+1:d3);
  net.inputweights{1,1}.initfcn = 'rands';
  net.layers{1}.transferFcn = 'tansig';
  net.layers{2}.transferFcn = 'purelin';
  net = init(net);
  train(net,imp1,targ1,[],[],VV,VT);
  % simulation on the full dataset
  y1 = sim(net,imp);
  % bias of layers addapted for direct calculation with the size of the dataset 
  
  B1 = net.b{1}*ones(1,size(imp,2)); % all the coloumns are identical, and equal to net.b{1}
  B2 = net.b{2}*ones(1,size(imp,2));
  imp2 = mapminmax('apply',imp,net.inputs{1}.processSettings{1});
  OutLayer1 = tansig(net.IW{1}*imp2+B1);
  OutLayer2 = purelin(net.LW{2}*OutLayer1+B2);
  y2 = mapminmax('reverse',OutLayer2,net.outputs{2}.processSettings{1});
  % now you can compare y1 and y2
  plot(1:d3,y1,'o',1:d3,y2,'x');
  % NOTE THE *10^4 in the Y axis