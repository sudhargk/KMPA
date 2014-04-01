
function []=perceptronmain()
    eta=0.5;
    dataset = 'linearlySeparable';
    path = fullfile(pwd,'..','..','..','data', dataset ,'data');
    load(path);    
    charMat = [1 1 0; -1 0 1; 0 -1 -1];
    w = train(trainset,charMat,eta);
    c = testData(testset,w,charMat);
    display(c);
    visualize(testset,w,charMat);
end

function [w] = train(trainset,charMat,eta)
    for index = 1 : size(charMat,2)
        w(index,:) = perceptron(trainset(abs(charMat(:,index))==1), eta);
    end
end

function [w]=perceptron(D, eta)
    update=1;
    D1 = [ones(size(D{1},1),1) D{1}];
    D2 = [ones(size(D{2},1),1) D{2}];
    w = zeros(1,size(D1,2));
    [rows1 ~]= size(D1);
    [rows2 ~]= size(D2);
    Data=[D1;D2];
    d1=ones(rows1,1);
    d2=-ones(rows2,1);
    d=[d1;d2];                              %desired output
    s=rows1+rows2;
    up=ones(1,s);
    idx = randperm(s);
    while update==1 
        for i=1:s
            y = Data(idx(i),:)*w';
            if y*d(idx(i))<=0
                w = w + eta*Data(idx(i),:)*d(idx(i));
                up(idx(i))=1;
            else
                up(idx(i))=0;
            end
        end
        number_of_updates = up * up';
        if number_of_updates > 0
            update=1;
        else update=0;
        end
    end
end


function []=graph( w, D1, D2)
    hold on;
%     p1=plot(D1(:,2),D1(:,3),'.');
%     p2=plot(D2(:,2),D2(:,3),'o');
    xlabel('X');           
    ylabel('Y');
%     set(p2, 'Color', 'r');
    %ezplot('y = (-w(2)/w(3))*x - w(1)/w(3)', [-20 20 -20 20])
    X=linspace(-5,20);
    Y=(-w(2)/w(3))*X - w(1)/w(3);
    plot(X,Y);
end


function [C]= testData(dataset,weight,charMat)
    numClass = size(dataset,1);
    numSample = cellfun(@length,dataset);
    totalSample = sum(numSample);
    actualClass = getActualClass(numSample);
    dataset = cell2mat(dataset);
    decidedClass = classify(dataset,weight,charMat);
    targets=full(ind2vec(actualClass));
    outputs = full(ind2vec(decidedClass));
    figure(),plotconfusion(targets,outputs),set(gcf, 'WindowStyle', 'docked');
    figure(),plotroc(targets,outputs),set(gcf, 'WindowStyle', 'docked');
    C = zeros(numClass,numClass);
    for i = 1:totalSample
        C(actualClass(i),decidedClass(i)) = C(actualClass(i),decidedClass(i)) + 1;
    end
end

function [C] = getActualClass(numSample)
    numClass = length(numSample);
    totalSample = sum(numSample);
    S = cumsum(numSample);
    C = ones(1,totalSample);
    A = 1:totalSample;
    for i = 1 : numClass-1
        C = C + (A > S(i));
    end
end

function [decidedClass]= classify(dataset,weight,charMat)
    dataset = [ones(size(dataset,1),1) dataset];
    votes = charMat*sign(weight*dataset');
    [val,decidedClass] =max(votes);
    indices = 1:length(val);
    indices=indices(val==0);
    if(~isempty(indices))
        estimate = charMat*weight*dataset(indices,:)';
        [~,newClasses] = max((estimate));
       decidedClass(indices)=newClasses;    
    end
end


function  visualize(testset,weight,charMat)
    numClass = size(testset,1);
    numSample = cellfun(@length,testset);
    AC = getActualClass(numSample);
    labels = strcat({'Class '}, num2str((1:numClass)'));
    group = ordinal(AC, labels);
    D = cell2mat(testset);
    mn = min(D); mx = max(D); n = 1000;
    
    clrLite = [1 0.6 0.6 ; 0.6 1 0.6 ; 0.6 0.6 1; 1 0.6 1];
    clrDark = [0.7 0 0 ; 0 0.7 0 ; 0 0 0.7; 0.7 0 0.7];

    [X, Y] = meshgrid( linspace(mn(1),mx(1),n), linspace(mn(2),mx(2),n) );
    Xl = X(:); Yl = Y(:);
    gridDC = classify([Xl Yl], weight,charMat);
    DC = classify(D, weight,charMat);
    
    %plotting all  points
     figure(),set(gcf, 'WindowStyle', 'docked'),hold on;
    image(Xl, Yl, reshape(gridDC, n, n))
    axis xy, box on, colormap(clrLite);
    
    %Superimposing data points
    gscatter(D(:,1), D(:,2), group, clrDark, '.', 15)
    
    %Superimposing wrongly classified points
    bad = (DC ~= AC);
    plot(D(bad,1), D(bad,2), 'yx', 'MarkerSize', 10)
    axis([mn(1) mx(1) mn(2) mx(2)]);
    axis tight
    xlabel('Dimension 1'), ylabel('Dimension 2'); 
    hold off   
end
