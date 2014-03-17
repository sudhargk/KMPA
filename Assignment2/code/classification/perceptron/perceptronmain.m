


function []=perceptronmain()
    eta=0.5;
    dataset = 'linearlySeparableData';
    path = fullfile(pwd,'..','..','..','data', dataset, 'data');
    load(path);
%     D1=[ones(size(trainset{1,1},1),1) trainset{1,1}];
%     D2=[ones(size(trainset{2,1},1),1) trainset{2,1}];
%     D3=[ones(size(trainset{3,1},1),1) trainset{3,1}];
%     w12=[0 0 0];
%     w13=[0 0 0];
%     w23=[0 0 0];
    %D1 = [ 1 2; 2 3; 4 5];
    %D2 = [0 -1; 1 -1; 5 -5];
    
     charMat = [1 1 0; -1 0 1; 0 -1 -1];
%     for index = 1 : size(charMat,2)
%         w(index,:) = perceptron(trainset(abs(charMat(:,index))==1), eta);
%     end
    w = train(trainset,charMat,eta);
    c = testData(testset,w,charMat);
    display(c);
    visualize(testset,w,charMat);
%     display(w12);
%     display(w13);
%     display(w23);
%     graph( w12, D1, D2);
%     graph( w13, D1, D3);
%     graph( w23, D2, D3);
end

function [w] = train(trainset,charMat,eta)
    for index = 1 : size(charMat,2)
        w(index,:) = perceptron(trainset(abs(charMat(:,index))==1), eta);
    end
%     graph(w(1,:));
%     graph(w(2,:));
%     graph(w(3,:));
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
    min1=inf;
    min2=inf;
    minp1=[0 0 0];
    minp2=[0 0 0];
    while update==1 
        for i=1:s
%             y = Data(idx(i),:)*w';
%             if y >= 0 && d(idx(i)) == 0
%                 w = w - eta*Data(idx(i),:);
%                 up(idx(i))=1;
%             elseif y <= 0 && d(idx(i)) == 1
%                 w = w + eta*Data(idx(i),:);
%                 up(idx(i))=1;
%             else
%                 up(idx(i))=0;
%             end
            y = Data(idx(i),:)*w';
            if y*d(idx(i))<=0
                w = w + eta*Data(idx(i),:)*d(idx(i));
                up(idx(i))=1;
            else
                up(idx(i))=0;
            end
%             if d(idx(i)) == 1 && min1 > abs(y)
%                 minp1 = Data(idx(i),:);
%                 min1 = abs(y);
%             end
%             if d(idx(i)) == -1 && min2 > abs(y)
%                 minp2 = Data(idx(i),:);
%                 min2 = abs(y);
%             end
        end
        number_of_updates = up * up';
        if number_of_updates > 0
            update=1;
        else update=0;
        end
    end
%     minp= (minp1 + minp2)/2;
%     w(1) = -w(2)*minp(2)-w(3)*minp(3);
%     graph(w,D1,D2);
%     pause;
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

% function [decidedClass]= classify(dataset,weight,charMat)
%     dataset = [ones(size(dataset,1),1) dataset];
%     estimate = charMat*weight*dataset';
%     [~,decidedClass] = max((estimate));
% end

function [decidedClass]= classify(dataset,weight,charMat)
    dataset = [ones(size(dataset,1),1) dataset];
    estimate1 = weight*dataset';
    estimate = charMat*weight*dataset';
    for i = 1 : size(dataset,1)
        [~,temp]=max(estimate(:,i));
        if(temp == 1)
            if(estimate1(1,i)<0 || estimate1(2,i)<0)
                [~,temp]=max(estimate(2:3,i));
                if(temp == 2)
                    if(estimate1(1,i)>0 || estimate1(3,i)<0)
                        temp=3;
                        if(estimate1(2,i)>0 || estimate1(3,i)>0)
                            [~,temp]=max(estimate(:,i));
                        end
                    end
                else
                    if(estimate1(2,i)>0 || estimate1(3,i)>0)
                        temp=2;
                        if(estimate1(1,i)>0 || estimate1(3,i)<0)
                            [~,temp]=max(estimate(:,i));
                        end
                    end
                end
            end
        elseif(temp == 2)
            if(estimate1(1,i)>0 || estimate1(3,i)<0)
                [~,temp]=max(estimate([1 3],i));
                if(temp == 1)
                    if(estimate1(1,i)<0 || estimate1(2,i)<0)
                        temp=3;
                        if(estimate1(2,i)>0 || estimate1(3,i)>0)
                            [~,temp]=max(estimate(:,i));
                        end
                    end
                else
                    if(estimate1(2,i)>0 || estimate1(3,i)>0)
                        temp=1;
                        if(estimate1(1,i)<0 || estimate1(2,i)<0)
                            [~,temp]=max(estimate(:,i));
                        end
                    end
                end
            end
        else
            if(estimate1(2,i)>0 || estimate1(3,i)>0)
                [~,temp]=max(estimate([1 2],i));
                if(temp == 1)
                    if(estimate1(1,i)<0 || estimate1(2,i)<0)
                        temp=2;
                        if(estimate1(1,i)>0 || estimate1(3,i)<0)
                            [~,temp]=max(estimate(:,i));
                        end
                    end
                else
                    if(estimate1(1,i)>0 || estimate1(3,i)<0)
                        temp=1;
                        if(estimate1(1,i)<0 || estimate1(2,i)<0)
                            [~,temp]=max(estimate(:,i));
                        end
                    end
                end
            end
        end
        decidedClass(i) = temp;
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
    figure(1), hold on;
    image(Xl, Yl, reshape(gridDC, n, n))
    axis xy, box on, colormap(clrLite);
    
    %Superimposing data points
    gscatter(D(:,1), D(:,2), group, clrDark, '.', 15)
    
    %Superimposing wrongly classified points
    bad = (DC ~= AC);
    plot(D(bad,1), D(bad,2), 'yx', 'MarkerSize', 10)
    axis([mn(1) mx(1) mn(2) mx(2)]);
    axis equal
    xlabel('Dimension 1'), ylabel('Dimension 2'); 
    hold off   
end
