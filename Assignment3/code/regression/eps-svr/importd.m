function [X,T] = importd(type, filetype)
    t = fullfile(pwd, '..','..','..','data', type, [filetype '.txt']);
    Z = importdata(t);
    X = Z(:,1:size(Z,2)-1);
    T = Z(:,size(Z,2));
end
