function [M,tichonovDist,width] = computeClusterMeans(X,numBasis)
    [~,M,~]  = kmeans(X,numBasis,'start','sample');
    tichonovDist = pdist2(M,M)^2;
    width = max(max(tichonovDist))/(2*numBasis);
    tichonovDist = tichonovDist/width;
    tichonovDist = exp(-tichonovDist);
end