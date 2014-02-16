function [M, V] = computeClusterMeans(X,numBasis)
    [idx,M,dsum]  = kmeans(X,numBasis-1);
    n = histc(idx, unique(idx)); % counts number of points in each cluster
    sigma = dsum(:) ./ n(:);
    V = mean(sigma);
end