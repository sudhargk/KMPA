function [M, V] = computeKMeans(X,numBasis)
    [idx,M,dsum]  = kmeans(X,numBasis);
    n = histc(idx, unique(idx)); % counts number of points in each cluster
    sigma = dsum(:) ./ n(:);
    V = mean(sigma);
end