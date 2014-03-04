<<<<<<< HEAD
function [M, V] = computeClusterMeans(X,numBasis)
    [idx,M,dsum]  = kmeans(X,numBasis-1);
%     n = histc(idx, unique(idx)); % counts number of points in each cluster
%     sigma = dsum(:) ./ n(:);
%     V = mean(sigma);
        V = trace(cov(X)) / size(X,2); 
end
=======
function [M, V] = computeClusterMeans(X,numBasis)
    [idx,M,dsum]  = kmeans(X,numBasis-1);
%     n = histc(idx, unique(idx)); % counts number of points in each cluster
%     sigma = dsum(:) ./ n(:);
%     V = mean(sigma);
        V = trace(cov(X)); 
end
>>>>>>> 77cca7424c1d03cd09657342b67db729b05e9345
