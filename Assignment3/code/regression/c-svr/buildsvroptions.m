function [svroptions] = buildsvroptions(cost,kernel,gamma,coef,degree)
    soptions ='-s 3';
    koptions = '-t';
    switch(kernel)
        case 'linear'  
            koptions = [koptions ' 0'];
        case 'polynomial'
            koptions = [koptions ' 1'];
            koptions = [koptions ' -g ' num2str(gamma)];
            koptions = [koptions ' -r ' num2str(coef)];
            koptions = [koptions ' -d ' num2str(degree)];
        case 'gaussian'
            koptions = [koptions ' 2'];
            koptions = [koptions ' -g ' num2str(gamma)];
    end
    coptions = ['-c ' num2str(cost)];
    boptions = '-b 1';
    svroptions = [soptions ' ' koptions ' ' coptions,' ',boptions];
end