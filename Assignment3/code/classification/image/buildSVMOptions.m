function [svmoptions,is_custom_kernel] = buildSVMOptions(cost,kernel,gamma,coef,degree)
    soptions ='-s 0';
    koptions = '-t';
    is_custom_kernel = false;
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
        case 'histogram'
             koptions = [koptions ' 4'];
             is_custom_kernel = true;
    end
    coptions = ['-c ' num2str(cost)];
    boptions = '-b 1';
    svmoptions = [soptions ' ' koptions ' ' coptions,' ',boptions];
end

function [svmoptions,is_custom_kernel] = buildSVMOptions(cost,kernel,gamma,coef,degree)
    soptions ='-s 0';
    koptions = '-t';
    is_custom_kernel = false;
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
        case 'histogram'
             koptions = [koptions ' 4'];
             is_custom_kernel = true;
    end
    coptions = ['-c ' num2str(cost)];
    boptions = '-b 1';
    svmoptions = [soptions ' ' koptions ' ' coptions,' ',boptions];
end