function [svroptions] = buildsvroptions(cost,coef,gamma,epsilon,nu)
%    soptions ='-s 3';
    koptions = [' -t' ' 2' ' -g ' num2str(gamma)];
%     switch(kernel)
%         case 'linear'  
%             koptions = [koptions ' 0'];
%         case 'polynomial'
%             koptions = [koptions ' 1'];
%             koptions = [koptions ' -g ' num2str(gamma)];
%             koptions = [koptions ' -r ' num2str(coef)];
%             koptions = [koptions ' -d ' num2str(degree)];
%         case 'gaussian'
%             koptions = [koptions ' 2'];
%             koptions = [koptions ' -g ' num2str(gamma)];
%     end
%     potions = [ '-p', num2str(epsilon)]
%     coptions = ['-c ' num2str(cost)];
%     boptions = '-b 1';
%     svroptions = [soptions ' ' koptions ' ' coptions,' ',boptions];
        svroptions = [ ' -s 4' ' ' koptions ' -c ' num2str(cost) ' -p ' num2str(epsilon) ' -r ' num2str(coef) ' -n ' num2str(nu)];
end
