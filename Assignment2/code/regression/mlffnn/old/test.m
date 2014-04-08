function test()
pattern =  [0.1 0.1 0.1; 0.1 0.95 0.95; 0.95 0.1 0.95;0.95 0.95 0.1];
eta = 1.0;
alpha = 0.4;
tol = 0.001;
Q =4;
n=2; q=2; p=1;
Wih = 2*ones(n+1,q)-1.5;
Whj = 2*ones(q+1,p)-1.5;
DeltaWih = zeros(n+1,q);
DeltaWhj = zeros(q+1,p);
DeltaWihOld = zeros(n+1,q);
DeltaWhjOld = zeros(q+1,p);

Si = [ones(Q,1) pattern(:,1:2)];
D= pattern(:,3);
Sh = [1 zeros(1,q)];
Sy = zeros(1,p);
deltaO = zeros(1,p);
deltaH =zeros(1,q+1);
sumerror = 2*tol;
epoch =1;
while(sumerror>tol)
    sumerror = 0;
    for k=1:Q
        Zh = Si(k,:)*Wih;
        Sh = [1 1./(1+exp(-Zh))];
        Yj = Sh*Whj;
        Sy = 1./(1+exp(-Yj));
        Ek =D(k,:)-Sy;
        deltaO  = Ek.*Sy.*(1-Sy);
        for h =1:q+1
           DeltaWhj(h,:) = deltaO * Sh(h); 
        end
        for h =2:q+1
           deltaH(h) = (deltaO * Whj(h,:)')*Sh(h)*(1-Sh(h)); 
        end
        for i =1:n+1
           DeltaWih(i,:) = deltaH(2:q+1) * Si(k,i); 
        end
        Wih = Wih + eta*DeltaWih + alpha*DeltaWihOld;
        Whj = Whj + eta*DeltaWhj + alpha*DeltaWhjOld;
        DeltaWihOld = DeltaWih;
        DeltaWhjOld = DeltaWhj;
        sumerror = sumerror+ sum(Ek.^2);
    end
    display([ 'epoch ' num2str(epoch) ' : '  num2str(sumerror)]);
    epoch=epoch+1;
end
end
