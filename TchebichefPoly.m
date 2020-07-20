function F = TchebichefPoly(N, Ord)

Pnx = double(zeros(Ord+1,N));           % Tchebichef polynomials
for x = 0:N-1
    x1 = x + 1;
    Pnx(0+1,x1) = 1.0;                  % P0 = 1.0
    Pnx(1+1,x1) = (1 - N + 2.0*x) / N;  % P1 = (1-N+2x); divided by N to map Tchebichef polynomials within [-1,1]
    for p = 2:Ord
        p1 = p + 1;
        Pnx(p1,x1) = ( double(2*p-1)*Pnx(1+1,x1)*Pnx(p1-1,x1) - ...
            double((p-1))*(1.0-double((p-1)*(p-1))/double(N*N))*Pnx(p1-2,x1) ) / double(p);
    end
end

R = Rho(N);
for x = 0:N-1
    x1 = x + 1;
    for p = 0:Ord
        p1 = p + 1;
        Pnx(p1,x1) = Pnx(p1,x1) / sqrt(R(p1,1));
    end
end

F = Pnx;

end