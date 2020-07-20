function R = Rho(N)
% This function is created to compute the rho (normalized factor) 
% of Tchebichef polynomials.

R = zeros(N,1);
R(0+1) = 1.0;

for p = 1:N-1
    p1 = p + 1;
    R(p1,1) = R(p1-1,1) * (1.0 - double((p*p))/double((N*N)));  % (1-p^2/N^2)
end

for p = 0:N-1
    p1 = p + 1;
    R(p1,1) = R(p1,1) * double(N) / double((2*p + 1));      % N(1-P^2/N^2)/(2p+1)
end

end