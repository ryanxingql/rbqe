function B = TchebiFocus(Image, order)

Ord = order;                    % Order of Tchebichef polynomials
[M, N] = size(Image);
Image = Image./sqrt(M*N*mean2(Image.^2));
I = Image - mean2(Image);
P1 = TchebichefPoly(M, Ord); % P1 = Tchebichef polynomials of x-axis
P2 = TchebichefPoly(N, Ord); % P2 = Tchebichef polynomials of y-axis
B = P1*I*P2';                % Compute Tchebichef moments

%M2 = size(P1, 1);            % Determine size of P1
%N2 = size(P1, 1);            % Determine size of P2
%B = fliplr(triu(fliplr(B),0)); % Extract low order Tchebichef moments
%I2 = I.^2;
%B2 = B(1:M2,1:N2).^2;
%F = abs( sum(I2(:)) - sum(B2(:)))/abs(sum(B2(:)));

end