function a1114b()
%==========================================================================
% S1 - Fourier series solution (lambda-corrected) for the 2D Laplace
%      equation, parallel-bars (barra-barra) configuration
%==========================================================================
% Supplementary Material S1 for:
%   "From White-Box to Epistemic Agent: Teaching Laplace's Equation via
%    Low-Cost Experiments and Physics-Informed Neural Networks"
%
% Authors: Nestor Forero (analytical derivation and original code),
%          Alexander Agudelo Cardenas, Carlos Pena,
%          Esperanza Rodriguez Carmona
% ORCID:   0000-0003-0598-2317 (corresponding author)
% Contact: alexander.cardenas@esing.edu.co
%
% Repository: https://github.com/alexanderagudelocardenas/TEME-ECT-PINN-Laplace
% License:    MIT
%--------------------------------------------------------------------------
% DESCRIPTION
%--------------------------------------------------------------------------
% Closed-form (analytical) solution of the lambda-corrected Laplace
% equation
%
%       Uxx + Uyy = lambda * U
%
% for the parallel-bars electrode configuration, obtained via separation
% of variables and superposition of two Fourier series problems:
%
%   u1(x,y): satisfies u1(x,0)=0, u1(x,b)=1.5, u1(0,y)=0, u1(a,y)=0
%   u2(x,y): satisfies u2(0,y)=f, u2(a,y)=f, u2(x,0)=0, u2(x,b)=0
%
%   u(x,y) = u1(x,y) + u2(x,y)
%
% lambda was estimated experimentally from Tracker-digitized voltage data
% (see S3_data), averaging the finite-difference residual
% lambda_ij = (1/U_ij) * [ (U(i+1,j)-2U(i,j)+U(i-1,j))/hx^2
%                        + (U(i,j+1)-2U(i,j)+U(i,j-1))/hy^2 ]
% over all interior nodes: lambda = 0.029940 cm^-2 (n=30 Fourier terms).
%
% METHOD CORRESPONDENCE: this is Method M3 (White-box, lambda-Fourier) in
% the article's model progression table.
%
% *** OPEN ITEM - PLEASE CONFIRM BEFORE PUBLICATION ***
% The derivation text (source document) states the experimental domain as
% a = 12, b = 24. This code instead uses aa = 8, bb = 16 (see below).
% These values disagree and must be reconciled with Nestor before this
% script is treated as final for the article's reproducibility package.
%--------------------------------------------------------------------------

n = 30;                        % number of Fourier terms

alpha = (0.029940 + 0*1.75)^0.5;   % sqrt(lambda), lambda = 0.029940 cm^-2
%alpha = 1;
%alpha1 = abs(alpha);

% --- Fourier coefficients a_k (u2 branch, x-boundary conditions) --------
for i = 1:1:n
    a1 = pi*i;
    dd = -2*1.5/sinh(alpha);
    dd = dd*1/((alpha^2 + a1^2));
    dd = dd*(cosh(alpha)*sin(a1)*alpha + a1*sinh(alpha)*cos(a1));
    d(i) = dd;
end
%d'

% --- Fourier coefficients b_k (u1 branch, y-boundary condition) --------
for i = 1:1:n
    a1 = (alpha + (pi*(2*i-1))^2)^0.5;
    b(i) = 6/(pi*(2*i-1)*sinh(a1));
end
%b';

% --- Domain dimensions (cm) --- SEE OPEN ITEM ABOVE ---------------------
aa = 8;
bb = 16;

% --- Evaluation grid -----------------------------------------------------
x = [0:1/8:1];
y = x; %
y = [0:1/16:1];

[m1 n1] = size(x)
[m2 n2] = size(y)

x = aa*x;
y = bb*y;

j3 = 1;

for j1 = 1:1:n1
    for j2 = 1:1:n2
        suma = 0;

        % u2 contribution (a_k series)
        for i = 1:1:n
            c1 = (pi^2*i^2 + alpha).^0.5;
            c2 = c1*x(j1);
            a1 = c1*y(j2);
            dd2 = d(i); %*(1-cosh(c1))/sinh(c1)
            dd3 = (1 - cosh(c1))/sinh(c1);
            dd2 = dd2*(cosh(c2/aa) + dd3*sinh(c2/aa));
            suma = suma + dd2*sin(a1/bb); %
        end

        % u1 contribution (b_k series)
        for i = 1:1:n
            c1 = pi*(2*i-1);
            c2 = (c1^2 + alpha)^0.5;
            suma = suma + b(i)*sin(c1*x(j1)/aa).*sinh(c2*y(j2)/bb);
        end

        %suma = suma*10^(-0);
        AA(j1,j2) = suma;
        uu(j3,:) = [x(j1) y(j2) suma];
        j3 = j3 + 1;
        %plot3(x(j1),y(j2),suma,'.r')
        %hold on
    end
end

uu = uu';
%uu'
%
AA'

figure(1)
%plot3(24*uu(1,:),12*uu(2,:),uu(3,:),'.r')
plot3(uu(1,:), uu(2,:), uu(3,:), '.r')
xlabel('x')
ylabel('y')
zlabel('u(x,y)')
grid on
axis equal

[X Y] = meshgrid(x,y);

figure(2)
meshc(X, Y, AA');
xlabel('x')
ylabel('y')
zlabel('u(x,y)')
axis equal;
grid on;

end
