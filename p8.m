function NestedDissection_anysize()
close all
fsz = 12; % fontsize
% n = 30; % the set of unknown grid points is n-by-n
% [A,b,sol] = TestMatrixA(n + 2);

% %% load data
apprx_sol = load('standard_sol.mat').x;
data_down = readmatrix("maze_data_down.csv");
data_right = readmatrix("maze_data_right.csv");
data_down(1,:) = [];
data_down(:,1) = [];
data_right(1,:) = [];
data_right(:,1) = [];

%% draw maze
fig = 1;
draw_maze(data_down,data_right,fig)
% each maze cell is a node
% generate adjacency matrix
[m,n] = size(data_down);
N = m*n;
A = make_adjacency_matrix(data_down,data_right);

%% convert adjacency matrix to stochastic matrix 
row_sums = sum(A,2); % find rowsums of A
R = spdiags(1./row_sums,0,N,N);
P = R*A;

%% setup linear system for the committor problem
exitA = 1;
exitB = N;
x = zeros(N,1);
x(exitB) = 1;
I = speye(N);
L = P - I;
b = -L*x;
ind_exits = union(exitA,exitB);
ind_unknown = setdiff((1:N),ind_exits);

%% make the system symmetric positive definite
% L = R^{-1}A - I
% R^{1/2}LR^{-1/2} = R^{1/2}AR^{1/2} - I is symmetric
% Lsymm = R^{1/2}LR^{-1/2}, bsymm = R^{1/2}*b
% D = R^{1/2}
r_sqrt = sqrt(row_sums);
D = spdiags(r_sqrt,0,N,N);
Dinv = spdiags(1./r_sqrt,0,N,N);
Lsymm = D*L*Dinv;
bsymm = D*b;

%% Nested Dissection
% Nested Dissection algorithm solves Ax=b
% It rearranges A using a permutation matrix P that it constructs
% Note: P' = P^{-1}
% Hence it solves PAP'Px = Pb
% It decomposes PAP' = LU
% Hence we need to solve LUPx = Pb
% It first solves Ly = Pb by y = L\(Pb)
% Then it solves UPx = y by x = P'(U\y)

%%
% The grid size is n-by-n
level = 0;
[L,U,P,A] = MyDissection(-Lsymm,m,n,level);
y = L\(P*(-bsymm));
x = P'*(U\y);
fprintf('norm diff = %d\n',norm(x - apprx_sol));
figure;
spy(A);
set(gca,'fontsize',fsz);
grid
title(sprintf('Sparsity pattern of P*A*P^T for nx = %d, ny = %d',n,n),'Fontsize',20);
axis ij

end

%%
function [L,U,P,A] = MyDissection(A,nx_grid,ny_grid,level)
A0 = A;
[m,n] = size(A);
if m ~= n
    fprintf("A is not square: (%d-by-%d)\n",m,n);
    L=0;
    U=0;
    P=0;
    A=0;
    return
end
% if level is even do vertical split
% if level is odd do horizontal split
N_grid = nx_grid*ny_grid; % the grid size
par = mod(level,2); % parity
switch par
    case 0 % vertical split
        if nx_grid >= 3
            nx_Omega1 = floor(nx_grid/2); % # of columns in Omega1
            N_Omega1 = nx_Omega1*ny_grid; % |Omega1| 
            ind_Omega3 = N_Omega1 + 1 : N_Omega1 + ny_grid; % indices of Omega3
            N_Omega3 = length(ind_Omega3);
            ind_Omega1 = 1 : N_Omega1; % indices of Omega1
            ind_Omega2 = N_Omega1 + N_Omega3 + 1 : N_grid; % indices of Omega2
            N_Omega2 = length(ind_Omega2);
            ny_Omega1 = ny_grid;
            nx_Omega2 = nx_grid-nx_Omega1-1;
            ny_Omega2 = ny_grid; 
        else
            % [L,U] = lu(A);
            L = ichol(A);
            U = L';
            P = speye(N_grid);
            return
        end    
    case 1 % horizontal split
        if ny_grid >= 3
            ny_Omega1 = floor(ny_grid/2); % # of rows in Omega1
            N_Omega1 = ny_Omega1*nx_grid; % |Omega1|
            ind_Omega3 = ny_Omega1 + 1 : ny_grid : N_grid; % indices of Omega3
            N_Omega3 = length(ind_Omega3);
            [ii,jj] = meshgrid(1 : ny_Omega1,1 : nx_grid);
            ind_Omega1 = sort(sub2ind([ny_grid,nx_grid],ii(:)',jj(:)'),'ascend'); % indices of Omega1
            [ii,jj] = meshgrid(ny_Omega1 + 2 : ny_grid,1 : nx_grid);
            ind_Omega2 = sort(sub2ind([ny_grid,nx_grid],ii(:)',jj(:)'),'ascend'); % indices of Omega2 
            N_Omega2 = length(ind_Omega2);
            nx_Omega1 = nx_grid;
            nx_Omega2 = nx_grid;
            ny_Omega2 = ny_grid-ny_Omega1-1; 
        else
            % [L,U] = lu(A);
            L = ichol(A);
            U = L';
            P = speye(N_grid);
            return
        end    
    otherwise
        fprintf('Error: par = %d\n',par);
        return
end
% fprintf('size(A) = [%d,%d], nxy = %d, N_Omega1 = %d, N_Omega2 = %d, N_Omega3 = %d\n',...
%     size(A,1),size(A,2),ny_grid,N_grid,N_Omega1,N_Omega2,N_Omega3);
% fprintf('ind_Omega2(1) = %d, ind_Omega2(end) = %d\n',ind_Omega2(1),ind_Omega2(end));
A11 = A(ind_Omega1,ind_Omega1);
A22 = A(ind_Omega2,ind_Omega2);

[L11,U11,P11,~] = MyDissection(A11,nx_Omega1,ny_Omega1,level + 1);
[L22,U22,P22,~] = MyDissection(A22,nx_Omega2,ny_Omega2,level + 1);

P1 = speye(N_grid);
P1(ind_Omega1,ind_Omega1) = P11;
P1(ind_Omega2,ind_Omega2) = P22;
% set up the permutation matrix P
% this command puts ones in positions 
% with indices (1 : nxy,[ind1(:)',ind2(:)',ind(:)'])
P = sparse(1 : N_grid,[ind_Omega1(:)',ind_Omega2(:)',ind_Omega3(:)'],ones(1,N_grid));
P = P*P1;
A = P*A0*P';
% extract nonzero blocks of A
A11 = A(1 : N_Omega1,1 : N_Omega1);
istart2 = N_Omega1 + 1;
ifinish2 = N_Omega1 + N_Omega2;
istart3 = ifinish2 + 1;
A22 = A(istart2 : ifinish2,istart2 : ifinish2);
A13 = A(1 : N_Omega1,istart3 : end);
A23 = A(istart2 : ifinish2,istart3 : end);
A31 = A(istart3 : end,1 : N_Omega1);
A32 = A(istart3 : end,istart2 : ifinish2);
A33 = A(istart3 : end,istart3 : end);
% compute the Schur complement
S33 = A33 - A31*(U11\(L11\A13)) - A32*(U22\(L22\A23));
% compute LU factorization of S33
[L33,U33] = lu(S33);
% form the LU decomposition of A
L = sparse(N_grid,N_grid);
L(1 : N_Omega1,1 : N_Omega1) = L11;
L(istart2 : ifinish2,istart2 : ifinish2) = L22;
L(istart3 : end,istart3 : end) = L33;
L(istart3 : end,1 : N_Omega1) = (U11'\A31')';
L(istart3 : end,istart2 : ifinish2) = (U22'\A32')';
U = sparse(N_grid,N_grid);
U(1 : N_Omega1,1 : N_Omega1) = U11;
U(istart2 : ifinish2,istart2 : ifinish2) = U22;
U(istart3 : end,istart3 : end) = U33;
U(1 : N_Omega1,istart3 : end) = L11\A13;
U(istart2 : ifinish2,istart3 : end) = L22\A23;

% fprintf('nx = %d, ny = %d, level = %d, size(A) = [%d,%d]\n',nx_grid,ny_grid,level,n,n);
% fprintf('norm(full(A - L*U)) = %d\n',norm(full(A - L*U)));
end

%% make the matrix, the right-hand side, and the exact solution for 
function [A,b,sol] = TestMatrixA(n)
% Solving PDE -grad * (a(x,y) * grad(u)) = f(x,y)
% in the domain D = [0,1]^2 discretized to an n-by-n grid
% Boundary condition: u = 0 on the boundary of D
n1 = n - 1;
h = 1/n1;
Iinner = 2 : n - 1;

t = linspace(0,1,n);
[x,y] = meshgrid(t,t);
% set f(x,y) so that the exact solution is the given one 
uexact = x.*(1-x).*y.*(1-y); % u exact
a = 1 + x + 2*y.^2;
f = diff_oper(h,uexact,a);

A = make_matrix(a,h);
b = f(:);

uinner = uexact(Iinner,Iinner);
sol = uinner(:);
end
%%
function Lu = diff_oper(h,u,a)
n = length(u);

    as = 0.5*(a + circshift(a,[1,0]))/h^2;
    an = 0.5*(a + circshift(a,[-1,0]))/h^2;
    aw = 0.5*(a + circshift(a,[0,1]))/h^2;
    ae = 0.5*(a + circshift(a,[0,-1]))/h^2;
    ap = aw + ae + as + an;
    I = 2 : n - 1;

Lu = ap(I,I).*u(I,I) - as(I,I).*u(I - 1,I) ...
    - an(I,I).*u(I + 1,I) - aw(I,I).*u(I,I - 1)...
    - ae(I,I).*u(I,I + 1);
end

%%
function A = make_matrix(a,h)
h2 = h^2;
n = length(a);
n2 = n - 2;
cN = 0.5*(a + circshift(a,[-1 0]))/h2;
cS = 0.5*(a + circshift(a,[1 0]))/h2;
cW = 0.5*(a + circshift(a,[0 1]))/h2;
cE = 0.5*(a + circshift(a,[0 -1]))/h2;
cP = -(cE + cW + cN + cS);
Iinner = 2 : n - 1;
an = -cN(Iinner,Iinner);
as = -cS(Iinner,Iinner);
aw = -cW(Iinner,Iinner);
ae = -cE(Iinner,Iinner);
ap = -(an + as + ae + aw);
% Conversion to column vectors
cn = an(:); cs = as(:); cw = aw(:); ce = ae(:); cp = ap(:);

% First set up the mathrix A without taking care of B and D
A = spdiags([circshift(cw,[-n2 0]),circshift(cs,[-1 0]),cp,...
    circshift(cn,[1 0]),circshift(ce,[n2 0])],[-n2,-1,0,1,n2],n2*n2,n2*n2);
% Take care of the Dirichlet BCs: u_y(x,0) = u_y(x,1) = 0
for j = 1 : n2 - 1 
    A(n2*j,n2*j + 1) = 0;
    A(n2*j + 1,n2*j) = 0;
end
end

function draw_maze(data_down,data_right,fig)
[m,n] = size(data_down);
figure(fig);
hold on;
line_width = 3;
col = 'k';
% plot outer lines
plot(0.5+(1:n),0.5+zeros(1,n),'color',col,'Linewidth',line_width);
plot(0.5+(0:n-1),0.5+m*ones(1,n),'color',col,'Linewidth',line_width);
plot(0.5+zeros(1,m),0.5+(1:m),'color',col,'Linewidth',line_width);
plot(0.5+m*ones(1,n),0.5+(0:m-1),'color',col,'Linewidth',line_width);
% plot vertical lines
for i = 1 : m
    for j = 1 : n-1
        if data_right(i,j) == 0
            plot(0.5+[j,j],0.5+[i-1,i],'color',col,'Linewidth',line_width);
        end
    end
end
% plot horizontal lines
for j = 1 : n
    for i = 1 : m-1
        if data_down(i,j) == 0
            plot(0.5+[j-1,j],0.5+[i,i],'color',col,'Linewidth',line_width);
        end
    end
end
axis ij
axis off
daspect([1,1,1])
end

function A = make_adjacency_matrix(data_down,data_right)
[m,n] = size(data_down);
mn = m*n;
A = sparse(mn);
for i = 1 : m
    for j = 1 : n-1
        if data_right(i,j) == 1
            ind = (j-1)*m + i;
            A(ind,ind+m) = 1;
            A(ind+m,ind) = 1;
        end
    end
end
for j = 1 : n
    for i = 1 : m-1
        if data_down(i,j) == 1
            ind = (j-1)*m + i;
            A(ind,ind+1) = 1;
            A(ind+1,ind) = 1;
        end
    end
end
end

        
        
        
        
        

        
