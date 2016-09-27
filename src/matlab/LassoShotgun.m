% Shotgun : Parallel SCD
clear all;
format long;
load('Mug32_singlepixcam.mat'); lambda=0.05;
[N,d] = size(A);

x_org = zeros(d,1);
condition = true;

%difference of the lambda that Matlab solver uses and what we use
%Matlab use the division with N for L2 norm part
 
x=x_org;
%Normalising the data columnwise
A = normc(A);
AtA = A' * A;
ytA = y' * A;
 
% initialise
iter = 0;
P = 20;
rho=max(eigs(AtA));
P_opt=d/rho;

%One line function call for normalised data
deltaF = @(x) (x' * AtA - ytA) + lambda; 
beta = 1;% for LASSO
 
%weight=sqrt(sum(A.^2))=20.2485 for each column thus instead of vector we
%can use scalar for scaling the result. Otherwise we should normalise to
%the value where L-2 norm of lambda is lambda but weigths are individual 
%for the different features now it enoungh to divide the opt_x/20.2485  
 
% Experiment with two optimal values one from matlab lasso implementation
% this is not used because they have a different functional form than
% what is given in our paper.
x_opt  = lasso(A,y,'Lambda',lambda);
x_opt2 = solveLasso(y,A,lambda);

% LASSO 
F = @(x) 0.5 * (norm(A * x - y).^2)  +lambda * norm(x,1);

x_opt_collection = zeros(10, 16);

% run Shotgun 10 times
for index=1:10 
	iterations=[];
    % calculate for different number of threads P in each iteration
    % so that when the code finishes we have a set of results 
    % which can be plotted. To see if there is a linear decrease in the 
    % number of iterations taken to converge.
	P_Val=[1 2 4 6 8 10 20 30 40 50 60 70 80 90 100 110];
	for P=P_Val
    		normVal = [];
    		iter = 0;
    		condition= true;
            % clear x to be zeros(d,1)
    		x=x_org;
    		while condition
       			iter = iter + 1;
       			% choose random subset of P indexes from 1..d
       			% get the random weights x_i for i in the set P
       			randIdxs = randperm(d);
       			randPidxs = randIdxs(1:P);
       			x_subset = x(randPidxs);
 
       			deltaF_vals = deltaF(x);
       			deltaF_subset = deltaF_vals(randPidxs);

                % P parallel updates
       			parfor j = 1:P
          			 delta_xj = max(-1 * x_subset(j), -1 * deltaF_subset(j) / beta );
           			x_subset(j) = x_subset(j) + delta_xj;
     		  	end
       			x(randPidxs) = x_subset;
 
      		 	% In Fig 2: Y axis is # of iterations taken
	       		% to reach within 0.5% of true value x*
 
       			normVal = [(F(x) - F(x_opt2.beta))/F(x_opt2.beta) normVal];
       			condition = (normVal(1) > 0.005) && iter < 10000;

                % Print value of F(x) every 100 iterations to check for convergence
       			% temp = iter;
       			if mod(iter,100) == 0
           			fprintf('Iterations Taken = %d NormVal = %f\n',iter,F(x));
       			end
    		end
    	iterations=[iterations iter];
	end
	x_opt_collection(index,:) =  iterations;
end
save('x_opt_collection.m', 'x_opt_collection')
