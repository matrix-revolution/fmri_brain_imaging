function[min_weight, min_bias] = fmri_shooting(trngX, trngY, valX, valY)

disp(size(trngX))
disp(size(trngY))
disp(size(valX))
disp(size(valY))

[N, d] = size(trngX);
%d = 2500;
u(N,1) = 0;
u(:) = 1;
lambda_max = 2*max(abs(trngX'*(trngY- (1/N)*sum(trngY, 1)*u)));
disp(sum(trngY,1));
disp('lambda max');
disp(lambda_max);

x_ik_squared = trngX.*trngX; 
a = 2 * sum(x_ik_squared, 1);
disp('size of a')
disp(size(a))
k = 10;
b_old = 0;
w_old(d,1) = 0;
w_new(d,1) = 0;
training_RMSE(1:k) = 0;
validation_RMSE(1:k) = 0;
prev_valRMSE = 0;
total_non_zero(1:k) = 0;
lambda_values(1:k) = 0;
weight_values(1:k , 1:d) = 0;
bias_values(1:k) = 0;
for num=1:1
    prev_valRMSE = 0;
    lambda_max = 2*max(abs(trngX'*(trngY- (1/N)*sum(trngY, 1)*u)));
    for steps=1:k
        st = sprintf('steps %d .',steps);
        disp(st)
        not_converged = 1;
        count = 1;
        trRMSE_new = 0;
        while not_converged || count > 217640
            r_old = trngY - (trngX*w_old + b_old*u);
            b_new = (1/N)*(u'*r_old) + b_old;
            r_new = r_old - b_new*u + b_old*u;
            for i = d:-1:1
                c_i = 2*trngX(:,i)'*(r_new + trngX(:,i)*w_old(i)');
                if c_i < -lambda_max
                    w_new(i) = (c_i+lambda_max)/a(i);
                elseif -lambda_max <= c_i  && c_i <= lambda_max
                    w_new(i) = 0;
                elseif c_i > lambda_max
                    w_new(i) = (c_i-lambda_max)/a(i);
                end
                r_new = r_new + trngX(:,i)*(w_old(i) - w_new(i))';
            end
            w_old = w_new;
            b_old = b_new;

            count = count + 1;

            sum_squared_trR_new = r_new.^2;
            trRMSE_new = sqrt(mean(sum_squared_trR_new));

            val_r = valY - (valX'*w_old + b_old);
            sum_squared_val = val_r.^2;
            valRMSE = sqrt(mean(sum_squared_val));

            if prev_valRMSE ~=0 && (valRMSE > prev_valRMSE || abs(valRMSE-prev_valRMSE)< 0.005) 
                not_converged = 0;
            end
            prev_valRMSE = valRMSE;
        end
        lambda_values(steps) = lambda_max;
        disp(lambda_values)
        total_non_zero(steps) = nnz(w_old);
        training_RMSE(steps) = trRMSE_new;
        disp(training_RMSE)
        validation_RMSE(steps) = valRMSE;
        disp(validation_RMSE)
        lambda_max = lambda_max/2;
        weight_values(steps, :) = w_old;
        bias_values(steps) = b_old;        
    end
    
%     disp('w_old')
%     disp(w_old')
    disp('lambda')
    disp(lambda_values)
    disp('nnz')
    disp(total_non_zero)
    disp('training rmse')
    disp(training_RMSE)
    disp('validation rmse')
    disp(validation_RMSE)
 
end
[sortedVal, valIdx] = sort(validation_RMSE);
min_valRMSE = validation_RMSE(valIdx(1));
disp('min rmse')
disp(min_valRMSE)
min_weight = weight_values(valIdx(1), :);
min_weight = min_weight';
min_bias = bias_values(valIdx(1));
disp('lambda')
disp(lambda_values)
disp('nnz')
disp(total_non_zero)
disp('training rmse')
disp(training_RMSE)
disp('validation rmse')
disp(validation_RMSE)
x1 = lambda_values;
y1 = training_RMSE;
figure % new figure window
plot(x1,y1)

hold on

y2 = validation_RMSE;
plot(x1,y2)
xlabel('lambda values')
ylabel('training_rmse and validation_rmse')

hold off % reset hold state

figure
y3 = total_non_zero;
plot(x1, y3)
xlabel('lambda values')
ylabel('sparsity')

% filename = 'hw1data/featureTypes.txt';
% delimiterIn = '\n';
% A = importdata(filename,delimiterIn);
% [sorted_feature, idx_low] = sort(min_weight);
% disp('lowest rate features')
% disp(A(idx_low(1:10)))
% [sorted_feature1, idx_high] = sort(min_weight, 'descend');
% disp('highest rate features')
% disp(A(idx_high(1:10)))










