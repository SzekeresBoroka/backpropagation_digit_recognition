readDigits = 36;
[xTrain, dTrain, xTest, dTest] = Load(readDigits);

% injecting bias
%xTrain = [ones(N,1) xTrain];

% defining activation function & its derivative
theta = @tansig;
dtheta = @(x) 1-theta(x).*theta(x);

% number of neurons in the consecutive layers
layers = [3 12 12 1];
%size(dTest)
%size(xTrain,3)
W = generic_backpropagation(xTrain, dTrain, layers, theta, dtheta, 5000, 0.01, 0.003);

y = zeros(size(xTest,1),size(xTest,2), readDigits);
v = zeros(size(xTest,1),size(xTest,2), readDigits);
for i = 1:readDigits
    in = xTest(:,:,i);
    [Y,V] = forward_propagation(in, W, theta);
    y(:,:,i) = Y{end};
    v(:,:,i) = V{end};
end                               
Draw(imgsTest, labelsTest)



function [xTrain, dTrain, xTest, dTest] = Load(readDigits)
    [xTrain, d1] = readMNIST('mnist_dataset/train-images.idx3-ubyte', ...
                                 'mnist_dataset/train-labels.idx1-ubyte', readDigits);
    [xTest, d2] = readMNIST('mnist_dataset/test-images.idx3-ubyte', ...
                               'mnist_dataset/test-labels.idx1-ubyte', readDigits);
%     outputs = eye(10,10);
%     dTrain = zeros(readDigits, 10);
%     dTest = zeros(readDigits, 10);
%     for i=1:readDigits
%        dTrain(i,:) = outputs(d1(i)+1,:); 
%        dTest(i,:) = outputs(d2(i)+1,:); 
%     end
    dTrain = d1;
    dTest = d2;
end

function Draw(imgs, labels)
    close all;
    nrows = 6;
    ncols = 6;
    for i = 1:nrows
        for j = 1:ncols
            k = (i-1)*ncols + j;
            subplot(nrows, ncols, k);
            imshow(imgs(:,:,k));
            xlabel(labels(k));
        end
    end
    set(gcf, 'Position', [50,211,560,690]);
    MinGui()
end

function MinGui()
    set(gcf(), 'MenuBar', 'none');
    set(gcf(), 'MenuBar', 'none');
end
