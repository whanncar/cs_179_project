%Andrew Janzen GPU

load mnist_test
numImages = 1000;
numOutputNeurons = 10;
padBuffer = [2,2];
padSize = 32; %size(data(:,1,1)) + 2*padBuffer(1,1);

yin  = labels(1:numImages,:);

W_corr{i} = zeros(padSize,padSize);
count = zeros(numOutputNeurons ,1);
for i = 1:numOutputNeurons
    %initialize the convolutional weights for each layer
    W_sum{i} = zeros(padSize,padSize);
end

count = zeros(10,1);
for i = 1:numImages
    Z = padarray(data(:,:,i),padBuffer);
    count(yin(i)+1) = count(yin(i)+1) + 1;
    W_sum{yin(i)+1} = W_sum{yin(i)+1} + Z;
end

figure(1)
for i = 1:10
    subplot(4,4,i);
    image(W_sum{i}/count(i))
end


ytest = labels(numImages:numImages*2,:);
numTestSamples = length(ytest);

y_max = zeros(numTestSamples,1);

for i = 1:numTestSamples
    max = 0;
    maxNum = -100;
    Y = padarray(data(:,:,i + (numImages - 1)),padBuffer);
    for j = 1:numOutputNeurons
        [~,maxC] = corrNorm(W_sum{j},Y);  %222 errors using this method
        %[~,maxC] = corrNorm(X{j},Y);   %338
        
        %[~,maxC] = corrNorm(W_corr{j},Y);
        
        if(maxC > max)
            max = maxC;
            maxNum = j - 1; %0-9 rather than 1-10
        end
    end
    y_max(i) = maxNum;
    fprintf('%d\n',i);

end

Y_z = y_max - ytest;
nnzero = nnz(Y_z)
percentageCorrect = nnzero/numTestSamples


