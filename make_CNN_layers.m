function lgraph=make_CNN_layers(sz,classnum,normIN)
% developed by Ashley Kiemen, Johns Hopkins University 2019
% design of sample CNN network - other pretrained networks also optional

if nargin==2;normIN='zerocenter';end


    layers = [
        imageInputLayer([sz sz 3],'Name','input','normalization',normIN) % 151x151x3
%         imageInputLayer([sz sz 3],'Name','input')  
%         imageInputLayer([sz sz 3],'Name','input','normalization','rescale-symmetric')
        convolution2dLayer(3,4,'Stride',1,'Padding','same','Name','conv1') % 151x151x4
        batchNormalizationLayer('Name','BN1')
        reluLayer('Name','ReLu1')
        
        convolution2dLayer(3,8,'Stride',1,'Padding','same','Name','conv2') % 151x151x8
        dropoutLayer(0.15,'Name','dropout1')
        batchNormalizationLayer('Name','BN2')
        reluLayer('Name','ReLu2')
        
        convolution2dLayer(3,8,'Stride',1,'Padding','same','Name','conv2b') % 151x151x8
        batchNormalizationLayer('Name','BN2b')
        reluLayer('Name','ReLu2b')
        
        convolution2dLayer(3,16,'Stride',2,'Name','conv2c') % 75x75x16
        batchNormalizationLayer('Name','BN2c')
        reluLayer('Name','ReLu2c')
        
        convolution2dLayer(3,16,'Stride',2,'Name','conv3a') % 75x75x16
        batchNormalizationLayer('Name','BN3a')
        reluLayer('Name','ReLu3a')
        
        additionLayer(3,'Name','add3')
        reluLayer('Name','ReLu4')
        
        convolution2dLayer(3,16,'Stride',1,'Padding','same','Name','conv5') % 75x75x32
        batchNormalizationLayer('Name','BN5')
        reluLayer('Name','ReLu5')
        
        convolution2dLayer(3,32,'Stride',1,'Padding','same','Name','conv5b') % 75x75x32
        batchNormalizationLayer('Name','BN5b')
        reluLayer('Name','ReLu5b')
        
        convolution2dLayer(3,32,'Stride',2,'Padding','same','Name','conv6') % 36x36x32
        batchNormalizationLayer('Name','BN6')
        reluLayer('Name','ReLu6')
        
        convolution2dLayer(3,32,'Stride',1,'Padding','same','Name','conv7') % 36x36x32
        batchNormalizationLayer('Name','BN7')
        reluLayer('Name','ReLu7')

        convolution2dLayer(3,32,'Stride',2,'Padding','same','Name','conv8') % 18x18x32
        batchNormalizationLayer('Name','BN8')
        reluLayer('Name','ReLu8')
        maxPooling2dLayer(2,'Stride',2,'Name','pool8')  % 9x9x32
        
        convolution2dLayer(3,64,'Stride',1,'Padding','same','Name','conv9') % 9x9x64
        dropoutLayer(0.15,'Name','dropout9')
        batchNormalizationLayer('Name','BN9')
        reluLayer('Name','ReLu9')
        
        convolution2dLayer(3,128,'Stride',1,'Padding','same','Name','conv10') % 9x9x32
        batchNormalizationLayer('Name','BN10')
        reluLayer('Name','ReLu10')
        maxPooling2dLayer(2,'Stride',2,'Name','pool10')  % 4x4x32
        %dropoutLayer(0.1,'Name','dropout_10')
        
        fullyConnectedLayer(10,'Name','fully_connected11') % 1x12
        batchNormalizationLayer('Name','BN11')
        reluLayer('Name','ReLu11')
        
        fullyConnectedLayer(classnum,'Name','fully_connected12')% 1 x #classes
        batchNormalizationLayer('Name','BN12')
        
        softmaxLayer('Name','softmax')
        classificationLayer('Name','output')];
    lgraph = layerGraph(layers);


    % add branch #1
    parallel1 = [
        convolution2dLayer(1,16,'Stride',1,'Padding','same','Name','conv3b')
        batchNormalizationLayer('Name','BN3b')
        convolution2dLayer(3,16,'Stride',2,'Name','conv3b2')
        batchNormalizationLayer('Name','BN3b2')
        reluLayer('Name','ReLu3b')];
    lgraph = addLayers(lgraph,parallel1);
    lgraph = connectLayers(lgraph,'ReLu2c','conv3b');
    lgraph = connectLayers(lgraph,'ReLu3b','add3/in3');

    % add branch #2
    parallel2 = [
        convolution2dLayer(1,16,'Stride',1,'Padding','same','Name','conv3c')
        batchNormalizationLayer('Name','BN3c')
        convolution2dLayer(3,16,'Stride',1,'Padding','same','Name','conv3c2')
        batchNormalizationLayer('Name','BN3c2')
        convolution2dLayer(3,16,'Stride',2,'Name','conv3c3')
        batchNormalizationLayer('Name','BN3c3')
        reluLayer('Name','ReLu3c')];
    lgraph = addLayers(lgraph,parallel2);
    lgraph = connectLayers(lgraph,'ReLu2c','conv3c');
    lgraph = connectLayers(lgraph,'ReLu3c','add3/in2');
%     figure(101),plot(lgraph);
end