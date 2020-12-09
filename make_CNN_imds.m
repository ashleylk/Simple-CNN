function accuracy=make_CNN_imds(imdsTrain,imdsVal,imdsTest,pthSave,CNNset,normIN)
% developed by Ashley Kiemen, Johns Hopkins University 2019
% trains convolutional neural network using saved image data store images

if isempty(pthSave);sv=0;else; sv=1;end % only save training progress if path given
if nargin<5;normIN='zerocenter';end
disp(['image normalization: ',normIN]);

% design model layers for CNN
% classnum=length(unique(imdsTrain.Labels));
% sz=size(readimage(imdsTrain,1),1);
classnum=CNNset.classnum;
sz=CNNset.sz;
lgraph=make_CNN_layers(sz,classnum,normIN);
warning('on','nnet_cnn:warning:GPULowOnMemory')
h=findall(groot,'Type','Figure');
close(h);

% train with pixels & key
options = trainingOptions('adam',...  % stochastic gradient descent solver
    'MaxEpochs',24,...
    'MiniBatchSize',128,... % datapoints per 'mini-batch' - ideally a small power of 2 (32, 64, 128, or 256)
    'Shuffle','every-epoch',...  % reallocate mini-batches each epoch (so min-batches are new mixtures of data)
    'ValidationData',imdsVal,...
    'ValidationPatience',5,... % stop training when validation data doesn't improve for __ iterations 5
    'InitialLearnRate',0.001,...  %     'InitialLearnRate',0.0005,...
    'LearnRateSchedule','piecewise',... % drop learning rate during training to prevent overfitting
    'LearnRateDropPeriod',1,... % drop learning rate every _ epochs
    'LearnRateDropFactor',0.75,... % multiply learning rate by this factor to drop it
    'ValidationFrequency',256,... % initial loss should be -ln( 1 / # classes )
    'OutputFcn', @(info)savetrainingplot(info,pthSave,sv),... % save training progress as image
    'Plots','training-progress','verbose',0); % view progress while training
%     'ExecutionEnvironment','multi-gpu',... % use all available gpu's for training
% net = trainNetwork(trainpix,trainkey,lgraph,options);
net = trainNetwork(imdsTrain,lgraph,options);

% validate with test & testkey
predicttest=classify(net,imdsTest);
testkey=imdsTest.Labels;
accuracy=sum(predicttest == testkey)/numel(testkey);
disp(['accuracy = : ',num2str(round(accuracy*100)),'%']);

% confusion matrix and network
h=figure;plotconfusion(testkey,predicttest);set(gcf,'color','w');
if sv
    saveas(h,[pthSave,'Confusion_matrix.png']);
    save([pthSave,'net.mat'],'net','lgraph','accuracy','CNNset','predicttest','-v7.3');
end
h=findall(groot,'Type','Figure');
close(h);


end



function stop=savetrainingplot(info,pthSave,sv)
stop=false;  %prevents this function from ending trainNetwork prematurely
if info.State=='done' && sv   %check if all iterations have completed
    saveas(findall(groot, 'Type', 'Figure'),[pthSave,'training_process.png'])
end


end






% net.Layers
% layer = 5;
% name = net.Layers(2).Name
% channels = 1:8;
% I = deepDreamImage(net,layer,channels, ...
%     'PyramidLevels',1);
% figure,montage(I),title(['Layer ',name,' Features'])