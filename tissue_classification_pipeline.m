function tissue_classification_pipeline(xlfile,which_case)
% developed by Ashley Kiemen, Johns Hopkins University 2019
% pipeline for the construction of a convolutional neural network
% for use in classifying images using manual annotation data


% external MATLAB function utilized here:
% utilizes xml2struct2.m:
% Written by W. Falkena, ASTI, TUDelft, 21-08-2010
% Attribute parsing speed increased by 40% by A. Wanner, 14-6-2011
% Added CDATA support by I. Smirnov, 20-3-2012
% Modified by X. Mo, University of Wisconsin, 12-5-2012


%% load info from excel file
[~,~,a]=xlsread(xlfile);
b=cell2mat(a(2:end,1));
b=find(b==which_case)+1;
b=a(b,2:end);

% 1. input settings (logical 0 or 1)
readfiles=b{1};
format_data=b{2};
make_imds=b{3};
save_data=b{4};
train_model=b{5};
class_im=b{6};

% path to image annotation data
pth=b{7};
pthdata=[pth,'mat files\'];
umpix=b{8}; % um/pixel of images used % 1=10x, 2=5x, 4=16x
CNN_outfolder=b{9};
gauss_sample=b{10}; % 1 if you want gaussian sampled data, 0 if not
pthim=b{11}; % path to tif images to classify
pthimds=b{12}; % path where to save imagedatastore CNN images
pthCNN=[pthdata,CNN_outfolder];

% define actions to take per annotation class
WS{1}=b{13};   % remove whitespace if 0, keep only whitespace if 1
WS{2}=b{14};                       % add removed whitespace to this class
WS{3}=b{15};   % rename classes accoring to this order
WS{4}=b{16};  % reverse priority of classes
WS{5}=b{17};
for k=1:length(WS)
    if isnan(WS{k});WS{k}=[];end
    if isstr(WS{k});WS{k}=str2num(WS{k});end
end 
skp=b{18};  % subsampling factor when classifying images
numpixels=b{19};

% design data inputs
if gauss_sample
    GS=200;
    [skipnum,sz]=ReduceSamplingW(2*GS+1,471);
else
    GS=110;sz=221;skipnum=1:221;
end
CNNset.sz=sz;
CNNset.GS=GS;
CNNset.skipnum=skipnum;
CNNset.classnum=length(unique(WS{3}));

% calculate tissue space
pthTA=calculate_tissue_space_082(pthim);
disp(pthimds)

%% make image datastore

% 1 path to mat files
if ~isfolder(pthdata);mkdir(pthdata);end
if ~isfolder(pthCNN);mkdir(pthCNN);end

% 2 caclulate svs resolutions & translate xml files to mat files
if readfiles
    % reads xml annotation files and saves as mat files
    load_xml(pth,pthdata,0.5);
end

% 3 fill annotation outlines and delete unwanted pixels
if format_data
    fill_annotations(pth,pthim,pthdata,WS,umpix,pthTA);
end

% 4 define datatiles of each class for deep learning model
if make_imds
    build_imds(pthdata,numpixels,CNNset,pthimds);
end

% 5 save datatiles to imagedatastore
if save_data
    save_imds(pthim,pthdata,pthimds);
end

%% load datatiles from indices and train model
if train_model
    % split datastore into training / validation / test set
    imds = imageDatastore(pthimds,'IncludeSubfolders',true,'LabelSource','foldernames');
    numpix0=length(imds.Labels)/length(unique(imds.Labels)); % total number of datapoints
    datnum=floor(numpix0*0.1);  % 80% for training, 10% for validation & testing
    [imdsTrain,imdsVal,imdsTest,~] = splitEachLabel(imds,datnum*8,datnum,datnum,'randomize');
    
    % train model
    make_CNN_imds(imdsTrain,imdsVal,imdsTest,pthCNN,CNNset,'zscore');
end

%% classify images
if class_im
    save([pthCNN,'net.mat'],'CNNset','-append');
    ww=0;   % label for pixels in classified image that are not tissue space
    if ~exist('diml','var');diml=[2 25];end%[1 200 500];
    p=15000; % number of pixels to classify at once (limited by computer memory)
    classify_image([pthCNN,'net.mat'],CNN_outfolder,pthim,skp,ww,pthTA,p);
end