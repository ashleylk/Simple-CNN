function save_imds(pthim,pthdata,outpth0)
% developed by Ashley Kiemen, Johns Hopkins University 2019
% creates image data store images from image data store design

disp('building CNN data tiles')
if ~exist('outpth0','var');outpth0=[pthdata,'imds\'];end
% numper is grid of annotation numbers to load
% pixels is image number, annotation label, and index of annotations
load([outpth0,'imds.mat'],'numper','pixels','infonm','classnum','CNNset');
disp(numper)
disp(sum(numper))

tic;
% make image datastore outpths
for kp=1:classnum
    outpth=[outpth0,num2str(kp),'\'];
    if ~isfolder(outpth);mkdir(outpth);end
end

counts=zeros([1 classnum]);
for p=1:length(infonm) % for each image
    
    % load data and image
    load([pthdata,infonm{p},'.mat'],'ind');
    if isempty(ind);disp(['skipping image ',infonm{p}]);continue;end
    
    imnm=[infonm{p},'.tif'];
    I0=im2single(imread([pthim,imnm]));
    I1=I0(:,:,1);
    I2=I0(:,:,2);
    I3=I0(:,:,3);
    clearvars I0

    for k=unique(ind(:,2)') % for each annotation layer

        cc=ind(:,2)==k;
        [y,x]=ind2sub(size(I1),ind(cc,3));

        bb=min([length(y),numper(p,k)]);
        if bb==0;continue;end
        a=pixels(pixels(:,1)==p & pixels(:,2)==k,3);

        L=length(a);
        ii=get_local_window([x(a) y(a)],size(I1),CNNset.GS,CNNset.skipnum);
        Ig1=I1(ii);Ig2=I2(ii);Ig3=I3(ii);
        Ig=cat(3,Ig1,Ig2,Ig3);
        P1=sqrt(size(Ig,2));P2=size(Ig,1);
        Ig=permute(Ig,[2 3 1]);
        Ig=reshape(Ig,[P1 P1 3 P2]);
    
        for pk=1:P2
            im=Ig(:,:,:,pk);
            imwrite(im,[outpth0,num2str(k),'\',num2str(counts(k)+pk),'.png']);
        end
        counts(k)=counts(k)+pk;
        
    end
    disp(['image ',num2str(p),' of ',num2str(length(infonm)),' done.'])
end
toc;
end