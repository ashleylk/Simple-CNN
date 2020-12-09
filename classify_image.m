function classify_image(datafile,outnm,pthim,skp,ww,pthTA,p)
% developed by Ashley Kiemen, Johns Hopkins University 2019
% classifies images using trained network

if ~exist('p','var');p=15000;end  % pixels to classify per iteration (memory constraint)

outpth=[pthim,'classification_for_',outnm,'\'];
if ~isfolder(outpth);mkdir(outpth);end
imlist=dir([pthim,'*tif']);

outpthscores=[outpth,'scores\'];
if ~isfolder(outpthscores);mkdir(outpthscores);end

tic;
disp('loading model')
load(datafile,'net','CNNset');
fprintf('\ntime to load CNN: %0.0f seconds \n\n',toc)


disp(['starting from image ',num2str(m),': ',imlist(m).name(1:end-4)]);
for k=1:length(imlist)
    outnm=[outpth,imlist(k).name];
    if isfile(outnm);disp(['skipped ',imlist(k).name]);continue;end
    displine=['starting classification of image %0.0f: ',imlist(k).name,' \n'];
    fprintf(displine,k)
    
    % get subscripts of tissue space only (not whitespace around tissue)
    % and classify every fourth pixel row and column to save time
    I=imread([pthim,imlist(k).name]);
    Itissue=imread([pthTA,imlist(k).name]);
    Itissue=imdilate(Itissue,strel('disk',7));
    Itissue=imfill(Itissue,'holes');
    Iskip=zeros(size(Itissue));
    GS2=CNNset.GS*2+2;
    Iskip(1:skp:end-GS2,1:skp:end-GS2)=1;
    Iskip(1:GS2,:)=0;Iskip(:,1:GS2)=0;
    [y,x]=find(Iskip & Itissue);
    
    I1=I(:,:,1);
    I2=I(:,:,2);
    I3=I(:,:,3);
    scores=zeros([size(I1) CNNset.classnum]);
    
    % iterate over tissue space in p subscript intervals
    numiteration=ceil(length(x)/p);
    gxy=[1:p:length(x)+1 length(x)+1];
    imclassify=zeros(size(I1));
    
    % calculate pixel window and classify
    tic;
    for A=1:numiteration
        % calculate window of 61x61 pixels (1:4:end from 201x201 window)
        gx=x(gxy(A):gxy(A+1)-1);
        gy=y(gxy(A):gxy(A+1)-1);
        xy=[gx(:),gy(:)];
        ii=getImLocalWindowInd_v2AK(xy,size(I1),CNNset.GS,CNNset.skipnum);
        Ig=cat(3,I1(ii),I2(ii),I3(ii));

        % reshape data to input to CNN      101 x 101 x 3(RGB) x #images
        P1=sqrt(size(Ig,2));P2=size(Ig,1);
        Ig=permute(Ig,[2 3 1]);       
        Ig=reshape(Ig,[P1 P1 3 P2]);  
        
        % classify
        [predicttest,sc] = classify(net,Ig);
        ind=sub2ind(size(I1),gy,gx);
        imclassify(ind)=predicttest;
        for kk=1:CNNset.classnum
           tmp=scores(:,:,kk);
           tmp(ind)=sc(:,kk);
           scores(:,:,kk)=tmp;
        end
        displine='iteration %0.0f of %0.0f.  time elapsed: %0.0f seconds\n';
        fprintf(displine,A,numiteration,toc)
    end
    
    scores=scores(1:skp:end,1:skp:end,:);
    imclassify=imclassify(1:skp:end,1:skp:end);
    imclassify(imclassify==0)=ww;
    imclassify=uint8(imclassify);
    imwrite(imclassify,outnm);
    save([outpthscores,imlist(k).name(1:end-4),'.mat'],'scores');
%     figure,
%         subplot(1,2,1),imagesc(imclassify),axis equal, axis off
%         subplot(1,2,2),imshow(I),title(k)
%         ha2=get(gcf,'children');
%         linkaxes(ha2)
    fprintf(['time to classify image ',imlist(k).name,': %0.0f seconds \n\n'],toc)
    clearvars -except net outpthscores datafile outpth imlist k p tic pth skp ww pthTA pthim CNNset
end