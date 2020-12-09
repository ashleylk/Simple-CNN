function build_imds(pthdata,numpix,CNNset,outpth0)
% developed by Ashley Kiemen, Johns Hopkins University 2019
% designs image data store for CNN training from annotated samples

load([pthdata,'info.mat'],'infonm','infomat','classnum');
if ~isfolder(outpth0);mkdir(outpth0);end

% determine whether desired number of annotations exists
infomat=round(infomat*0.60); % no more than 60% of annotations from one image
tot=sum(infomat);
numpix=min([tot numpix]);
disp(numpix)

% define number of annotations per image
numper=zeros(size(infomat));
for k=1:size(infomat,2)
    col=infomat(:,k);
    totann=sum(infomat);
    if min(totann)==0;disp('some annotation layers missing');return;end
    
    app=ones(size(col));
    meannum=floor(numpix/sum(app));
    pp=min([col;meannum]);
    numper(app==1,k)=numper(app==1,k)+pp;
    
    count=0;
    while sum(numper(:,k))<numpix-length(app)
        % repeat with next lowest
        col=col-min(col(:));
        app(col==min(col(:)))=0;
        col(col==min(col(:)))=max(col(:));
        meannum=floor((numpix-sum(numper(:,k)))/sum(app));
        pp=min([col;meannum]);
        numper(app==1,k)=numper(app==1,k)+pp;
        count=count+1;
    end
    
    fix=numpix-sum(numper(:,k));
    if fix>0
        ii=find(app);
        kk=randperm(sum(app),fix);
        numper(ii(kk),k)=numper(ii(kk),k)+1;
    end
end

% make random list of indices
pixels=zeros([sum(numper(:)) 3]);
c0=1;
for kk=1:numel(infomat)
    [imnum,classn]=ind2sub(size(infomat),kk);
    if numper(kk)==0;continue;end % don't update c0 and c if numper is 0

    c=c0+numper(kk)-1;
    pixels(c0:c,1)=imnum;
    pixels(c0:c,2)=classn;
    pixels(c0:c,3)=randperm(infomat(kk),numper(kk))';
    c0=c+1;
end
CNNset.numclass=classnum;
save([outpth0,'imds.mat'],'numper','pixels','infonm','classnum','CNNset');

end