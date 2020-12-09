function load_xml(pth,pthdata,pr)
% developed by Ashley Kiemen, Johns Hopkins University 2019
% this scripts reads the xml file generated by imagescope aperio systme
% first, check if annotation coordinates are already saved to annotation
% datafile.  If not, load them for all images and save them there.

% external MATLAB function utilized here:
% utilizes xml2struct2.m:
% Written by W. Falkena, ASTI, TUDelft, 21-08-2010
% Attribute parsing speed increased by 40% by A. Wanner, 14-6-2011
% Added CDATA support by I. Smirnov, 20-3-2012
% Modified by X. Mo, University of Wisconsin, 12-5-2012


disp('loading annotation data from xml files')

tic;
imlist = dir([pth,'*xml']);


for kz=1:length(imlist)
    datafile=[pthdata,imlist(kz).name(1:end-3),'mat'];
    disp(imlist(kz).name)
    
    if exist([pth,imlist(kz).name],'file')
        if exist(datafile,'file') % if mat file exists, reload only if xml file has been updated
            load(datafile,'dm');
            if contains(dm,imlist(kz).date);continue;end
        end
        stt=xml2struct2([pth,imlist(kz).name]);
        try    a=isfield(stt.Annotations.Annotation.Regions,'Region');b=1;
        catch; a=isfield(stt.Annotations.Annotation{1}.Regions,'Region');b=2;end
        if a==1
            if b==1 
                disp('one annotation layer detected')
                layernum=9;
                [xyout,reduce_annotations]=load_annotations_one_layer(stt,layernum);
            else  
                disp('multiple annotation layers detected')
                [xyout,reduce_annotations]=load_annotations_multiple_layers(stt);
            end
            reduce_annotations=pr/reduce_annotations;
            xyout(:,3:4)=xyout(:,3:4)/reduce_annotations; % make annotations 0.5um/pixel
            dm=imlist(kz).date;
            if exist(datafile,'file')
                save(datafile,'xyout','reduce_annotations','dm','-append');
            else
                save(datafile,'xyout','reduce_annotations','dm','-v7.3');
            end
            toc;
        else
            continue % skip if annotation file is empty
        end
    end
end

end

function [xyout,reduce_annotations]=load_annotations_one_layer(stt,layernum)
    reduce_annotations=str2double(stt.Annotations.Attributes.MicronsPerPixel);    
    regionnum=length(stt.Annotations.Annotation.Regions.Region);
    xyout=[];
    RegionInfo={};
    for kr=1:regionnum
        % check vertex number
        if regionnum>1
            vertexnum=length(stt.Annotations.Annotation.Regions.Region{kr}.Vertices.Vertex);
            RegionInfo{kr}=stt.Annotations.Annotation.Regions.Region{kr}.Attributes;
                for kv=1:vertexnum
                    x=str2num(stt.Annotations.Annotation.Regions.Region{kr}.Vertices.Vertex{kv}.Attributes.X);
                    y=str2num(stt.Annotations.Annotation.Regions.Region{kr}.Vertices.Vertex{kv}.Attributes.Y);
                    xyout=[xyout;[layernum kr x y]];
                end
        else
            vertexnum=length(stt.Annotations.Annotation.Regions.Region.Vertices.Vertex);
            RegionInfo{kr}=stt.Annotations.Annotation.Regions.Region.Attributes;
            for kv=1:vertexnum
                x=str2num(stt.Annotations.Annotation.Regions.Region.Vertices.Vertex{kv}.Attributes.X);
                y=str2num(stt.Annotations.Annotation.Regions.Region.Vertices.Vertex{kv}.Attributes.Y);
                xyout=[xyout;[layernum kr x y]];
            end
        end
    end
end

function [xyout,reduce_annotations]=load_annotations_multiple_layers(stt)
    xyout=cell([length(stt.Annotations.Annotation) 1]);
    reduce_annotations=str2double(stt.Annotations.Attributes.MicronsPerPixel);
    
    parfor layer=1:length(stt.Annotations.Annotation)
        % for each annotation layer
        if ~isfield(stt.Annotations.Annotation{layer}.Regions,'Region');continue;end
        regionnum=length(stt.Annotations.Annotation{layer}.Regions.Region);
        RegionInfo={};
        for kr=1:regionnum
            % for each individual annotation
            if regionnum>1
                vertexnum=length(stt.Annotations.Annotation{layer}.Regions.Region{kr}.Vertices.Vertex);
                RegionInfo{kr}=stt.Annotations.Annotation{layer}.Regions.Region{kr}.Attributes;
                    for kv=1:vertexnum
                        x=str2num(stt.Annotations.Annotation{layer}.Regions.Region{kr}.Vertices.Vertex{kv}.Attributes.X);
                        y=str2num(stt.Annotations.Annotation{layer}.Regions.Region{kr}.Vertices.Vertex{kv}.Attributes.Y);
                        xyout{layer}=[xyout{layer};[layer kr x y]];
                    end
            else
                vertexnum=length(stt.Annotations.Annotation{layer}.Regions.Region.Vertices.Vertex);
                RegionInfo{kr}=stt.Annotations.Annotation{layer}.Regions.Region.Attributes;
                for kv=1:vertexnum
                    x=str2num(stt.Annotations.Annotation{layer}.Regions.Region.Vertices.Vertex{kv}.Attributes.X);
                    y=str2num(stt.Annotations.Annotation{layer}.Regions.Region.Vertices.Vertex{kv}.Attributes.Y);
                    xyout{layer}=[xyout{layer};[layer kr x y]];
                end
            end
        end 
    end
    xyout=cell2mat(xyout);
end
