function [ReducePoint,ReducePointNum]=ReduceSamplingW(PointNum,ReducePointNum,transf)
% developed by PeiHsun Wu, Johns Hopkins University 2019
% Create the skip sample calculate gaussian image window skipping system
% Create the monotonic but skipping sampling points in the range of r1 radius
% input
%     r1: radius of points (pixels) in windows
%     r2: desired number of points to reduce to ( note: its may be
%     different than actual sampling points, since program remove repeating
%     smapling points in integer).
%     transf: transfunction (or weighting function used for the sampling).

ux=(PointNum+1)/2;
x=[1:PointNum]-ux;
r1=max(x);
% x=-r1:1:r1;
if ~exist('transf')
    transf=[];
end
if isempty(transf)
    % the transfer function will be defined as 3time less size of r1; out
    % boundary radius
% single gaussian strategy
%     ra=r1/3;     
%     transf=1/sqrt(2*pi*ra^2)*exp(-x.^2/ra^2/(2));
%  Double gaussian strategy
    ra1=r1/2;
    ra2=r1/10;
    W1=2;
    W2=1;
    transf=W1*1/sqrt(2*pi*ra1^2)*exp(-x.^2/ra1^2/(2))+W2*1/sqrt(2*pi*ra2^2)*exp(-x.^2/ra2^2/(2));
end

transfcs=cumsum(transf);
ymin=min(transfcs);
ymax=max(transfcs);

colmin=find(transfcs(:)==ymin);
colmax=find(transfcs(:)==ymax);

colremove=[colmin(2:end);colmax(1:end-1)];
cc=ones(size(transfcs))==1;
cc(colremove)=0;


ys=linspace(ymin,ymax,ReducePointNum);
xs=interp1(transfcs(cc),x(cc),ys);
xsr=round(xs);
[uxsr,ia,~]=unique(xsr);  % removing the repeating pixels
uys=ys(ia);
ReducePoint=uxsr-min(uxsr)+1;
ReducePointNum=length(ReducePoint);

% checking the results output
% figure(922); clf;
% plot(x,transfcs);
% hold on;
% plot(xs,ys,'r+');
% plot(uxsr,uys,'c+');


return
   
end