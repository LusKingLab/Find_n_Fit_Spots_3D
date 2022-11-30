function [spotData3, params, potentialXYZ] = findIrregularSpots3(image_stack, varargin)
% To detect spots regardless whether it is diffraction-limited or not.
% Finds 3D positon of spots centorids. Meant to be used to detect spots, and then estimate spots 
% shape, pixels, boundaries etc using a different function.
% if spots are roundish (i.e. diffreaction-limited) and fit radius (below) is tight around the peak,
% it will provide a very good good estimate of the position and resonably good approximation of intensity.
%
% locate spots that may or may not be diffraction-limited in provided images. 
% Finds local peaks and then constructs a shell of user determined thickness at radius away from the 
% brightest pixels and decide whether to keep it based on the ratio of the peak intensity 
% to the intensity of pixels within the shell. Then calculates centroid of the spot using pixel values
% within spesified fitRadius that are above quantileThreshold value and also uses these pixels values
% (minus background estiamted from pixels below the quantileThreshold)  to estimate spot Intensity
%
% INPUT
% image_stack     - matrix representing 3D image
% optional parameters for detection:
% peakRadius - a pixel will only be considered a local maxima (i.e. as a potential spot) if it is the
%             brightest peak within radius defined by peakRadius 
% edgeDist   - radius from the peak where the outter edge of the shell will be constructed
% shellThickness - thickness of the shell constructed at edgeDist to be used in intensity ratio calculation
% centerDist - Radius from the peak that will be used to calculate peak intensity. 
%             Intensity of all pixels within centerDist from the
%             peak will be used in the intensity ratio calculation. Should 
%             typically be set to small values (i.e., 1-2)
% intensityRatioThreshold - if the ratio if peak to shell intensity is less
%             than intensityRatioThreshold, the spot will be discarded
% fitRadius  - radius (in 3D sense) in pixels from the peak used to perform centroid and intensity
%             calculations
% quantileThreshold - value of quantile(e.g. 0.05 - 5-percintile) that determines threshold value for 
%            intensity and centroid calculations. The threshold value will be used as a backrgound estimate.
%            Pixels with values above will be used for centroid and intensity caulcations. 
%            Pixels below will be ignored even if they are within fitRadius.  and will be used to estimate background. 
%             This permits a large fit radius to be selected for irregular spots. Pixels to fit can then be
%             chosen based on intensity by modulating quantileThreshold
%
% OUTPUT:
% spotData3 - spots data, organized as a cell array, with each entry
% having
%            .spotPosition - xyz coordinates of the spot centroid
%            .intensityRatio - intensity Ratio of outer shell pixels to the cneter pixels;
%           .spotIntensity  - spot intenisty calculated as total intenisty of spots above quantileThrehsold within fitRadius ;
% params - structure conataing values of used parameters, date and version info
% potentialXYZ - coordinates of all local peaks intially identified before filtering using intenistyRatio (just for checking)
%
% Ivan Surovtsev, 2021.04.12
% based on the original script by 
% Brad Parry, 2015 October


%specify default settings in case the user specifies no additional values
fitRadius = 3;
edgeDist = 2.5;
centerDist = 1;
peakRadius = 11;
shellThickness = 1;
quantileThreshold = .1;
intensityRatioThreshold = 1.195;

version='2021.06.03';
% comparing to 00: 
% added params output
% additionof random noise to avoid "doubling of spots" when there are close
% bright pixels with the same value


for k = 1:length(varargin)

    if strcmpi(varargin{k},'fitRadius')
        fitRadius = varargin{k+1};
    elseif strcmpi(varargin{k},'edgeDist')
        edgeDist = varargin{k+1};
    elseif strcmpi(varargin{k},'centerDist')
        centerDist = varargin{k+1};
    elseif strcmpi(varargin{k},'peakRadius')
        peakRadius = varargin{k+1};
    elseif strcmpi(varargin{k},'shellThickness')
        shellThickness = varargin{k+1};
    elseif strcmpi(varargin{k},'quantileThreshold')
        quantileThreshold = varargin{k+1};
    elseif strcmpi(varargin{k},'intensityRatioThreshold')
        intensityRatioThreshold = varargin{k+1};
    end
end

%make sure a couple parameters are integers...
edgeDist = ceil(edgeDist);
centerDist = ceil(centerDist);

%construct a mophological filtering element and prepare it for
%morphCompFilter
se = ballElement3(ceil(peakRadius*2)+2, peakRadius);
center = floor(size(se,1)/2);
se = double(se);
se(se==0) = nan;
se(center+1,center+1) = nan;

%iterate through all frames present in meshData
%for F = 1:size(images,3)%length(cellList.meshData)
    
    % im = imread(images{F});
    im_0=image_stack;
    imsize_0 = size(im_0); %in principle, this could be done outside the loop, assuming image size never changes
    
    % added: padding image around edges to avoid "boundary" problem
    % and below removed corresponding part in "morphCompFilter3"
    mrgn=max([edgeDist,center,fitRadius]);
    im_class=class(im_0);
    im=eval([im_class,'(zeros([',num2str(imsize_0+mrgn*2),']));']);
    im_2=double(im);
    imsize=size(im);
    im(mrgn+1:imsize(1)-mrgn,mrgn+1:imsize(2)-mrgn,mrgn+1:imsize(3)-mrgn)=im_0;
    % adding small artificil noise so not to have to bright close pixel
    % with the same values...
    im_2(mrgn+1:imsize(1)-mrgn,mrgn+1:imsize(2)-mrgn,mrgn+1:imsize(3)-mrgn)=double(im_0)+rand(size(im_0)); 

    %perform initial estimation of peak locations
%     peaks = morphCompFilter3(im, se, '>=',mrgn);
    % let's use with small random noise instead:
    peaks = morphCompFilter3(im_2, se, '>=',mrgn);
    %initial guesses of peaks will be where peaks == 1, so get these indices
    % was [r,c] = find(peaks == 1);
    ind = find(peaks == 1);
    [r,c,z]=ind2sub(imsize,ind);
    % added to avoid conflict when edgeDist>peakRadius and to remove points
    % which are too close to the edges of the images to have shell with edgeDist around the peak 
%   Let's remove all previos delaing with "Boundary" problem    
%     ind1=( r>edgeDist & r<imsize(1)-edgeDist);
%      ind2=( c>edgeDist & c<imsize(2)-edgeDist);
%      ind3=( z>edgeDist & z<imsize(3)-edgeDist);
%     ind= ind1 & ind2 & ind3;
%     r=r(ind);
%      c=c(ind);
%      z=z(ind);
    potentialXYZ=[c-mrgn,r-mrgn,z-mrgn];

    %find the relative locations of pixels that are distance edgeDist away from
    %a bright pixel and within a thin shell of thickness shellThickness
    %determine the pixels that are within the shell...
    e = ballElement3(edgeDist*2+1,edgeDist);
    e1 = ballElement3(edgeDist*2+1,edgeDist-shellThickness);
    edgePixels = e - e1;
    %indicies of pixels located within the shell 
    % was [pixelShellR,pixelShellC] = find(edgePixels == 1);
    [pixelShellR,pixelShellC,pixelShellZ] = ind2sub(size(e),find(edgePixels == 1));
    if isempty(pixelShellR), error('shellThickness is too small, increase to a positive integer'), end
    %determine relative locations
    pixelShellR = pixelShellR - (edgeDist + 1);
    pixelShellC = pixelShellC - (edgeDist + 1);
    pixelShellZ = pixelShellZ - (edgeDist + 1);
    
    %determine the relative locations of pixels to be used in center
    %calculation: centerDist away from the central peak pixel
    e = ballElement3(ceil(centerDist)*2 + 1, centerDist);
    % was [centralR,centralC] = find(e == 1);
    [centralR,centralC,centralZ] = ind2sub(size(e),find(e == 1));
    %determine relative locations
    centralR = centralR - (centerDist + 1);
    centralC = centralC - (centerDist + 1);
    centralZ = centralZ - (centerDist + 1);

    %do an initial sort through the peaks and eliminate the ones that are
    %not at least as bright as the intensityRatioThreshold
    intensityRatio = nan(1,length(r));
    for k = 1:length(r)
        %convert subscripts to index for current peak after applying
        %offsets specified by pixelShellR and pixelShellC

        % added to remove artifical pading pixels from the ratio
        % calculation
        r1=r(k) + pixelShellR;
         c1=c(k) + pixelShellC;
         z1=z(k) + pixelShellZ;
        ind1=( r1>mrgn & r1<imsize(1)-mrgn);
         ind2=( c1>mrgn & c1<imsize(2)-mrgn);
         ind3=( z1>mrgn & z1<imsize(3)-mrgn);
        ind= ind1 & ind2 & ind3;
        r1=r1(ind);
         c1=c1(ind);
         z1=z1(ind);
        % and put it back into shellInds 
        shellInds = sub2ind(imsize,r1,c1,z1);
        centralInds = sub2ind(imsize,r(k) + centralR,c(k) + centralC,z(k) + centralZ);
    
        %compare the intensity in the center to the intensity in the outter
        %shell
        shellMean = mean(double(im(shellInds)));
        centralMean = mean(double(im(centralInds)));
        intensityRatio(k) = centralMean / shellMean;    
    end
    
    %spots with intensityRatios less than the threshold will be eliminated
    kill = intensityRatio < intensityRatioThreshold;
    intensityRatio(kill) = [];
    r(kill) = [];
    c(kill) = [];
    z(kill) = [];

    spots = [];
    pos = [];

% Changed to a simple output of the list of spots:

ii_spot=0;
spotData3={};
        for k = length(r):-1:1
            
            ii_spot=ii_spot+1;

            spot = double(im(r(k)-fitRadius:r(k)+fitRadius,c(k)-fitRadius:c(k)+fitRadius,z(k)-fitRadius:z(k)+fitRadius));
            %spotOrig = spot;
            %spots{end+1} = spot;
            spot = spot - quantile(spot(:),quantileThreshold);
            spot = (spot>0).*spot;

            pos(end+1,:) = g2d(spot) - (fitRadius+1) +  [c(k)-mrgn, r(k)-mrgn, z(k)-mrgn] ;
            
            spotData3{ii_spot}.spotPosition=pos(end,:);
            spotData3{ii_spot}.intensityRatio = intensityRatio(k);
            spotData3{ii_spot}.spotIntensity = sum(spot(:));

            %eliminate used spots
            r(k) = [];
            c(k) = [];
            z(k) = [];

        end

% prepare list of used params for output    
params.script=mfilename;
 params.date=date;
 params.version=version;
params.fitRadius = fitRadius;
 params.edgeDist = edgeDist;
 params.centerDist = centerDist;
 params.peakRadius = peakRadius;
 params.shellThickness = shellThickness;
 params.intensityRatioThreshold = intensityRatioThreshold;
 params.quantileThreshold = quantileThreshold;
 
end

function peaks = morphCompFilter3(im, se, operation,mrgn)
%im = image to be filtered
%
%se = morphological element to perform comparisons, matrix of Nan and 1
%the pixel at the center. The central value (as determined by center, see
%below), must be NaN or the operator should be one of [<>]=
%all dimensions of se must be odd and have the same length
%
%center = specifies the central location of se. center*2 + 1
%, will be broadcast center = center + [0,0]
%operation for comparison: >,<,<=,>=

if mod(size(se,1),2) ~= 1 || mod(size(se,2),2) ~= 1
    error('All dimensions of se must be of odd length')
end
if size(se,1) ~= size(se,2)
    error('Dimensions of se must be the same length')
end

%center needs to be chosen so that size of se = 2*center+1
center = floor(size(se,1)/2);
% was [x,y] = meshgrid(-center:center,-center:center);
[x,y,z] = meshgrid(-center:center,-center:center,-center:center);
%convert the structure element to an array of relative indices
x = x.*se;
y = y.*se;
z = z.*se;
%remove indices which are NaN
x(isnan(x)) = [];
y(isnan(y)) = [];
z(isnan(z)) = [];

%some housekeeping to make sure that we do not try to examine pixels that
%don't exist, that is, pixels that would be outside of the image bounds
% was
% ix0 = center;
% ix1 = size(im,1)-center+1;
% ix2 = size(im,2)-center+1;
% ix3 = size(im,3)-center+1;
% now let's use our margins since we paded the image:
 ix0 = mrgn;
 ix1 = size(im,1)-mrgn+1;
 ix2 = size(im,2)-mrgn+1;
 ix3 = size(im,3)-mrgn+1;
%finally extract the 'central' part of the image -- the structuring
%element, se, can be placed anywhere on this image without asking for
%values that do not exist in the full image
% was IMcenter = im(ix0:ix1,ix0:ix2);
IMcenter = im(ix0:ix1,ix0:ix2,ix0:ix3);

%initialze the array that will store the peaks
peaks = zeros(size(im));
peaks(ix0:ix1,ix0:ix2,ix0:ix3) = 1;


for k = 1:length(x)
    
    %The central part of the image (IMcenter) will
    %be held still and the full image will be shifted past according to the
    %shifts that were determined from the structuring element se. Each
    %shift is compared against IMcenter by the method specified by
    %operation. Successive shifted comparisons are multiplied together so
    %that all shifts of a given pixel must satisfy the condition determined
    %by operator for that main pixel to be kept as 1 in peaks. For a given
    %pixel r,c to satisfy the operation of all pixels in the region of r,c
    %(as determined by se) must be satisfied.
    
    %create relative indices
    R0 = ix0 + y(k);
    R1 = ix1 + y(k);
    C0 = ix0 + x(k);
    C1 = ix2 + x(k);
    Z0 = ix0 + z(k);
    Z1 = ix3 + z(k);

    if operation == '>'
        peaks(ix0:ix1,ix0:ix2,ix0:ix3) = peaks(ix0:ix1,ix0:ix2,ix0:ix3).* (IMcenter > im(R0:R1,C0:C1,Z0:Z1));
    elseif operation == '>='
        peaks(ix0:ix1,ix0:ix2,ix0:ix3) = peaks(ix0:ix1,ix0:ix2,ix0:ix3).* (IMcenter >= im(R0:R1,C0:C1,Z0:Z1));
    elseif operation == '<'
        peaks(ix0:ix1,ix0:ix2,ix0:ix3) = peaks(ix0:ix1,ix0:ix2,ix0:ix3).* (IMcenter < im(R0:R1,C0:C1,Z0:Z1));
    elseif operation == '<='
        peaks(ix0:ix1,ix0:ix2,ix0:ix3) = peaks(ix0:ix1,ix0:ix2,ix0:ix3).* (IMcenter >= im(R0:R1,C0:C1,Z0:Z1));
    end
    
end
end

function se = ballElement3(sz,radius)
% attempt to make operating on 3D images. Below, original Brad's notes
%
%a structuring element of size sz will be constructed so that all pixels within 
%radius of the center will be set to 1 and all pixels of distance > radius from 
%center will be 0

sz = floor(sz/2)*2+1;
center = ceil(sz/2);
[x,y,z] = meshgrid(1:sz,1:sz,1:sz);
se = ((x-center).^2 + (y-center).^2 + (z-center).^2).^(1/2) <= radius;
end

function [xyz] = g2d(image)
% attempt to make operating on 3D images. Below, original Brad's notes
%
%calculate the centroid, weighted mean of a 2d matrix
image = double(image);
M0 = sum(image(:));
%x = repmat(1:size(image,2),[size(image,1), 1]);
im_size=size(image);
[x,y,z] = meshgrid(1:im_size(2),1:im_size(1),1:im_size(3));
Mx = sum(x(:).*image(:));
%ay(:,1) = 1:size(image,1);
%y = repmat(ay,[1,size(image,2)]);
My = sum(y(:).*image(:));
Mz = sum(z(:).*image(:));
xyz = [Mx/M0, My/M0, Mz/M0];
end