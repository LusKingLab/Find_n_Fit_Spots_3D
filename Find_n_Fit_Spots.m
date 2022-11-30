
% the script was written to detect spots in the 3D image (without fitting), and then fit detected spots with a Gaussian to find positions and spot intenisty more precisely.

% REQUIRES:
%   load_tiff_stack
%   findIrregularSpots3
%   fitSpots3

%% INPUT DATA I: filenames, spots IDs, fit parameters etc

% FILENAME for image to analyze
file_image='C:\_Users_\Ivan\Projects\Chp7 Quantification Nick Linda 2021\Spots examples\E175_3265_count_04_R3D.tif';

%========================================================
%% INPUT DATA II: appearance etc
% for spots visualization

im_sc=[0, 0.5]; % image scaling as [black_px_intenisty, white_px_intensity]

%========================================================

%% LOAD IMAGE

% load(file_spots)
Fluo=load_tiff_stack(file_image);
n_frames=size(Fluo,3);


%========================================================
%% DETECT SPOTS
% to detect spots we need specify parameters (or default values will be used)
% need to play with parameters values to optmize detection
% First parameters to try to change are 'intensityRatioThreshold' and 'peakRadius'

% say we have more-or-less good parameters
param.peakRadius=3.5;
param.intensityRatioThreshold=1.1;
param.shellThickness=1;
param.edgeDist=7;
param.centerDist=1.5;
param.fitRadius=3;

% spot detection itself
disp('Starting spot detection...')
tic
[spotData, spotDetection_params]= findIrregularSpots3(Fluo,'peakRadius',param.peakRadius,...
    'intensityRatioThreshold',param.intensityRatioThreshold,...
    'shellThickness',param.shellThickness,...
    'edgeDist',param.edgeDist,...
    'centerDist',param.centerDist, ...
    'fitRadius',param.fitRadius);

disp('... DONE with the spot detection!!!')
toc

beep

%========================================================
%% FILTER SPOTS (if asked) by Intensity or Ratio
% For first iteration, skip filtering and check results of spot detection
% using next section
 % Then check whtheer you can remove some unwanted spots based on Spot Intensity or Intensity Ratio values
% if so, setup min level for these parameters and run filter
% It is posisble to extend this filter to exlcude spots with values above allowed max 

filter_spots=1; % whether to filter Spots by Intensity and IntensityRatio. If so, provide low_cutoffs
% cutoffs:
SI_cutoff=0.85*10000;
IR_cutoff=1.1;

if filter_spots
    spotData_F={};
    ss=0;
    % filter based on cutoffs
    for jj=1:length(spotData)
        if spotData{jj}.spotIntensity>SI_cutoff && spotData{jj}.intensityRatio>IR_cutoff
            ss=ss+1;
            spotData_F{ss}=spotData{jj};
        end
        
    end
    % keep track of filter pars
    filter_pars.SI_cutoff=SI_cutoff;
    filter_pars.IR_cutoff=IR_cutoff;
    %     keep old spotData, JIC
    spotData_0F=spotData;
    % and assign filtered list to spotData
    spotData=spotData_F;
end



%=========================================================================
%% SHOW RESULTS of spot detection/filtering
% This section wil generate max projection image and will show detected spots
% with or without values for  'spotIntensity' and 'intensityRatio'
% Note that image will be scaled according to parameters specified below

show_pars=1; % to show or not spotIntensity (top-right)  and intensityRatio (top-left) from spotList

% % generate max-projection
% image=max(Fluo, [], 3);

% setup the image scaling
im_min=min(Fluo(:));
im_max=max(Fluo(:));
im_range=im_max-im_min;
im_0=im_min+im_sc(1)*im_range;
im_1=im_min+im_sc(2)*im_range;

spotTable=spotData_2_Table(spotData);
figure;

str_spotID={};
str_spotSI={};
str_spotIR={};
    for ii=1:length(spotData)
        str_spotID{ii} = num2str(ii);
        str_spotSI{ii} = num2str(spotTable(ii, 5)/10000, 3);
        str_spotIR{ii} = num2str(spotTable(ii, 6), 3);
    end

 if show_pars
        disp('showing spot parameters: cyan = SpotIntensity/10000 red = IntensityRatio')
 end   
for frame=1:n_frames
    image= Fluo(:,:,frame);
     hold off
    imshow(image,[im_0,im_1],'InitialMagnification',200,'Border','tight');
     hold on
     set(gcf, 'Name', ['frame = ', num2str(frame)])
    
    ind = (round(spotTable(:, 4))==frame);

    plot(spotTable(ind, 2), spotTable(ind, 3), '+g', 'LineWidth', 2)
    text(spotTable(ind, 2)+2, spotTable(ind, 3)+2, str_spotID(ind),'Color', 'g', 'FontSize', 9)

    if show_pars
        %disp('showing spot parameters: cyan = SpotIntensity/10000 red = IntensityRatio')
        text(spotTable(ind, 2)+1, spotTable(ind, 3)-3, str_spotSI(ind), 'Color', 'c', 'FontSize',8)
            text(spotTable(ind, 2)-4, spotTable(ind, 3)-3, str_spotIR(ind),'Color','r','FontSize',8)
    end
    
    pause
end


%% FIT SPOTS 
% say we have a list of spots that we want to analyze further as an array
% goodSpots

good_spots=[141,128,75,47];

% JIC give fileterd spotData a different name
spotDataF=spotData(good_spots);

%
% FIT PARAMS
mrgn=10; % margin around the spot center  for the image cropping
R_fit=7;    % radius oc the circular area arounfd the spot center to be used for fitting

% VISUALIZATION params
im_sc= [0,1]; % contrast, in fractions of dynamic range of provided image (i.e. relAtive to max and min)
ms=4; % marker size for the 3D image/fit plot

% [new_spotData, params, FitResults_G, FitResults_GE] = fitSpots3(Fluo, spotDataF, 'marginRadius', mrgn, 'fitRadius', R_fit, 'fitWeight','unity');

% [new_spotData, params, FitResults_G, FitResults_GE] = fitSpots3(Fluo, spotDataF, 'marginRadius', mrgn, 'fitRadius', R_fit, 'fitWeight','sqrt');

[new_spotData, params, FitResults_G, FitResults_GE] = fitSpots3(Fluo, spotDataF, 'marginRadius', mrgn, 'fitRadius', R_fit, 'fitWeight','linear');




