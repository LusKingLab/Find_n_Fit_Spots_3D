function [new_spotData, params, FitResults] = fitSpots3(Image, spotData, varargin)

% Uses xyz positional information from spotData to choose best z-slice and cut small ROI from provided 3D Image 
% around previously detected spots.  Then it fits this 2D image with 2D Gaussian or 2D Gaussian + 2D erf 
% (errror function, to account of spot being at the edge fluorrescent background, such as edge of nucleus with some nuclear fluorescence)
%
% Check what input is provided as a spotData - 
% for now we will assume that output of findIrregularSpots3 is provided
% There is some flexibility in ROI size, use of weights for fitting etc. 

% INPUT:
% Image     - matrix representing 3D image
% spotData - structure array representing previously detected spots (for example output offindIrregularSpots3 )
%
% OPTIONAL parameters for fitting (should be provided as pair 'Name', value  when calling function, otherwise default values will be used):
%         parameter   default value
%           fitRadius = 9 - fit radius around dected spots, only pixels wihtin will be use din fitting 
%   marginRadius = 10 - margin around the spot center  for the image cropping (should be not smaller then fitRadius)
%            fitWeight = 'uniform' - weights  for pixel values  during fitting (other options - 'sqrt', 'linear')
%             fitModel  = 'Gauss' - fit model function (other options - 'Gauss+Erf', 'all')
%       showFitData = true - whether to show or not results of fitting (disabling greatly speeds up the process)
% showFitWindow = true - whether to show 3D data + fitting  (disabling greatly speeds up the process)
%        imageScale = [0,1] - relative (to image min and max) scaling of the image, for visual oupt only
%         markerSize = 6 - spot marker size
%             textPrefix = [] - text input to add to the plots
%             pauseOn = true - wthere to pause or not after fitting of each spot
%
% OUTPUT:
% new_spotData3 - spots data, organized same as spotData, as a cell array, with each entry
% having fields 
%            .spotPosition - xyz coordinates of the spot centroid
%            .intensityRatio - intensity Ratio of outer shell pixels to the cneter pixels;
%           .spotIntensity  - spot intenisty calculated as total intenisty of spots above quantileThrehsold within fitRadius ;
% params - structure conataing values of used parameters, date and version info
% FitResults  - results of fitting organized as a cell arary with each entry being a structure corresponding to one type of model with
% fields (each field is a table with index corresponding to the index of the spot in the spotData):
%             .coeff  - best fit values of all coefficients in the fitting model, in the same order
%             .conf_int(ii, :)  - confidence intervals for the best-fit values, two columns for low and upper bound, in the same order as above
%             .GoF(ii, :)  - 'Goodness of Fit' organized as  [average-relative-error, adjusted-R^2, Root-Mean-SquareError];
%             .spotInfo(ii, :) = [frame, ii];
%
% Ivan Surovtsev, 2021.07.02



%specify default settings in case the user specifies no additional values
fitRadius = 9;
marginRadius = 10;
imageScale = [0,1];
markerSize = 6;
fitWeight = 'uniform';
fitModel = 'Gauss';
showFitData = true;
showFitWindow = true;
textPrefix = [];
pauseOn = true;
%silent = false;

version='v0.2: 2022.08.20';

% v0.2: 
%       - made overall flow changes
%       - one fit at a time (unless 'all' selected, this optin is not finished yet, can do it only without visual output)
%       - added showFitWindow parameter

% check and substitute values for parameters that are provided in input
n_in = length(varargin);
for k = 1:n_in
    if strcmpi(varargin{k},'fitRadius')
        fitRadius = varargin{k+1};
    elseif strcmpi(varargin{k},'fitWeight')
        fitWeight = varargin{k+1};
    elseif strcmpi(varargin{k},'fitModel')
        fitModel = varargin{k+1};
    elseif strcmpi(varargin{k},'marginRadius')
        marginRadius = varargin{k+1};
    elseif strcmpi(varargin{k},'imageScale')
        imageScale = varargin{k+1};
    elseif strcmpi(varargin{k},'markerSize')
        markerSize = varargin{k+1};
    elseif strcmpi(varargin{k},'showFitData')
        showFitData = varargin{k+1};
    elseif strcmpi(varargin{k},'showFitWindow')
        showFitWindow = varargin{k+1};
    elseif strcmpi(varargin{k},'textPrefix')
        textPrefix = varargin{k+1};
    elseif strcmpi(varargin{k},'pauseOn')
        pauseOn = varargin{k+1};
%     elseif strcmpi(varargin{k},'silent')
%         silent = varargin{k+1};
    end
end


% image cropping params
mrgn = marginRadius; % margin around the spot center  for the image cropping
% spot fitting params
R_fit = fitRadius;    % radius of the circular area arounfd the spot center to be used for fitting

% VISUALIZATION params
im_sc = imageScale; % contrast, in fractions of dynamic range of provided image (i.e. relAtive to max and min)
ms = markerSize; % marker size for the 3D image/fit plot
% colors for spot position marker        
col_0 = [0, 1, 1]; % initial (input) position
col(1, :) = [0.5, 0., 0.7]; % color for the fit result 
col(2, :) = [0.7, 0, 0.]; % more colors in case more than one model is used
col_exp = [0.3, 0.3, 1]; 


% For future to have different inputs workable...
% but currently only one
input_type = 'findIrregularSpots3';


% PREPARATION of auxilaury variables etc
xx=1:2*mrgn+1;
XX=repmat(xx, 2*mrgn+1,1);
YY=XX';
RR2=(XX(:)-mrgn-1).^2+(YY(:)-mrgn-1).^2;

% siwcth field name for spot position depending on input. For fiture development
% currently only one type expected defined explicitly
switch input_type
    case 'findIrregularSpots3'
        spot_xyz = 'spotPosition';
    otherwise
        disp('Sorry, do not know what to do with data provided as a spotList')
        disp('spotData is expected to be an ouput from findIrregularSpots3')
        return
end

% check weights options
switch fitWeight
    case 'uniform'
        disp('using "uniform" weights for fitting')
    case 'sqrt'
        disp('using "sqrt" weights for fitting')
    case 'linear'
        disp('using "linear" weights for fitting');
    otherwise
        disp('Sorry such option is not provided')
        disp('currently only "sqrt"(default), "unity", "linear"  are available ')
        disp('using "uniform" weights now')
        fitWeight = 'uniform';
end

% check fit model options and define fitting function(s)
switch fitModel
    case 'Gauss'  % Gaussian +constant
        disp('using  z = Gaussian(x,y) + c  for fitting')
        fit_model = fittype('c+ b*exp(-(x-x0).^2/(2*sigma2)-(y-y0).^2/(2*sigma2))',...
                                        'dependent',{'z'},'independent',{'x','y'},...
                                        'coefficients',{'c',  'x0', 'y0',  'b', 'sigma2'});
        output_order = ['c, x0, y0, b, sigma2'];
         fit_model_1 = 'c+ b*exp(-(x-x0).^2/(2*sigma2)-(y-y0).^2/(2*sigma2))';
    case 'Gauss+Erf' % Gaussian+ erf for backround
        disp('using  z = Gaussian(x,y) + erf(x,y) + c  for fitting')
        fit_model = fittype('c+a*erf((cos(phi)*(x-x0)-sin(phi)*(y-y0))/gamma) + b*exp(-(x-x0).^2/(2*sigma2)-(y-y0).^2/(2*sigma2))',...
                                        'dependent',{'z'},'independent',{'x','y'},...
                                        'coefficients',{'c', 'x0', 'y0', 'b', 'sigma2', 'a', 'phi', 'gamma'});
        output_order = ['c, x0, y0, b, sigma2, a, phi, gamma'];
        fit_model_1= 'c+a*erf((cos(phi)*(x-x0)-sin(phi)*(y-y0))/gamma) + b*exp(-(x-x0).^2/(2*sigma2)-(y-y0).^2/(2*sigma2))';
    case 'all'
        disp('will fit using all available models:');
         disp('1. z = Gaussian(x,y) + c')
         disp('2.  z = Gaussian(x,y) + erf(x,y) + c')
         fit_model{1} = fittype('c+ b*exp(-(x-x0).^2/(2*sigma2)-(y-y0).^2/(2*sigma2))',...
                                     'dependent',{'z'},'independent',{'x','y'},...
                                     'coefficients',{'c',  'x0', 'y0',  'b', 'sigma2'});
           output_order{1} = ['c, x0, y0, b, sigma2'];
           fit_model_1{1} = 'c+ b*exp(-(x-x0).^2/(2*sigma2)-(y-y0).^2/(2*sigma2))';
         fit_model{2} = fittype('c+a*erf((cos(phi)*(x-x0)-sin(phi)*(y-y0))/gamma) + b*exp(-(x-x0).^2/(2*sigma2)-(y-y0).^2/(2*sigma2))',...
                                         'dependent',{'z'},'independent',{'x','y'},...
                                         'coefficients',{'c', 'x0', 'y0', 'b', 'sigma2', 'a', 'phi', 'gamma'});
           output_order{2} = ['c, x0, y0, b, sigma2, a, phi, gamma'];
           fit_model_1{2}= 'c+a*erf((cos(phi)*(x-x0)-sin(phi)*(y-y0))/gamma) + b*exp(-(x-x0).^2/(2*sigma2)-(y-y0).^2/(2*sigma2))';
otherwise
        disp('Sorry such option is not provided')
        disp('currently only "Gauss"(default), "Gauss+Erf"  are available ')
        disp('will use  z = Gaussian(x,y) + c  for fitting')
        fitModel = 'Gauss';
end

% prepare collectors for fit results
switch fitModel
    case {'Gauss', 'Gauss+Erf'}
        FitResults = [];
        one_model = true;
    case 'all'
        FitResults{1} = [];
        FitResults{2} = [];
        one_model = false;
end

% generating fig windows
if showFitWindow
    disp('x = original position estimate, + = best-fit estimate ')
    fig_spot = figure;
      movegui(fig_spot, 'northwest')
    fig_spot3D = figure;
      movegui(fig_spot3D, 'north')
    switch fitModel
        case {'Gauss', 'Gauss+Erf'} 
             fig_spot1D = figure;
               movegui(fig_spot1D, 'northeast')
        case 'all'
             fig_spot1D{1} = figure;
             fig_spot1D{2} = figure;
    end
end
    
%% SPOT FITTING  from spotData list

fitResults = {};
n_spots = length(spotData);
new_spotData = spotData;

for ii=1:n_spots
    
    disp(['Fitting spot#', num2str(ii)])
    
    %frame=spots_2_fit(ii, 1);
    %spotID=spots_2_fit(ii, 2);
    
    % get spot data
    spotData_1 = spotData{ii};
    spotPos = spotData_1.(spot_xyz);
    x = spotPos(1);
     y = spotPos(2);
     z = spotPos(3);
    pos = round([x,y]);
    frame = round(z);
    
    % CROP ROI from the image
    % pad image in case spot is close to the border
    image_z = Image(:, :, frame);
    im_size = size(image_z);
    im_class = class(image_z);
    imageX1 = eval([im_class,'(zeros([',num2str([im_size(1)+mrgn*2,im_size(2)+mrgn*2]),']));']);
    im_sizeX = size(imageX1);
    imageX1(mrgn+1:im_sizeX(1)-mrgn, mrgn+1:im_sizeX(2)-mrgn) = image_z;
    
    % crop ROI
    image_1 = imageX1(pos(2)+mrgn-mrgn:pos(2)+mrgn+mrgn, pos(1)-mrgn+mrgn:pos(1)+mrgn+mrgn);
    im_min = min(image_1(image_1>0));
    im_max = max(image_1(image_1>0));
    im_range = im_max-im_min;
    im_0 = im_min+im_sc(1)*im_range;
    im_1 = im_min+im_sc(2)*im_range;
    
    % FITTING
    % initial estimates for coefficients and their bounds:
    switch fitModel
        case 'Gauss'  % Gaussian +constant
            coeff_0= [min(image_1(:)), mrgn+1, mrgn+1, max(image_1(:))-median(image_1(:)), 3];
            lb = [0, 1, 1, 0, 0];
            ub = [2*max(image_1(:)), 2*mrgn+1, 2*mrgn+1,  Inf, mrgn^2];
        case 'Gauss+Erf' % Gaussian+ erf for backround
            coeff_0= [min(image_1(:)), mrgn+1, mrgn+1, max(image_1(:))-median(image_1(:)), 3, std(double(image_1(:))), 1, 2];
            lb = [0, 1, 1, 0, 0, 0, 0, 0];
            ub = [2*max(image_1(:)), 2*mrgn+1, 2*mrgn+1, Inf, mrgn^2, Inf, 2*pi, 2*mrgn];
        case 'all'
            coeff_0{1} = [min(image_1(:)), mrgn+1, mrgn+1, max(image_1(:))-median(image_1(:)), 3];
            lb{1} = [0, 0, 0, 0, 0];
            ub{1} = [2*max(image_1(:)), 2*mrgn+1, 2*mrgn+1,  Inf, 12.25];
            coeff_0{2} = [min(image_1(:)), mrgn+1, mrgn+1, max(image_1(:))-median(image_1(:)), 3, std(double(image_1(:))), 1, 2];
            lb{2} = [0, 0, 0, 0, 0, 0, 0, 0];
            ub{2} = [2*max(image_1(:)), 2*mrgn+1, 2*mrgn+1, Inf, mrgn^2, Inf, 2*pi, 2*mrgn];
    end
    %coeff_GE_0= [min(image_1(:)), std(double(image_1(:))), 1, mrgn+1, mrgn+1, 2, max(image_1(:))-median(image_1(:)), 3];
    %coeff_GE_0= [min(image_1(:)), 2*(max(image_1(:))-median(image_1(:))), 1, mrgn+1, mrgn+1, 2, max(image_1(:))-median(image_1(:)), 3];
    
    % convert px value into double as Matlab don't like integers sometime: 
    ZZ=double(image_1(:));
    
    % set weights
    switch fitWeight
        case 'uniform'
            WW = ones(size(XX));
        case 'sqrt'
            WW = sqrt(ZZ-min(ZZ));
        case 'linear'
            WW = ZZ-min(ZZ);
        otherwise
            WW = ones(size(XX));
    end
    
    % fit only pixel within radius R_fit
    ind = RR2<R_fit^2;
    
    % Finally fitting
    if one_model % fit with one chosen model
        [fit_res, gof, fit_info] = fit([XX(ind), YY(ind)], ZZ(ind), fit_model, 'Startpoint', coeff_0, 'Lower', lb, 'Upper', ub, 'Weights', WW(ind));
        if showFitData
            disp(fit_res)
        end
        % get coeff values, conf intervals and some goodness of fit measures
        coeff = coeffvalues(fit_res);
        coeff_int = confint(fit_res);
        delta_coeff_int = diff(coeff_int, 1);
        av_err_sum = sum(delta_coeff_int./coeff)/length(coeff);
        adjRsq = gof.adjrsquare;
        rmse = gof.rmse;
        % collect results
        FitResults.coeff(ii, :) = coeff;
        FitResults.conf_int(ii, :) = reshape(coeff_int, 1, 2*size(coeff_int,2));
        FitResults.GoF(ii, :) = [av_err_sum, adjRsq, rmse];
        FitResults.spotInfo(ii, :) = [frame, ii];
    else % when we want to try all available models then we need to make a choice which one is the best
        for jj = 1: length(fit_model) % cycle through all available models
            [fit_res{jj}, gof{jj}, fit_info{jj}] = fit([XX(ind), YY(ind)], ZZ(ind), fit_model{jj}, 'Startpoint', coeff_0{jj}, 'Lower', lb{jj}, 'Upper', ub{jj}, 'Weights', WW(ind));
            if showFitData
                disp(fit_res{jj})
            end
            % get coeff values, conf intervals and some goodness of fit measures
            coeff = coeffvalues(fit_res);
            coeff_int = confint(fit_res);
            delta_coeff_int = diff(coeff_int, 1);
            av_err_sum = sum(delta_coeff_int./coeff)/length(coeff);
            adjRsq = gof.adjrsquare;
            rmse = gof.rmse;
            % collect results
            FitResults{jj}.coeff(ii, :) = coeff;
            FitResults{jj}.conf_int(ii, :) = reshape(coeff_int, 1, 2*size(coeff_int,2));
            FitResults{jj}.GoF(ii, :) = [av_err_sum, adjRsq, rmse];
            FitResults{jj}.spotInfo(ii, :) = [frame, ii];
            % choice is not ready yet
        end
    end
    
    % ADD NEW POSITIONS, spot Intenisty, wdth etc to spotData  
    % coefficients for Gaussian (common for both models)
    c = coeff(1);
     x_0 = coeff(2);
     y_0 = coeff(3);
     b = coeff(4);
     sigma2 = coeff(5);
    % add spot Intenisty, wdth etc to spotData
    new_spotData{ii}.SI = 2*pi*b*sigma2; % Spot Intensity
     new_spotData{ii}.b = b; % background
     new_spotData{ii}.w = sqrt(sigma2); % spot width
     new_spotData{ii}.xyz = [pos(1)+x_0-(mrgn+1), pos(2)+y_0-(mrgn+1), z]; % updated positions
     new_spotData{ii}.fitModel = fit_model_1; % what model was used
     new_spotData{ii}.fit_coeff = coeff;
     new_spotData{ii}.coeff_conf_int = coeff_int;
     new_spotData{ii}.GoF = [av_err_sum, adjRsq, rmse];
    
    % SHOW spot and fitting results if asked
    if showFitWindow
        % calculate values for the best-fit curve
        if one_model
%             % coefficients common for both models
%             c = coeff(1);
%             x_0 = coeff(2);
%             y_0 = coeff(3);
%             b = coeff(4);
%             sigma2 = coeff(5);
            switch fitModel
                case 'Gauss'  % Gaussian +constant
                    ZZfit = c+ b*exp(-(XX-x_0).^2/(2*sigma2)-(YY-y_0).^2/(2*sigma2));
                case 'Gauss+Erf' % Gaussian+ erf for backround
                    a = coeff(6);
                    phi = coeff(7);
                    gamma = coeff(8);
                    ZZfit = c+a*erf((cos(phi)*(XX-x_0)-sin(phi)*(YY-y_0))/gamma) + b*exp(-(XX-x_0).^2/(2*sigma2)-(YY-y_0).^2/(2*sigma2));
            end
        else %case 'all' % NOT READY
            disp('not ready to show all results')

        end

        % 1. Show as an Image
        figure(fig_spot);
         set(gcf, 'Name', [textPrefix, 'z = ',num2str(frame), ' spot # ', num2str(ii)])
         hold off
        imshow(image_1, [im_0,im_1], 'InitialMagnification', 1600, 'Border', 'tight');
         hold on
           % add initial position 
        plot(mrgn+1+x-pos(1), mrgn+1+y-pos(2), 'x', 'Color', col_0)
           % add best-fit spot center
        figure(fig_spot);
        plot(x_0, y_0, '+','MarkerSize', ms, 'MarkerEdgeColor', col(1, :), 'MarkerFaceColor', col(1, :));
        
        % 2. Show image and fit data as a 3D plot
        figure(fig_spot3D);
         hold off
           % image data
        plot3(XX(:), YY(:), image_1(:), 'o', 'MarkerSize', ms, 'MarkerFaceColor', col_exp, 'MarkerEdgeColor', col_exp)
         hold on
           % add best fit surface
        mesh(XX, YY, ZZfit, 'EdgeColor', col(1, :));
         legend({'exp image', 'Gauss + c', 'Gauss + erf'}, 'Location', 'northeast' , 'FontSize', 10)
         set(gcf,'Name',[textPrefix, 'z = ',num2str(frame), ' spot # ',num2str(ii)])
        
        % 3. Show image and fit as an 1D-slice through the center of the fit
        figure(fig_spot1D);
         hold off
        ind_x= (XX==round(x_0));
         ind_y= (YY==round(y_0));
        % plot(xx, image_1(round(x_0),:),'o','MarkerSize',ms,'MarkerEdgeColor',col_X, 'MarkerFaceColor',colX);
        plot(xx, image_1(ind_x), 'o', 'MarkerSize', ms, 'MarkerEdgeColor', col_exp, 'MarkerFaceColor', col_exp);
        hold on
        plot(xx, image_1(ind_y), 'o', 'MarkerSize', ms, 'MarkerEdgeColor', col_exp, 'MarkerFaceColor', 'w');
        plot(xx, ZZfit(ind_x), '-', 'LineWidth', 1, 'Color', col(1, :));
        plot(xx, ZZfit(ind_y), ':', 'LineWidth', 1, 'Color', col(1, :));
         legend({'X-slice', 'Y-slice', 'Fit X-slice', 'Fit Y-slice'}, 'Location', 'northeast' , 'FontSize', 10)
         set(gcf,'Name',[textPrefix, 'z = ',num2str(frame), ' spot # ',num2str(ii)])
         %title (model_fit, 'FontSize', 10)
     
        disp('... paused... Press any key to proceed to the next spot ')
        if pauseOn
            pause
        end
    end
        
end

% close fig windows
 if showFitWindow
     close([fig_spot, fig_spot3D, fig_spot1D])
 end

% save parameters
params.script = mfilename;
params.date = date;
params.version = version;
params.fitRadius = fitRadius;
params.marginRadius = marginRadius;
params.imageScale = imageScale;
params.markerSize = markerSize;
params.fitWeight = fitWeight;
params.showFitData = showFitData;
params.showFitWindow = showFitWindow;

disp('... done with fitting')
disp('Fit curve equation:'), 
disp(fit_model)

disp ('FitResults .coeff and .conf_int contain best-fit coefficients values as colummns and low/up bounds as column pairs')
disp ('In the following order')
disp(output_order)

end


