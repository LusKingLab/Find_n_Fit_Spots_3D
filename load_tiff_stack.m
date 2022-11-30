function [TLimages, Image_info]=load_tiff_stack(file_name)
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%function load_tiff_stack
%@author: Ivan Surovtsev 
%@date: 02.02.2012   
%@copyright 2012-2015 Yale University
%==========================================================================
% OUTPUT
% TLimages = Matlab multidimensional images, i.e. 3D matrix where 2D matrices are images, and 3rd
%          dimension is time
% Image_info = a 3-value array [w,h,n_frames], where w and h are image
%          width and heigth, and n_frames is a number of frames in TL
%
% INPUT
% file_name = valid file name (including path and extension), 
%              i.e.'V:\Ivan Surovtsev\Dir1\2011_12_28\IPTG 10uM\t_180min GFP.tif';
% ! Note, files should be nultidimensional tiff not stk.
%
%=========================================================================
% PURPOSE: This function loads time lapse stack into Matlab multidimensional images 
% Publication: specify Journal name and year such as eLife 2014
% Publication Author:
% Publication Title:
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

% LOAD IMAGES
info=imfinfo(file_name);
 im_w=info(1).Width;
 im_h=info(1).Height;
 frame_fin=length(info);
Image_info=[im_h,im_w,frame_fin]; 
  
  disp('loading images...');

bits=info(1).BitDepth;

bit_depth=['uint',num2str(bits)];
  
TLimages=zeros(im_h,im_w,frame_fin,bit_depth);
for frame=1:frame_fin
    TLimages(:,:,frame)=imread(file_name,frame);
end
  disp('done!');



end