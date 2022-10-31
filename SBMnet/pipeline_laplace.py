import os
from scipy import misc
#from BG_median import BG_median
from BG_median import var_median
from BG_median import skew_median
from BG_median import kurtosis_median
from BG_median import gaussian_median
from BG_median import laplace_median
import imageio

def SBM_processing_pipeline(dataset_path, result_path):
    '''The pipeline for processing the SBMnet dataset. This pipeline will
        generate a 'results' folder in the 'result_path' to save one estimated
        background image for each video generated by a background modeling
        method.

        Modify the line 28 to call you method, in the form of:
        BG_result = YOUR_METHOD(video_path);
        where 'video_path' is the input path of a video;
              'BG_result' is the estimated background image.
        input:
            dataset_path: path of the SBMnet dataset folder;
            result_path: path of the results folder.''' 

    category_list = ['backgroundMotion', 'basic', 'clutter', 'illuminationChanges', 'intermittentMotion', 'jitter', 'veryLong', 'veryShort']

    for category in category_list:
        category_path = os.path.join(dataset_path, category)

        for video in os.listdir(category_path):
            print(['Now processing: "' + category + ' / ' + video + '" video'])
            video_path = os.path.join(category_path, video, 'input')

    ######################call your method#####################
            BG_result = laplace_median(video_path)

        #save the image
            result_video_path = os.path.join(result_path, 'results', category, video)

            if ~os.path.exists(result_video_path):
                os.makedirs(result_video_path)
                imageio.imwrite(os.path.join(result_video_path, 'RESULT_background.jpg'), BG_result)

SBM_processing_pipeline("./SBMnet_dataset", "./SBMnet_laplace")
