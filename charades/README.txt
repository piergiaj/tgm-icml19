###########################################################
     ________  _____    ____  ___    ____  ___________
    / ____/ / / /   |  / __ \/   |  / __ \/ ____/ ___/
   / /   / /_/ / /| | / /_/ / /| | / / / / __/  \__ \ 
  / /___/ __  / ___ |/ _, _/ ___ |/ /_/ / /___ ___/ / 
  \____/_/ /_/_/  |_/_/ |_/_/  |_/_____/_____//____/  
                                                    
###########################################################

The Charades Dataset
allenai.org/plato/charades/
Initial Release, June 2016

Gunnar A. Sigurdsson
Gul Varol
Xiaolong Wang
Ivan Laptev
Ali Farhadi
Abhinav Gupta

If this work helps your research, please cite:
@article{sigurdsson2016hollywood,
author = {Gunnar A. Sigurdsson and G{\"u}l Varol and Xiaolong Wang and Ivan Laptev and Ali Farhadi and Abhinav Gupta},
title = {Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding},
journal = {ArXiv e-prints},
eprint = {1604.01753}, 
year = {2016},
url = {http://arxiv.org/abs/1604.01753},
}

Relevant files:
README.txt (this file)
license.txt (the license file, this must be included)
Charades.zip:
  Charades_v1_train.csv (the training annotations)
  Charades_v1_test.csv (the testing annotations)
  Charades_v1_classes.txt (the classes)
  Charades_v1_objectclasses.txt (the primary object classes)
  Charades_v1_verbclasses.txt (the primary verb classes)
  Charades_v1_mapping.txt (mapping from activity to object and verb)
  Charades_v1_classify.m (evaluation code for video-level classification)
  Charades_v1_localize.m (evaluation code for temporal action detection)
  test_submission_classify.txt (example test output to evaluate an algorithm)
  test_submission_localize.txt (example test output to evaluate an algorithm)
Charades_v1.zip (the videos)
Charades_caption.zip (contains evaluation code for caption generation)
Charades_v1_rgb.tar (the videos stored as jpg frames at 24 fps)
Charades_v1_flow.tar (the flow stored as jpg frames at 24 fps)
Charades_v1_features_rgb.tar.gz (fc7 features from the RGB stream of a Two-Stream network)
Charades_v1_features_flow.tar.gz (fc7 features from the Flow stream of a Two-Stream network)
Please refer to the website to download any missing files.


###########################################################
Charades_v1.zip 
###########################################################
The zipfile contains videos encoded in H.264/MPEG-4 AVC (mp4) using ffmpeg:
ffmpeg -i input.ext -vcodec libx264 -crf 23 -c:a aac -strict -2 -pix_fmt yuv420p output.mp4
The videos, originally in various formats, maintain their original resolutions and framerates.


###########################################################
Charades_v1_classes.txt 
###########################################################
Contains each class label (starting at c000) followed by a human-readable description of the action, such as "c008 Opening a door"


###########################################################
Charades_v1_train.csv and Charades_v1_test.csv
###########################################################
A comma-seperated csv, where a field may be enclosed by double quotation marks (") in case it contains a comma. If a field has multiple values, such as multiple actions, those are seperated by semicolon (;). The file contains the following fields:

- id:
Unique identifier for each video.
- subject:
Unique identifier for each subject in the dataset
- scene:
One of 15 indoor scenes in the dataset, such as Kitchen
- quality:
The quality of the video judged by an annotator (7-point scale, 7=high quality)
- relevance: 
The relevance of the video to the script judged by an annotated (7-point scale, 7=very relevant)
- verified:
'Yes' if an annotator successfully verified that the video matches the script, else 'No'
- script:
The human-generated script used to generate the video
- descriptions:
Semicolon-separated list of descriptions by annotators watching the video
- actions:  
Semicolon-separated list of "class start end" triplets for each actions in the video, such as c092 11.90 21.20;c147 0.00 12.60
- length:
The length of the video in seconds

This can be loaded into MATLAB as follows:

f = fopen('Charades_v1_train.csv');
header = textscan(f,repmat('%s ',[1 10]),1,'Delimiter',',');
csv = textscan(f,repmat('%q ',[1 10]),'Delimiter',',');
actions = csv{10};
actions_in_first_video = regexp(actions{1},';','split');


This can be loaded into python as:

import csv
with open('Charades_v1_train.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        actions = row['actions'].split(';')

Please refer to the evaluation code for usage examples.


###########################################################
Charades_v1_classify.m
###########################################################
Evaluation code for video-level classification. Each video has zero or more actions. This script takes in a "submission file" which is a csv file of the form:

id vector

where 'id' is a video id for a given video, and 'vector' is a whitespace delimited list of 157 floating point numbers representing the scores for each action in a video. An example submission file is provided in test_submission_classify.txt

The evaluation script calculates the mean average precision (mAP) for the videos. That is, the average of the average precision (AP) for a single activity in all the videos.


###########################################################
Charades_v1_localize.m
###########################################################
Evaluation code for frame-level classification (localization). Each frame in a video has zero or more actions. This script takes in a "submission file" which is a csv file of the form:

id framenumber vector

where 'id' is a video id for a given video, 'framenumber' is the number of frame described below, and 'vector' is a whitespace delimited list of 157 floating point numbers representing the scores of each action in a frame. An example submission file is provided in test_submission_localize.txt (download this file with get_test_submission_localize.sh).

To avoid extremely large submission files, the evaluation script evaluates mAP on 25 equally spaced frames throughout each video. The frames are chosen as follows

for j=1:frames_per_video
    timepoint(j) = (j-1)*time/frames_per_video;

That is: 0, time/25, 2*time/25, ..., 24*time/25.

The baseline performance was generated by calculating the action scores at 75 equally spaced frames in the video (our batchsize) and picking every third prediction.

For more information about localization, please refer to the following publication:
@article{sigurdsson2016asynchronous,
author = {Gunnar A. Sigurdsson and Santosh Divvala and Ali Farhadi and Abhinav Gupta},
title = {Asynchronous Temporal Fields for Action Recognition},
journal={arXiv preprint arXiv:1612.06371},
year={2016},
pdf = {http://arxiv.org/pdf/1612.06371.pdf},
code = {https://github.com/gsig/temporal-fields},
}


###########################################################
Charades_v1_rgb.tar
###########################################################
These frames were extracted at 24fps using the following ffmpeg call for each video in the dataset:

line=pathToVideo
MAXW=320
MAXH=320
filename=$(basename $line)
ffmpeg -i "$line" -qscale:v 3 -filter:v "scale='if(gt(a,$MAXW/$MAXH),$MAXW,-1)':'if(gt(a,$MAXW/$MAXH),-1,$MAXH)',fps=fps=24" "/somepath/${filename%.*}/${filename%.*}_%0d.jpg";

The files are stored as Charades_v1_rgb/id/id-000000.jpg where id is the video id and 000000 is the number of the frame at 24fps.


###########################################################
Charades_v1_flow.tar
###########################################################
The flow was calculated similarly at 24fps using the OpenCV "Dual TV L1" Optical Flow Algorithm (OpticalFlowDual_TVL1_GPU)
The flow for each frame is stored as id-000000x.jpg and id-000000y.jpg for the x and y components of the flow respectively. 
The flow is mapped to the range {0,1,...,255} with the following formula:
y = 255*(x-L)/(H-L)
y = max(0,y)
y = min(255,y)
where L=-20 and H=20, the lower and high bounds of the optical flow.
The files are stored as Charades_v1_flow/id/id-000000x.jpg where id is the video id, 000000 is the number of the frame at 24fps, and x is either x or y depending on the optical flow direction (optical flow is stored as two seperate grayscale images for the two channels)


###########################################################
Charades_v1_features_{rgb,flow}.tar.gz
###########################################################
Using the two-stream code available at github.com/gsig/charades-algorithms we extracted fc7 features (after ReLU) from VGG-16 rgb and optical flow streams using the provided models (twostream_rgb.t7 and twostream_flow.t7). Logistic regression (linear layer, softmax, and cross entropy loss) on top of these features gives 18.9% accuracy on Charades classification. Simplified code for extracting the features is as follows:

fc7 = model.modules[37].output
for i=1,fc7:size(1) do
   out.write(string.format('%.6g',fc7[i]))

There are two folders Charades_v1_features_rgb/ and Charades_v1_features_flow/ for the two streams.
The features are stored in a whitespace delimited textfile for the 4096 numbers. 

The files are stored as Charades_v1_features/rgb/id/id-000000.txt where id is the video is and 000000 is the number of the frame at 24fps, which matches the provided rgb and flow data. To limit the file size, we include every 4th frame, but the frame numbers correspond to 24fps, so the numbers are 1,5,9,13, etc.
The features can be loaded as follows:

Python: 
fc7 = numpy.loadtxt('id/id-000000.txt')

Torch: 
file = io.open('id/id-000000.txt')
xx = torch.Tensor(file:lines()():split(' '));
file:close()


###########################################################
Baseline algorithms on Charades 
###########################################################
Code for multiple activity recognition algorithms are provided at:
https://github.com/gsig/charades-algorithms


###########################################################
CHANGELOG
###########################################################
6/1/16
Initial release

2/27/17
Adding support for evaluating localization. New evaluation script for localization. 
'length' column was added to Charades_v1_train.csv and Charades_v1_test.csv to have an official length of each video.
Adding details about provided RGB and Flow data.

5/14/17
Adding details about provided fc7 features.
Improving scene annotations ('scene' column in _train and _test).


###########################################################
