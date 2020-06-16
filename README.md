# On Joint Estimation of Pose, Geometry and svBRDF from a Handheld Scanner


#### [Project Page](https://avg.is.tuebingen.mpg.de/publications/schmitt2020cvpr) | [Paper](http://www.cvlibs.net/publications/Schmitt2020CVPR.pdf) | [Spotlight Video](https://www.youtube.com/watch?v=_xxSQPD9qU0) | [Presentation](http://www.cvlibs.net/publications/Schmitt2020CVPR_slides.pdf) | [Poster](http://www.cvlibs.net/publications/Schmitt2020CVPR_poster.pdf)

![teaser](teaser.png)

This is the source code repository for our CVPR publication [On Joint Estimation of Pose, Geometry and svBRDF from a Handheld Scanner](http://www.cvlibs.net/publications/Schmitt2020CVPR.pdf).

By Carolin Schmitt, Simon Donné, Gernot Riegler, Vladlen Koltun and Andreas Geiger.


If you find our code or paper useful, please consider citing

    @inproceedings{Schmitt2020CVPR,
      title = {On Joint Estimation of Pose, Geometry and svBRDF from a Handheld Scanner},
      author = {Schmitt, Carolin and Donne, Simon and Riegler, Gernot and Koltun, Vladlen and Geiger, Andreas},
      booktitle = { Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
      year = {2020}
    }


## Installation

After cloning this repository, you will need to make sure that all dependencies are in place.
The easiest way to do so is to use our installation script that sets up a conda environment and installs all required packages for you.

To use it first make sure that you have a working installation of conda: <https://conda.io/>.

Next, please check the CUDA version of your system with

    which nvcc

By default, the script compiles Pytorch 1.4 for CUDA 10.0.
But this can easily be changed in 'installEnv.sh':
In line 69 change the 'cudatoolkit=10.0' to the CUDA version of your system.

Then run the script via

    ./installEnv.sh

It takes care of all of the installation requirements.


## Sensor data

We provide two options to download our sensor data: You can either download the data for only one object or the data of all 9 objects that are shown in the publication.
At the moment, the data of only one object is uploaded but the rest will be added soon!
Download the pre-processed captures via

    mkdir data
    cd data/
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/handheld_svbrdf_geometry/data_single.zip
    unzip data_single.zip



## Run the method

After downloading and extracting the data locally, you need to adapt the 'path_localization' dictionary in 'code/general_settings.py'.
- input_data_base_folder: The path of the 'captures' folder.
- output_base_folder: Where to save the output files.
- calibration_base_folder: The path of the 'calibration' folder.
- gt_scan_folder: The path of the 'gt_scans' folder.

At the end of 'main.py' you can specify for which object and which center_view you want to run the optimization and evaluation.

Make sure that the conda environment is activated and you are in the correct folder:

    conda activate handheld_svbrdf_geometry
    cd code/

Then run

    python main.py


### Evaluation

Prior to releasing this code, we refactored the full repository.
We observe that the quantitative results we are getting now are slightly different than the results reported in the paper.
Since this occurs consistently over all baselines, most likely only the evaluation code changed slightly.
We are working on resolving it but note that the results still support all our findings in the publication.

