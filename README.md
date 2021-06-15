
# DCASE 2021: Sound Event Localization and Detection with Directional Interference

[Please visit the official webpage of the DCASE 2021 Challenge for details missing in this repo](http://dcase.community/challenge2021/task-sound-event-localization-and-detection).
   
As the baseline method for the [SELD task](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-localization-detection-and-tracking), we use the SELDnet method studied in the following papers, with  Activity-Coupled Cartesian Direction of Arrival (ACCDOA) representation as the output format. If you are using this baseline method or the datasets in any format, then please consider citing the following two papers. If you want to read more about [generic approaches to SELD then check here](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-localization-detection-and-tracking).

1.  [Archontis Politis, Sharath Adavanne, Daniel Krause, Antoine Deleforge, Prerak Srivastava, and Tuomas Virtanen. "A dataset of dynamic reverberant sound scenes with directional interferers for sound event localization and detection", submitted to DCASE 2021](https://arxiv.org/pdf/2106.06999.pdf)

2.  [Sharath Adavanne, Archontis Politis, Joonas Nikunen and Tuomas Virtanen, "Sound event localization and detection of overlapping sources using convolutional recurrent neural network" in IEEE Journal of Selected Topics in Signal Processing (JSTSP 2018)](https://arxiv.org/pdf/1807.00129.pdf)

2. [Kazuki Shimada, Yuichiro Koyama, Naoya Takahashi, Shusuke Takahashi, and Yuki Mitsufuji, "ACCDOA: Activity-Coupled Cartesian Direction of Arrival Representation for Sound Event Localization and Detection" in the The international Conference on Acoustics, Speech, & Signal Processing (ICASSP 2021)](https://arxiv.org/pdf/2010.15306.pdf)

NOTE: The baseline only supports detection of one instance of a sound class in a given time frame. However, the training data can consist of multiple instances of the same sound class in a given time frame. If participants are planning to build an SELD system that can detect such multiple instances of the same class, then you will have to modify this code accordingly. On the other hand, the provided metric code - `SELD_evaluation_metrics.py` and the class wrapper for it `cls_compute_seld_results.py` both support multiple instances of the same class. So you can continue to use them for your multi-instance SELD system.

## BASELINE METHOD

In comparison to the SELDnet studied in [1], we have changed the output format to ACCDOA [2] to improve its performance.
 * **ACCDOA output format**: The original SELDnet employed multi-task output, we simplify the architecture by employing the ACCDOA format, which encodes both SED and DOA information in one single output.

The final SELDnet architecture is as shown below. The input is the multichannel audio, from which the different acoustic features are extracted based on the input format of the audio. Based on the chosen dataset (FOA or MIC), the baseline method takes a sequence of consecutive feature-frames and predicts all the active sound event classes for each of the input frame along with their respective spatial location, producing the temporal activity and DOA trajectory for each sound event class. In particular, a convolutional recurrent neural network (CRNN) is used to map the frame sequence to a single ACCDOA sequence output which encodes both [sound event detection (SED)](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-detection) and [direction of arrival (DOA)](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-localization-and-tracking) estimates in the continuous 3D space as a multi-output regression task. Each sound event class in the ACCDOA output is represented by three regressors that estimate the Cartesian coordinates x, y and z axes of the DOA around the microphone. If the vector length represented by x, y and z coordinates are greater than 0.5, the sound event is considered to be active, and the corresponding x, y, and z values are considered as its predicted DOA.

<p align="center">
   <img src="https://github.com/sharathadavanne/seld-dcase2021/blob/master/images/CRNN_SELD_DCASE2021.png" width="400" title="SELDnet+ACCDOA Architecture">
</p>


The figure below visualizes the SELDnet input and outputs for one of the recordings in the dataset. The horizontal-axis of all sub-plots for a given dataset represents the same time frames, the vertical-axis for spectrogram sub-plot represents the frequency bins, vertical-axis for SED reference and prediction sub-plots represents the unique sound event class identifier, and for the DOA reference and prediction sub-plots, it represents the distances along the Cartesian axes. The figures represents each sound event class and its associated DOA outputs with a unique color. Similar plot can be visualized on your results using the [provided script](visualize_SELD_output.py).

<p align="center">
   <img src="https://github.com/sharathadavanne/seld-dcase2021/blob/master/images/SELDnet_output.jpg" width="300" title="SELDnet input and output visualization">
</p>

## DATASETS

The participants can choose either of the two or both the following datasets,

 * **TAU-NIGENS Spatial Sound Events 2021 - Ambisonic**
 * **TAU-NIGENS Spatial Sound Events 2021 - Microphone Array**

These datasets contain recordings from an identical scene, with **TAU-NIGENS Spatial Sound Events 2021 - Ambisonic** providing four-channel First-Order Ambisonic (FOA) recordings while  **TAU-NIGENS Spatial Sound Events 2021 - Microphone Array** provides four-channel directional microphone recordings from a tetrahedral array configuration. Both formats are extracted from the same microphone array, and additional information on the spatial characteristics of each format can be found below. The participants can choose one of the two, or both the datasets based on the audio format they prefer. Both the datasets, consists of a development and evaluation set. The development set consists of 600, one minute long recordings sampled at 24000 Hz. All participants are expected to use the fixed splits provided in the baseline method for reporting the development scores. We use 400 recordings for training split (fold 1 to 4), 100 for validation (fold 5) and 100 for testing (fold 6). The evaluation set consists of 200, one-minute recordings, and will be released at a later point.

More details on the recording procedure and dataset can be read on the [DCASE 2021 task webpage](http://dcase.community/challenge2021/task-sound-event-localization-and-detection).

The two development datasets can be downloaded from the link - [**TAU-NIGENS Spatial Sound Events 2021 - Ambisonic and Microphone Array**](https://doi.org/10.5281/zenodo.4568781) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4568781.svg)](https://doi.org/10.5281/zenodo.4568781)


## Getting Started

This repository consists of multiple Python scripts forming one big architecture used to train the SELDnet.
* The `batch_feature_extraction.py` is a standalone wrapper script, that extracts the features, labels, and normalizes the training and test split features for a given dataset. Make sure you update the location of the downloaded datasets before.
* The `parameter.py` script consists of all the training, model, and feature parameters. If a user has to change some parameters, they have to create a sub-task with unique id here. Check code for examples.
* The `cls_feature_class.py` script has routines for labels creation, features extraction and normalization.
* The `cls_data_generator.py` script provides feature + label data in generator mode for training.
* The `keras_model.py` script implements the SELDnet architecture.
* The `SELD_evaluation_metrics.py` script implements the metrics for joint evaluation of detection and localization.
* The `seld.py` is a wrapper script that trains the SELDnet. The training stops when the SELD error (check paper) stops improving.
* The `cls_compute_seld_results.py` script computes the metrics results on your DCASE output format files. You can switch between using polar or Cartesian based scoring. Ideally both should give identical results.

Additionally, we also provide supporting scripts that help analyse the results.
 * `visualize_SELD_output.py` script to visualize the SELDnet output


### Prerequisites

The provided codebase has been tested on python 3.6.9/3.7.3 and Keras 2.2.4/2.3.1


### Training the SELDnet

In order to quickly train SELDnet follow the steps below.

* For the chosen dataset (Ambisonic or Microphone), download the respective zip file. This contains both the audio files and the respective metadata. Unzip the files under the same 'base_folder/', ie, if you are Ambisonic dataset, then the 'base_folder/' should have two folders - 'foa_dev/' and 'metadata_dev/' after unzipping.

* Now update the respective dataset name and its path in `parameter.py` script. For the above example, you will change `dataset='foa'` and `dataset_dir='base_folder/'`. Also provide a directory path `feat_label_dir` in the same `parameter.py` script where all the features and labels will be dumped.

* Extract features from the downloaded dataset by running the `batch_feature_extraction.py` script. Run the script as shown below. This will dump the normalized features and labels in the `feat_label_dir` folder.

```
python3 batch_feature_extraction.py
```

You can now train the SELDnet using default parameters using
```
python3 seld.py
```

* Additionally, you can add/change parameters by using a unique identifier \<task-id\> in if-else loop as seen in the `parameter.py` script and call them as following
```
python3 seld.py <task-id> <job-id>
```
Where \<job-id\> is a unique identifier which is used for output filenames (models, training plots). You can use any number or string for this.

In order to get baseline results on the development set for Microphone array recordings, you can run the following command
```
python3 seld.py 2
```
Similarly, for Ambisonic format baseline results, run the following command
```
python3 seld.py 4
```

* By default, the code runs in `quick_test = True` mode. This trains the network for 2 epochs on only 2 mini-batches. Once you get to run the code sucessfully, set `quick_test = False` in `parameter.py` script and train on the entire data.

* The code also plots training curves, intermediate results and saves models in the `model_dir` path provided by the user in `parameter.py` file.

* In order to visualize the output of SELDnet and for submission of results, set `dcase_output=True` and provide `dcase_dir` directory. This will dump file-wise results in the directory, which can be individually visualized using `visualize_SELD_output.py` script.

## Results on development dataset

As the [SELD evaluation metric](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-localization-detection-and-tracking#h.ragsbsp7ujs) we employ the joint localization and detection metrics proposed in [1], with extensions from [2] to support multi-instance scoring of the same class.

1. [Annamaria Mesaros, Sharath Adavanne, Archontis Politis, Toni Heittola, and Tuomas Virtanen, "Joint Measurement of Localization and Detection of Sound Events", IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA 2019)](https://tutcris.tut.fi/portal/files/21452683/mesaros_Joint_localization_and_detection_WASPAA2019.pdf)

2. [Archontis Politis, Annamaria Mesaros, Sharath Adavanne, Toni Heittola, and Tuomas Virtanen, "Overview and Evaluation of Sound Event Localization and Detection in DCASE 2019", IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP 2020)](https://arxiv.org/pdf/2009.02792.pdf)

There are in total four metrics that we employ in this challenge.
The first two metrics are more focused on the detection part, also referred as the location-aware detection, corresponding to the error rate (ER<sub>20°</sub>) and F-score (F<sub>20°</sub>) in one-second non-overlapping segments. We consider the prediction to be correct if the prediction and reference class are the same, and the distance between them is below 20&deg;.
The next two metrics are more focused on the localization part, also referred as the class-aware localization, corresponding to the localization error (LE<sub>CD</sub>) in degrees, and a localization Recall (LR<sub>CD</sub>) in one-second non-overlapping segments, where the subscript refers to _classification-dependent_. Unlike the location-aware detection, we do not use any distance threshold, but estimate the distance between the correct prediction and reference.

The evaluation metric scores for the test split of the development dataset is given below

| Dataset | ER<sub>20°</sub> | F<sub>20°</sub> | LE<sub>CD</sub> | LR<sub>CD</sub> |
| ----| --- | --- | --- | --- |
| Ambisonic (FOA) | 0.69 | 33.9 % | 24.1&deg; | 43.9 % |
| Microphone Array (MIC) | 0.74 | 24.7 % | 30.9&deg; | 38.2 % |

**Note:** The reported baseline system performance is not exactly reproducible due to varying setups. However, you should be able to obtain very similar results.

## Submission

* Before submission, make sure your SELD results look good by visualizing the results using `visualize_SELD_output.py` script
* Make sure the file-wise output you are submitting is produced at 100 ms hop length. At this hop length a 60 s audio file has 600 frames.

For more information on the submission file formats [check the website](http://dcase.community/challenge2021/task-sound-event-localization-and-detection)

## License

Except for the contents in the `metrics` folder that have [MIT License](metrics/LICENSE.md). The rest of the repository is licensed under the [TAU License](LICENSE.md).
