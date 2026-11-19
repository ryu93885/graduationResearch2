**GRADUATION RESEARCH**

**PROGRESS REPORT**

Revised Research Structure, Decisions to Date, and Items Under Discussion

**Comparing the Decision Evidence of  
Pneumonia Classification and Detection Models  
with Physician Annotations in Chest Radiographs**

*— A Rule-Based Baseline, a ResNet Classifier with Grad-CAM,  
and a Faster R-CNN Detector —*

(Tentative title — TBD)

**Prepared: 19 November 2026**

Status: Research planning stage — research structure revised; implementation and experiments not yet started

*This report summarizes the current research plan and progress. It is not a final thesis and does not present experimental results.*

# Document Information

<table>
<colgroup>
<col style="width: 37%" />
<col style="width: 62%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>Item</strong></th>
<th><strong>Description</strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Document type</td>
<td>Graduation research progress report (research-planning stage, second report)</td>
</tr>
<tr class="even">
<td>Date prepared</td>
<td>19 November 2026</td>
</tr>
<tr class="odd">
<td>Previous report</td>
<td>5 November 2026 (research concept, dataset selection, and experimental plan)</td>
</tr>
<tr class="even">
<td>Primary dataset</td>
<td>RSNA Pneumonia Detection Challenge 2018</td>
</tr>
<tr class="odd">
<td>Current status</td>
<td>Research structure revised after the supervisor's comments; data acquisition and implementation not yet started</td>
</tr>
<tr class="even">
<td>Status labels used in this report</td>
<td><p><strong>Decided</strong>: agreed and fixed.</p>
<p><strong>Under discussion</strong>: being discussed with the academic supervisor.</p>
<p><strong>TBD</strong>: to be decided; not yet discussed or not yet answered.</p></td>
</tr>
</tbody>
</table>

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p><strong>Central research question</strong></p>
<p>Does the evidence on which deep-learning models base their predictions of pneumonia-like opacities in chest radiographs agree with the regions annotated by physicians? Where it does not agree, can the difference be explained by shortcut learning or missing annotations, or does it point to plausible evidence that physicians did not mark?</p></td>
</tr>
</tbody>
</table>

# Abstract

This study examines whether the evidence used by deep-learning models to identify opacities suggestive of pneumonia in chest radiographs agrees with the regions annotated by physicians. Using the RSNA Pneumonia Detection Challenge 2018 dataset, three methods will be developed: a simple rule-based baseline that combines image features such as the amount and density of bright (white) pixels; an image classifier (Model A, ResNet) that assigns one of three labels (Lung Opacity, No Lung Opacity / Not Normal, and Normal); and an object detector (Model B, Faster R-CNN) that outputs Bounding Boxes for Lung Opacity.

The three methods will be compared on the same test set as a binary task (Lung Opacity or not) using accuracy and confusion matrices. The decision evidence of Model A will be visualized with Grad-CAM and compared with the physicians' Bounding Boxes using localization accuracy and the pointing game, following the original Grad-CAM paper. For Model B, the detected boxes themselves will be treated as its evidence and compared directly with the physicians' boxes. Evidence that differs from the physicians' annotations will first be examined for shortcut learning and missing annotations before any interpretation is made.

Since the previous report (5 November 2026), the research structure has been revised: the masking-based comparison of three input regions has been replaced by the classifier/detector design described above, and the supervisor's comments have been reflected. The backbone network, training procedure, and model-selection metric for Model A, and the choice of detector and pre-trained weights for Model B, are under discussion with the supervisor. Data acquisition, implementation, and experiments have not yet begun.

# Contents

| **Section** | **Title**                                            |
|-------------|------------------------------------------------------|
| Chapter 1   | Research Background, Purpose, and Research Questions |
| Chapter 2   | Dataset and Conditions of Use                        |
| Chapter 3   | Research Method and Model Design                     |
| Chapter 4   | Evaluation Methods and Verification Plan             |
| Chapter 5   | Experimental Results and Research Progress           |
| Chapter 6   | Discussion, Limitations, and Mitigation Strategies   |
| Chapter 7   | Conclusion and Future Research Plan                  |
| References  | Official resources and related literature            |

# Chapter 1. Research Background, Purpose, and Research Questions

## 1.1 Research Background

AI models are increasingly used in medicine. For common cases they have been reported to perform as well as or better than specialists, although they still fall short for rare cases. However, the basis of a deep-learning model's decision is not directly visible, and its explainability remains under debate. This is one reason why the clinical adoption of such models is still limited. Classification accuracy alone does not show which image regions or features a model used, and a model may rely on factors that are not related to the lesion, such as imaging conditions or text markers.

In this study, a region marked by physicians as an opacity suspicious for pneumonia is referred to as a “physician-annotated region.” The Bounding Boxes in the RSNA dataset are rectangles surrounding pulmonary opacities; they are not precise pixel-level lesion contours. Agreement or disagreement with these regions is therefore interpreted as agreement with the physicians' annotations, not with the complete diagnostic reasoning of physicians.

## 1.2 Research Purpose

The purpose of this study is to visualize the decision evidence of models that identify pneumonia-like opacities (Lung Opacity) in chest radiographs and to examine whether this evidence agrees with physician-annotated regions, as a way of assessing the reliability of the models' decisions. For the classifier, Grad-CAM, an Explainable AI (XAI) method, is used. For the detector, the predicted Bounding Boxes already show where the model is looking and are used directly as its evidence. In addition, where a model's evidence differs from the physicians' annotations, the study examines whether that evidence is plausible when compared across other images, and whether it could offer a new viewpoint for image-based detection.

## 1.3 Research Questions (Draft)

- RQ1: Are deep-learning models (classification and detection) more useful than a simple rule-based method for identifying images with pneumonia-like opacities?

- RQ2: Does the decision evidence of the models (Grad-CAM for the classifier and the predicted Bounding Boxes for the detector) agree with physician-annotated regions?

- RQ3: Where the evidence does not agree with the physicians' annotations, can it be regarded as plausible evidence when compared with other images, after shortcut learning and missing annotations have been ruled out?

## 1.4 Hypotheses and Decision Criteria

Formal hypotheses and the criteria for judging them are **TBD**. They will be written down before the test set is used. The evaluation metrics on which they will be based have been decided (Chapter 4).

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p><strong>Important interpretation note</strong></p>
<p>A model response outside the physician-annotated regions is not, on its own, sufficient to conclude that a new, medically useful finding has been discovered. The model may be using imaging conditions, text markers, devices, or other dataset-specific biases (shortcut learning), or the region may contain an opacity that was not annotated. These possibilities will be examined first (Section 4.6).</p></td>
</tr>
</tbody>
</table>

# Chapter 2. Dataset and Conditions of Use

## 2.1 Selected Dataset

The dataset is the chest-radiograph collection and annotations released for the RSNA Pneumonia Detection Challenge 2018. RSNA worked with the Society of Thoracic Radiology (STR), MD.ai, and other collaborators to prepare a dataset based on NIH ChestX-ray data for detecting and localizing opacities suspicious for pneumonia \[1\]\[4\].

| **Item**            | **Description**                                                                                                                                 |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| Image modality      | Chest radiographs (frontal views)                                                                                                               |
| Number of images    | Approximately 30,000 images                                                                                                                     |
| Source data         | NIH ChestX-ray8 (a public dataset containing approximately 112,000 images) \[18\]                                                               |
| Image format        | DICOM, with tags such as sex, age, and view position \[3\]                                                                                      |
| Labels              | Three classes: Lung Opacity, No Lung Opacity / Not Normal, and Normal. Only the final labels are used in this study.                            |
| Annotations         | Bounding Boxes indicating pulmonary opacities, given for Lung Opacity images                                                                    |
| Patient information | 12,274 individuals for 30,000 examinations; some patients have multiple examinations \[4\]                                                      |
| Reading process     | 18 radiologists from 16 institutions participated. Not all cases were read by three readers; 4,527 cases were read by three readers \[3\]\[4\]. |

## 2.2 Labels and Target Tasks

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p><strong>Target tasks (Decided)</strong></p>
<p><strong>Model A (classifier):</strong> three-class classification using the final labels.</p>
<p><strong>Model B (detector):</strong> detection of Lung Opacity Bounding Boxes.</p>
<p><strong>Rule-based baseline:</strong> binary classification (Lung Opacity or not). An extension to three classes may be considered if the binary version works.</p>
<p><strong>Common comparison of the three methods:</strong> binary task (Lung Opacity or not) on the same test set.</p></td>
</tr>
</tbody>
</table>

A preliminary data inspection was carried out in an earlier project that used the same dataset. Its main findings are listed below because they affect the design of this study. They will be re-checked after the data have been acquired for this project.

| **Finding (preliminary)** | **Value**                                                                                                                                    | **Relevance to this study**                                                           |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| Images with a final label | 29,684 (316 images have no final label)                                                                                                      | Images without a final label cannot be used.                                          |
| Class proportions         | Lung Opacity 7,106 (23.9%); No Lung Opacity / Not Normal 12,788 (43.1%); Normal 9,790 (33.0%)                                                | The classes are not strongly imbalanced, so accuracy remains informative.             |
| View position             | 77.3% of Lung Opacity images are AP views, compared with 45.7% of all images. View position alone gives a ROC-AUC of 0.708 for Lung Opacity. | A strong possible shortcut; evaluation will be stratified by AP and PA (Section 4.6). |

## 2.3 Conditions of Use

Under the conditions stated on the official RSNA distribution page, the de-identified images and annotations may be used for academic research and education. Users must not attempt to re-identify or contact individuals represented in the data, and sharing or redistribution requires the specified attribution: the RSNA official distribution page and Shih et al. (2019) for the RSNA data, and the NIH download page, Wang et al. (CVPR 2017), and the NIH Clinical Center for the underlying NIH data \[1\]\[2\]. The use of images in the thesis or presentation materials will be confirmed with the academic supervisor.

## 2.4 Data Acquisition

The data will be acquired from the official RSNA distribution page using Python, as planned in the previous report (requests for downloading, zipfile for extraction, pydicom for DICOM images, and pandas/json for labels and Bounding Boxes). Data acquisition for this project has **not started**. Whether the acquisition programs written in the earlier project will be reused is **TBD**.

# Chapter 3. Research Method and Model Design

## 3.1 Overall Research Workflow

The same patient-level data split is used for all three methods. All tuning uses the validation set, and the test set is used only once, after the models and thresholds have been fixed.

| **Stage 1**                    | **Stage 2**                             | **Stage 3**                   | **Stage 4**               | **Stage 5**                               | **Stage 6**               |
|--------------------------------|-----------------------------------------|-------------------------------|---------------------------|-------------------------------------------|---------------------------|
| Acquire RSNA images and labels | Split data by patient (70% / 10% / 20%) | Build the rule-based baseline | Train Model A and Model B | Compare the three methods on the test set | Analyze decision evidence |

## 3.2 Methods for Comparison

In this report, Model A and Model B refer to the classifier and the detector. They are different from Models A, B, and C in the previous report, which differed only in the input region.

| **Method**          | **Input, output, and approach**                                                                                                                                           | **Role in the study**                                                           | **Status**                                                      |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|-----------------------------------------------------------------|
| Rule-based baseline | Combines a few hand-crafted image features (e.g., the amount and density of bright pixels) with thresholds set by hand on the training data. Outputs Lung Opacity or not. | Reference point for judging the usefulness of deep learning (RQ1).              | Basic design decided; some items TBD (Section 3.3)              |
| Model A: classifier | ResNet pre-trained on ImageNet. Input: whole image (512×512). Output: probabilities of three classes.                                                                     | Classification (RQ1). Decision evidence visualized with Grad-CAM (RQ2, RQ3).    | Network and training procedure under discussion (Section 3.4)   |
| Model B: detector   | Faster R-CNN with a ResNet-FPN backbone. Input: whole image (512×512). Output: Bounding Boxes and confidence scores for Lung Opacity.                                     | Detection (RQ1). Predicted boxes used directly as decision evidence (RQ2, RQ3). | Detector and pre-trained weights under discussion (Section 3.5) |

## 3.3 Rule-Based Baseline

In response to the supervisor's question, the rule-based baseline is defined as a simple model that combines a small number of image features, such as the amount of bright (white) pixels, which is characteristic of pneumonia images, and the density of those pixels. Its role is limited to providing a reference point for the deep-learning models; it is not intended to compete with them. Taking into account the supervisor's comment that classical computer-vision algorithms may require much effort for little result, its scope has been narrowed to binary classification, and Bounding Box detection will not be attempted.

| **Item**                     | **Setting**                                                                                                                                              | **Status**                                                    |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| Task                         | Binary classification (Lung Opacity or not); no Bounding Box detection                                                                                   | Decided                                                       |
| Extension                    | Three-class classification, if the binary version works                                                                                                  | Possible extension                                            |
| Features                     | A few features such as the amount and density of bright pixels, chosen after analyzing the training data                                                 | Approach decided; exact features chosen during implementation |
| Rule design                  | Thresholds and combinations set by hand from the feature distributions of the training set; checked on the validation set; test set used once            | Decided                                                       |
| Region used for the features | Whole image (after normalizing brightness per image), a fixed central region, or the lung fields (which would require a separate lung-segmentation step) | TBD                                                           |
| Time limit                   | Upper limit on the time spent on the baseline                                                                                                            | TBD                                                           |

Bright pixels also arise from bones, the heart, tubes, text markers, and differences in exposure between AP and PA views. The region used for the features therefore matters, and it will be decided before implementation.

## 3.4 Model A: Image Classifier

The settings below are those of the current research structure. The supervisor has suggested starting with ResNet-50, training for a fixed number of epochs with preset patience, and comparing one standard setting with a couple of alternatives. These points are **under discussion**.

| **Item**                     | **Setting (current structure)**                                                                                                                                                                                                        | **Status**                                                                                                   |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| Network                      | ResNet-101 (torchvision); the final layer fc (2048→1000) is replaced by Linear (2048→3)                                                                                                                                                | Under discussion (ResNet-50 suggested by the supervisor: faster and often similar results)                   |
| Number of parameters         | 42,506,307 with ResNet-101; 23,514,179 with ResNet-50                                                                                                                                                                                  | Depends on the network                                                                                       |
| Pre-trained weights          | ImageNet (IMAGENET1K_V2); all layers trained                                                                                                                                                                                           | Decided                                                                                                      |
| Input and preprocessing      | 512×512 (1024×1024 downscaled by half); grayscale replicated to three channels                                                                                                                                                         | Decided                                                                                                      |
| Labels and output            | Final labels only; probabilities of three classes (softmax)                                                                                                                                                                            | Decided                                                                                                      |
| Loss function                | Cross-entropy                                                                                                                                                                                                                          | Decided; class weighting under discussion (the supervisor advises changing the loss only to favor one class) |
| Optimizer                    | SGD (momentum 0.9, weight decay 1e-4)                                                                                                                                                                                                  | Current plan; may change with the training procedure                                                         |
| Learning rate and batch size | Learning rate chosen from 0.001–0.01 on the validation set; batch size 16–32                                                                                                                                                           | Current plan; under discussion as part of the standard setting                                               |
| Training procedure           | Supervisor's proposal: fixed maximum number of epochs (up to 50 initially; about 10 may be enough) with early-stopping patience of 5 and 10; one standard setting and a couple of alternative settings, selected on the validation set | Under discussion                                                                                             |
| Model-selection metric       | Validation accuracy with error analysis by confusion matrix (supervisor's suggestion) or validation macro-averaged ROC-AUC (current structure)                                                                                         | Under discussion                                                                                             |
| Data augmentation            | Not specified. Avoiding horizontal flipping is proposed, because it moves the heart to the opposite side.                                                                                                                              | TBD                                                                                                          |

## 3.5 Model B: Object Detector

Faster R-CNN is planned because it shares the ResNet structure with Model A and is easy to evaluate. The supervisor has suggested starting with YOLO, which is easy to implement, or Mask R-CNN, for which code already exists. The choice is **under discussion**.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p><strong>Note on Mask R-CNN</strong></p>
<p>The RSNA dataset provides Bounding Boxes only and no lesion masks, so the mask branch of Mask R-CNN cannot be trained. Because torchvision's Mask R-CNN is Faster R-CNN with an added mask branch, the existing Mask R-CNN code could be used as Faster R-CNN by removing the mask branch. This option would satisfy both the use of existing code and the use of Faster R-CNN.</p></td>
</tr>
</tbody>
</table>

| **Item**                             | **Setting (current structure)**                                                                                                                                | **Status**                                                                              |
|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| Detector                             | Faster R-CNN with FPN (torchvision)                                                                                                                            | Under discussion (alternatives: YOLO; existing Mask R-CNN code without the mask branch) |
| Backbone                             | ResNet-101-FPN (ResNet-50-FPN if Model A changes)                                                                                                              | Under discussion                                                                        |
| Number of parameters                 | 60,239,065 with ResNet-101-FPN; 41,299,161 with ResNet-50-FPN (two classes, frozen batch normalization)                                                        | Depends on the backbone                                                                 |
| Output layers                        | Box classifier 1024→2 (background and Lung Opacity); box regressor 1024→8                                                                                      | Decided                                                                                 |
| Pre-trained weights                  | “Same as Model A”: whether this means the ImageNet weights or the trained Model A backbone is not yet fixed. FPN, RPN, and RoI Heads are trained from scratch. | Under discussion                                                                        |
| Frozen layers                        | First convolutional layer and layer1 frozen; batch normalization frozen                                                                                        | Current plan                                                                            |
| Optimizer, learning rate, batch size | SGD (momentum 0.9, weight decay 5e-4); 0.001 per image (0.004 for a batch of 4, 0.008 for 8); batch size 4–8                                                   | Current plan                                                                            |
| Training procedure                   | Same approach as Model A                                                                                                                                       | Under discussion                                                                        |

## 3.6 Preprocessing and Data Splitting

- Downscale the 1024×1024 images to 512×512, replicate the grayscale image to three channels, and scale the Bounding Boxes by the same factor. (Decided)

- Split the data once at the patient level into training (70%), validation (10%), and test (20%) sets, so that no patient appears in more than one set. The same split is used for all three methods. (Decided)

- Whether to stratify the split by class. (TBD)

- Use the validation set for all tuning and threshold selection; use the test set only once, after the models and thresholds have been fixed. (Decided)

# Chapter 4. Evaluation Methods and Verification Plan

## 4.1 Classification and Detection Performance

| **Target**            | **Metrics**                                                                                                                                                                                                         | **Status**                                                                                                                       |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Model A               | Accuracy; per-class precision, recall, and F1-score; macro-averaged F1-score; confusion matrix; per-class ROC-AUC                                                                                                   | List of metrics decided; which metric is primary is under discussion (accuracy and confusion matrix suggested by the supervisor) |
| Model B (box level)   | Precision, recall, F1-score, and mAP, counting a predicted box as correct when its IoU with a physician box is 0.5 or more                                                                                          | Decided                                                                                                                          |
| Model B (image level) | Accuracy and confusion matrix. An image is positive when it has at least one box with a confidence score at or above a threshold chosen on the validation set. ROC is not used, following the supervisor's comment. | Decided                                                                                                                          |

## 4.2 Common Comparison of the Three Methods

The three methods are compared on the same test set by converting their outputs to a binary decision (Lung Opacity or not) and reporting accuracy and the confusion matrix (true positives, false positives, false negatives, and true negatives). **(Decided)**

| **Method**          | **An image is counted as positive (Lung Opacity) when**                                                   |
|---------------------|-----------------------------------------------------------------------------------------------------------|
| Rule-based baseline | The rules classify it as positive.                                                                        |
| Model A             | The class with the highest predicted probability is Lung Opacity.                                         |
| Model B             | At least one predicted box has a confidence score at or above the threshold chosen on the validation set. |

## 4.3 Reference Performance Levels

Values are chosen from the referenced papers first and from general values second. The referenced papers \[5\]\[6\]\[7\] report results on ImageNet, PASCAL VOC, and COCO rather than chest radiographs, and the dataset paper \[4\] does not report model performance. The values below were therefore taken from general sources. If the supervisor's training procedure is adopted (Section 3.4), these values will be used only as reference levels for reading the results, not as targets to be reached by repeated retraining. **(Under discussion)**

| **Model** | **Reference value**                               | **Basis**                                                                                                                                                                    |
|-----------|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Model A   | Validation macro-averaged ROC-AUC of 0.80 or more | A common guideline regards an AUC of 0.8–0.9 as excellent discrimination \[11\]. No reported three-class macro-averaged AUC for this dataset was found.                      |
| Model B   | Validation mAP (IoU 0.5) of 0.40 or more          | Reported values on the RSNA dataset: 39.23% for an improved Faster R-CNN \[12\] and about 45–50% for other detectors \[13\]. Their evaluation conditions are to be verified. |

## 4.4 Decision Evidence of Model A (Grad-CAM)

Following the original Grad-CAM paper \[8\], the gradients of the pre-softmax score of the Lung Opacity class with respect to the feature maps of the last convolutional layer (layer4 of the ResNet) are used. With a 512×512 input, the map is 16×16; it is upsampled to 512×512 for comparison with the physicians' boxes. The evaluation uses the Lung Opacity images in the test set, because only these images have physician boxes. **(Decided)**

| **Type**      | **Metric**             | **Definition**                                                                                                                                                            | **Source**                    |
|---------------|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| Primary       | Localization accuracy  | Binarize the map at 15% of its maximum value, draw a box around the largest connected region, and count it as correct when its IoU with any physician box is 0.5 or more. | Original Grad-CAM paper \[8\] |
| Primary       | Pointing-game hit rate | Count a hit when the location of the maximum value of the map lies inside any physician box.                                                                              | \[8\]\[9\]                    |
| Supplementary | mIoU                   | Mean IoU between the binarized map and the area covered by the physician boxes; the binarization threshold maximizes mIoU on the validation set.                          | Saporta et al. \[10\]         |

- An image whose localization is correct (IoU of 0.5 or more) is regarded as one in which the decision evidence agrees with the physicians' annotation.

- The chance level of the pointing game (the fraction of the image covered by the physician boxes) is reported alongside the hit rate.

- Results are reported separately for correctly and incorrectly classified images and for single-reader and three-reader images, with 95% confidence intervals from patient-level bootstrapping.

- A sanity check that confirms the map changes when the model weights are randomized \[17\] is proposed. Whether to include it is TBD.

## 4.5 Decision Evidence of Model B (Predicted Boxes)

Following the supervisor's comment that object detectors already output boxes that show where they are looking, Grad-CAM is not applied to Model B. Its predicted boxes with a confidence score at or above the threshold are compared directly with the physicians' boxes in all test images and are divided into three categories. **(Decided)**

| **Category**                                        | **Definition**                                                                                       | **Treatment**                                  |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------|
| Agreement                                           | IoU of 0.5 or more with any physician box                                                            | Counted as correct in the box-level evaluation |
| Position mismatch                                   | Overlaps a physician box, but the IoU is below 0.5                                                   | Counted as a localization error                |
| Candidate for evidence that differs from physicians | Does not overlap any physician box (IoU = 0), including detections in images without physician boxes | Examined with the procedure in Section 4.6     |

Only boxes that do not overlap any physician box are treated as candidates, because a box that overlaps a physician box is likely to point to the same opacity with a shifted position.

## 4.6 Verification of Evidence That Differs from Physician Annotations

A strong Grad-CAM response outside the physician boxes (Model A) or a detection that does not overlap any physician box (Model B) is not immediately regarded as new evidence, because deep-learning models can use shortcuts \[14\]\[15\]\[16\] and some opacities may not have been annotated. The following three factors are examined in order, and responses explained by them are excluded. **(Decided)**

1.  **Imaging conditions:** Lung Opacity images are strongly biased toward portable (AP) views (Section 2.2). Performance and agreement metrics are calculated separately for AP and PA images to check whether the evidence changes with view position.

2.  **Incidental content:** Text markers, tubes and catheters, ECG electrodes, and similar objects are checked visually for Grad-CAM responses or detections. Images in which the maximum of the map or a box lies outside the lung fields are examined in particular detail.

3.  **Missing annotations:** Many images were read by a single radiologist, so a response outside the boxes may correspond to an opacity that was not annotated. The analysis is repeated on images read by three radiologists to see whether the same tendency appears.

Responses that cannot be explained by these factors and that appear repeatedly at the same anatomical location across many images are treated as candidates for plausible evidence. The final judgment on these candidates will be made with advice from the academic supervisor or specialists.

## 4.7 Relationship to the Research Questions

| **Research question**                                 | **Planned verification**                                                                                                                                                              |
|-------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RQ1: Usefulness of deep learning                      | Compare the accuracy and confusion matrices of the rule-based baseline, Model A, and Model B on the same test set (Section 4.2).                                                      |
| RQ2: Agreement of evidence with physician annotations | Model A: localization accuracy, pointing-game hit rate, and mIoU against the chance level (Section 4.4). Model B: proportion of boxes in each category (Section 4.5).                 |
| RQ3: Plausibility of differing evidence               | Apply the three-step check (Section 4.6) and review the remaining candidates with the supervisor or specialists. Visualization alone will not be used to claim a new medical finding. |

# Chapter 5. Experimental Results and Research Progress

As of 19 November 2026, data acquisition, model development, and training experiments have not yet begun. No numerical results or visualizations are available. This chapter records the current status of the experiments, the changes since the previous report, and the responses to the supervisor's comments.

## 5.1 Current Status of Experimental Results

All items are currently unmeasured; no values have been estimated or entered.

| **Target**          | **Experiment**                                       | **Current status**            |
|---------------------|------------------------------------------------------|-------------------------------|
| Rule-based baseline | Binary classification performance                    | Not conducted / not measured  |
| Model A             | Three-class classification performance               | Not conducted / not measured  |
| Model B             | Detection performance (box level and image level)    | Not conducted / not measured  |
| Common comparison   | Accuracy and confusion matrices of the three methods | Not conducted / not measured  |
| Decision evidence   | Grad-CAM of Model A and box categories of Model B    | Not conducted / not evaluated |

## 5.2 Changes Since the Previous Report

| **Item**            | **Previous report (5 November 2026)**                                                                   | **Current plan**                                                                                  |
|---------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Central question    | Whether information outside physician-annotated regions can be used for classification                  | Whether the models' decision evidence agrees with physician-annotated regions                     |
| Methods             | Three classifiers with different input regions: A (region-focused), B (full image), C (region-excluded) | Rule-based baseline, Model A (classifier), and Model B (detector)                                 |
| Masking experiments | Core of the design                                                                                      | Not included                                                                                      |
| XAI                 | Grad-CAM mainly on the full-image model; faithfulness and stability evaluated                           | Grad-CAM on the classifier only; the detector's boxes are used directly; metrics fixed in advance |
| Evaluation          | Accuracy, precision, recall, specificity, F1-score, ROC-AUC / PR-AUC                                    | Accuracy and confusion matrices for the common comparison; ROC not used for the detector          |
| Data split          | Patient-level split (method not fixed)                                                                  | Single patient-level split: 70% / 10% / 20%                                                       |

## 5.3 Responses to the Supervisor's Comments

| **Comment (summary)**                                                                                                                                                                                            | **Response**                                                                                                                                                                                                            | **Status**                                     |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| 1\. XAI: much questionable work is done under this label, but the student may pursue it.                                                                                                                         | Continue, because the topic is attractive precisely because it has not been studied sufficiently. Evaluation metrics are fixed in advance, results are compared with the chance level, and shortcuts are checked first. | Decided                                        |
| 2\. Rule-based system: meaning unclear; classical computer vision may take much effort for little result.                                                                                                        | A simple model combining a few image features (e.g., amount and density of bright pixels). Scope narrowed to binary classification; thresholds set by hand on the training data.                                        | Scope decided; region and time limit TBD       |
| 3\. Classifier: start with ResNet-50; evaluate by accuracy and confusion matrix; fixed epochs with patience 5 and 10; a standard setting plus a couple of alternatives; change the loss only to favor one class. | To be decided at the meeting with the supervisor.                                                                                                                                                                       | Under discussion                               |
| 4\. Detector: start with YOLO or the existing Mask R-CNN code; ROC is unlikely to help.                                                                                                                          | Faster R-CNN is planned (same ResNet structure as Model A, easy to evaluate); the existing Mask R-CNN code could be used without its mask branch. ROC has been removed from the detector evaluation.                    | Detector under discussion; ROC removal decided |
| 5\. Grad-CAM is mainly for classification; detectors already show where they look; drop it initially.                                                                                                            | Adopted: Grad-CAM is used for the classifier only, and the detector's boxes are compared directly with the physicians' boxes.                                                                                           | Decided                                        |

## 5.4 Progress Summary

| **Item**                         | **Current status**                                                                             | **Progress**                                      |
|----------------------------------|------------------------------------------------------------------------------------------------|---------------------------------------------------|
| Research direction               | Compare the decision evidence of AI models with physician annotations                          | Decided                                           |
| Research structure               | Rule-based baseline, classifier (Model A), and detector (Model B); Grad-CAM for the classifier | Decided (model details under discussion)          |
| Research questions               | Three research questions drafted                                                               | Draft                                             |
| Hypotheses and decision criteria | To be fixed before the test set is used                                                        | TBD                                               |
| Dataset                          | RSNA Pneumonia Detection Challenge 2018, final labels only                                     | Decided                                           |
| Data split                       | Single patient-level split (70% / 10% / 20%)                                                   | Decided                                           |
| Rule-based baseline design       | Binary classification with hand-set thresholds                                                 | Decided; region and time limit TBD                |
| Model A design                   | Network, training procedure, selection metric, and loss weighting                              | Under discussion                                  |
| Model B design                   | Detector, backbone, pre-trained weights, and training procedure                                | Under discussion                                  |
| Evaluation methods               | Common comparison, Grad-CAM metrics, and box categories                                        | Decided (Model A primary metric under discussion) |
| Reference performance levels     | ROC-AUC 0.80 (Model A) and mAP 0.40 (Model B)                                                  | Under discussion                                  |
| Data acquisition and inspection  | Acquire data and re-check the preliminary findings                                             | Not started                                       |
| Implementation and training      | Rule-based baseline, Model A, and Model B                                                      | Not started                                       |
| Evaluation and analysis          | Test-set comparison and decision-evidence analysis                                             | Not started                                       |
| Thesis writing                   | Results, discussion, and conclusions                                                           | Not started                                       |

# Chapter 6. Discussion, Limitations, and Mitigation Strategies

## 6.1 Current Discussion Policy

Agreement between Grad-CAM maps and physician boxes would show that the classifier responds to the annotated opacities, but it would not show that the model reasons in the same way as physicians. Disagreement, in turn, would not by itself indicate a new finding. In view of the supervisor's concern about the quality of XAI research, the evaluation metrics are fixed before the results are seen, results are compared with the chance level, shortcuts and missing annotations are examined first, and medical interpretations are made cautiously with advice from the supervisor or specialists.

| **Issue / limitation**                          | **Potential impact on the study**                                                                                     | **Mitigation strategy**                                                                                          |
|-------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Bounding Boxes are not complete lesion outlines | Normal tissue may be inside a box, and opacities may extend outside it.                                               | Use IoU-based criteria and the term “physician-annotated region,” and state the limitation.                      |
| Single-reader annotations                       | Some opacities may not be annotated, so a correct response may look like disagreement.                                | Repeat the analysis on images read by three radiologists (Section 4.6).                                          |
| Shortcut learning                               | View position, text markers, or devices may drive the predictions.                                                    | Stratify by AP and PA; check incidental content visually (Section 4.6).                                          |
| Coarse Grad-CAM maps                            | A 16×16 map localizes regions only roughly.                                                                           | Report the pointing game, which needs only the peak location, alongside IoU-based metrics; state the limitation. |
| Limits of XAI interpretation                    | A heatmap is not a complete explanation of the model's decision process.                                              | Fix metrics in advance; compare with the chance level; consider a sanity check (TBD); avoid causal claims.       |
| Effort on the rule-based baseline               | Classical methods may take much effort for little result.                                                             | Limit the scope to binary classification; set a time limit (TBD).                                                |
| Selected dataset                                | The dataset was selected to include many suspected pneumonia cases, so its prevalence differs from clinical practice. | Do not generalize values such as precision directly to clinical settings.                                        |
| Data leakage                                    | Images of the same patient in different sets would overestimate performance.                                          | Split by patient; use the test set only once.                                                                    |
| Data sharing and publication                    | Attribution requirements and university rules must be followed.                                                       | Use the official RSNA distribution and confirm the use of images with the supervisor.                            |

# Chapter 7. Conclusion and Future Research Plan

## 7.1 Next Tasks

| **Order** | **Task**                                                                                            | **Completion criteria**                                                             |
|-----------|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| 1         | Decide the items under discussion for Model A and Model B with the supervisor.                      | Decisions are recorded and the research structure is updated.                       |
| 2         | Acquire the RSNA data and inspect its structure; re-check the class proportions and view positions. | Files are acquired and the counts are confirmed.                                    |
| 3         | Create the patient-level 70% / 10% / 20% split.                                                     | A fixed split list exists with no patient in more than one set.                     |
| 4         | Build the rule-based baseline (binary classification).                                              | Validation accuracy and confusion matrix are obtained within the time limit.        |
| 5         | Implement and train Model A.                                                                        | The selected setting is chosen on the validation set.                               |
| 6         | Implement and train Model B.                                                                        | The selected setting and the confidence threshold are chosen on the validation set. |
| 7         | Evaluate the three methods on the test set once, and analyze the decision evidence.                 | Tables and figures for Sections 4.2, 4.4, and 4.5 are complete.                     |
| 8         | Verify evidence that differs from the physicians' annotations and write the thesis.                 | Conclusions are written in relation to the research questions.                      |

## 7.2 Items to Finalize Before Experiments

### Under discussion with the supervisor

- Model A: network (ResNet-101 or ResNet-50), training procedure (maximum epochs, patience, standard and alternative settings), model-selection metric, and class weighting of the loss.

- Model B: detector (Faster R-CNN, YOLO, or the existing Mask R-CNN code without the mask branch), backbone, pre-trained weights, and training procedure.

- Role of the reference performance levels (targets or reference values only).

### To be decided (TBD)

- Region used for the rule-based features, and the time limit for the baseline.

- Whether to include the Grad-CAM sanity check.

- Data augmentation.

- Whether to stratify the data split by class.

- Formal hypotheses and decision criteria.

- Final title of the research.

## 7.3 Preliminary Conclusion

The research structure has been revised to compare the decision evidence of a classifier and a detector with physician-annotated regions in the RSNA Pneumonia Detection Challenge 2018 dataset, with a simple rule-based baseline as a reference. Following the supervisor's comments, Grad-CAM is applied only to the classifier, the detector's boxes are used directly as its evidence, and the three methods are compared by accuracy and confusion matrices rather than ROC.

The next step is to settle the model settings that are under discussion, acquire and inspect the data, and create the patient-level split. The analysis of decision evidence will be carried out with metrics fixed in advance, and any evidence that differs from the physicians' annotations will be examined for shortcut learning and missing annotations before it is interpreted.

# References and Official Resources

\[1\] Radiological Society of North America (RSNA). “RSNA Pneumonia Detection Challenge (2018).” Official dataset distribution page. [Link](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018)

\[2\] RSNA. “Pneumonia Detection Challenge Terms of Use and Attribution.” [Link](https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/pneumonia-detection-challenge-terms-of-use-and-attribution.pdf)

\[3\] RSNA. “2018 RSNA Pneumonia Detection Challenge Dataset Description.” [Link](https://www.rsna.org/-/media/Files/RSNA/Education/AI%20resources%20and%20training/AI%20image%20challenge/RSNA-2018-Pneumonia-Detection-Challenge-Dataset-Description.ashx?hash=8331DDFA52AA5EE9C3C99B5983BB680D7BD2ABBF&la=en)

\[4\] Shih G, Wu CC, Halabi SS, et al. “Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia.” Radiology: Artificial Intelligence. 2019. [Link](https://doi.org/10.1148/ryai.2019180041)

\[5\] He K, Zhang X, Ren S, Sun J. “Deep Residual Learning for Image Recognition.” CVPR 2016. [Link](https://doi.org/10.1109/CVPR.2016.90)

\[6\] Ren S, He K, Girshick R, Sun J. “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.” NeurIPS 2015. [Link](https://arxiv.org/abs/1506.01497)

\[7\] Lin TY, Dollár P, Girshick R, He K, Hariharan B, Belongie S. “Feature Pyramid Networks for Object Detection.” CVPR 2017. [Link](https://doi.org/10.1109/CVPR.2017.106)

\[8\] Selvaraju RR, Cogswell M, Das A, et al. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.” ICCV 2017. [Link](https://doi.org/10.1109/ICCV.2017.74)

\[9\] Zhang J, Bargal SA, Lin Z, Brandt J, Shen X, Sclaroff S. “Top-Down Neural Attention by Excitation Backprop.” International Journal of Computer Vision. 2018. [Link](https://doi.org/10.1007/s11263-017-1059-x)

\[10\] Saporta A, et al. “Benchmarking saliency methods for chest X-ray interpretation.” Nature Machine Intelligence. 2022. [Link](https://doi.org/10.1038/s42256-022-00536-x)

\[11\] Hosmer DW, Lemeshow S, Sturdivant RX. Applied Logistic Regression, 3rd ed. Wiley. 2013.

\[12\] Yao S, et al. “Pneumonia Detection Using an Improved Algorithm Based on Faster R-CNN.” 2021. [Link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8081632/)

\[13\] “FPAENet: Pneumonia Detection Network Based on Feature Pyramid Attention Enhancement.” arXiv:2011.08706. 2020. [Link](https://arxiv.org/abs/2011.08706)

\[14\] Geirhos R, et al. “Shortcut learning in deep neural networks.” Nature Machine Intelligence. 2020. [Link](https://doi.org/10.1038/s42256-020-00257-z)

\[15\] DeGrave AJ, Janizek JD, Lee SI. “AI for radiographic COVID-19 detection selects shortcuts over signal.” Nature Machine Intelligence. 2021. [Link](https://doi.org/10.1038/s42256-021-00338-7)

\[16\] Zech JR, et al. “Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study.” PLOS Medicine. 2018. [Link](https://doi.org/10.1371/journal.pmed.1002683)

\[17\] Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. “Sanity Checks for Saliency Maps.” NeurIPS 2018. [Link](https://arxiv.org/abs/1810.03292)

\[18\] National Institutes of Health (NIH) Clinical Center. “ChestX-ray Dataset.” [Link](https://nihcc.app.box.com/v/ChestXray-NIHCC)

## Document Note

This report is a progress document based on the research structure as of 19 November 2026. Items marked “Under discussion” or “TBD” may change after the meeting with the supervisor. The preliminary data findings in Section 2.2 were obtained in an earlier project and will be re-checked after the data have been acquired. Bibliographic details and reported values from the literature will be verified against the original sources before the final thesis. Until experimental results are obtained, no conclusion about the research questions must be treated as a verified fact.
