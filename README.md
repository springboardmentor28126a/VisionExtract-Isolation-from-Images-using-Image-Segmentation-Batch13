# VisionExtract-Isolation-from-Images-using-Image-Segmentation-Batch13

**************************************************
Week 1-2 Update: Getting Hands-on with the Data
Goal: Tackle the 27GB COCO dataset and make sure I can actually use it.

This week was all about setting up the foundation. I successfully downloaded and extracted the full COCO 2017 dataset locally. Since 27GB is a lot to handle, my priority was ensuring the folder structure was correct and that my Python environment could read the files without crashing.

*Inspecting the Data*
To verify everything was working, I wrote a script to visualize the "ground truth" for my project.

Target Subject: "Person" class.

The Test: I randomly selected 5 images from the training set and generated their corresponding binary masks.

The Result: I successfully plotted the original images side-by-side with their segmentation masks. The masks accurately capture the pixel-level details of the subjects (people), proving that my pycocotools setup and file paths are working perfectly.