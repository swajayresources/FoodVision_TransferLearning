
"""07_food_vision_milestone_project_1.ipynb



# 07 Milestone Project 1: 🍔👁 Food Vision Big™



 **we've got the goal of beating [DeepFood](https://www.researchgate.net/publication/304163308_DeepFood_Deep_Learning-Based_Food_Image_Recognition_for_Computer-Aided_Dietary_Assessment)**, a 2016 paper which used a Convolutional Neural Network trained for 2-3 days to achieve 77.4% top-1 accuracy.

> 🔑 **Note:** **Top-1 accuracy** means "accuracy for the top softmax activation value output by the model" (because softmax ouputs a value for every class, but top-1 means only the highest one is evaluated). **Top-5 accuracy** means "accuracy for the top 5 softmax activation values output by the model", in other words, did the true label appear in the top 5 activation values? Top-5 accuracy scores are usually noticeably higher than top-1.

|  | 🍔👁 Food Vision Big™ 
|-----|-----|
| Dataset source | TensorFlow Datasets | 
| Train data | 75,750 images | 
| Test data | 25,250 images | 
| Mixed precision | Yes | 
| Data loading | Performanant tf.data API | 
| Target results | 77.4% top-1 accuracy (beat [DeepFood paper](https://arxiv.org/abs/1606.05675)) | 






## Check GPU


We're going to be using mixed precision training.

Mixed precision training was introduced in [TensorFlow 2.4.0](https://blog.tensorflow.org/2020/12/whats-new-in-tensorflow-24.html) (a very new feature at the time of writing).

What does **mixed precision training** do?

Mixed precision training uses a combination of single precision (float32) and half-preicison (float16) data types to speed up model training (up 3x on modern GPUs).

You can read the [TensorFlow documentation on mixed precision](https://www.tensorflow.org/guide/mixed_precision) for more details.

For now, if we want to use mixed precision training, we need to make sure the GPU powering our Google Colab instance (if you're using Google Colab) is compataible.

For mixed precision training to work, **you need access to a GPU with a compute compability score of 7.0+**.

Google Colab offers several kinds of GPU.

However, some of them **aren't compatiable with mixed precision training.**

Therefore to make sure you have access to mixed precision training in Google Colab, you can check your GPU compute capability score on [Nvidia's developer website](https://developer.nvidia.com/cuda-gpus#compute).

As of May 2023, the GPUs available on Google Colab which allow mixed precision training are:
* NVIDIA A100 (available with Google Colab Pro)
* NVIDIA Tesla T4

> 🔑 **Note:** You can run the cell below to check your GPU name and then compare it to [list of GPUs on NVIDIA's developer page](https://developer.nvidia.com/cuda-gpus#compute) to see if it's capable of using mixed precision training.
"""
