# IDDAG

Slideslive Presentation: 
https://recorder-v3.slideslive.com/?share=96768&s=09f41687-e6b3-4daa-ab8e-e751bf277f67

Codes will be updated soon.

### CelebA: Image to Image Translation
We all have seen these types of Male to Female image translation by generative models which is quite impressive. But have you ever wondered what facial features a model exactly introduce while this translation?
![1707017936465.jpeg](assets%2F1707017936465.jpeg)
Does it introduce facial features such as "wearing lipstick" or "makeup" ?
What other features is it adding to the translated images? Is it adding some
features that it should not?

In this paper, we evaluated an image generative model that were pre-trained on the popular CelebFaces Attributes (CelebA) Dataset with a classifier and we observed the following bar plot from 850 samples.
![1707017933870.jpeg](assets%2F1707017933870.jpeg)
We observed that addition of some attributes are logical such as 81% of all input images had "Wearing_Lipstic" as new attribute. Similarly, "Heavy_Makeup"(68%), "Wearing_Earrings" (33%) or "No_Beard" (27%). These attributes can be considered causally related to Sex (Male/Female) attribute.

Some attributes are added even though they are non-causal. For example: 36% of the translated images are detected having the "Attractive" attribute (an attribute of the CelebA dataset) and 24% of images are having the "Young" attribute.

We analyzed the reason behind this as following:
The CelebA dataset contains images with high frequency of [young, female] or [attractive, female] labels compared to other combinations. The image translation model memorize the correlation between "Female" and "Young" or between "Female" and "Attractive" during their training. As a result, during translation they do not consider if an attribute is causally related to Sex or not,
they add it based on the spurious correlation they learned. Thus, we see more young female in the translated images which was not the goal.

In our paper, we proposed a sound and complete algorithm called ID-DAG that can estimate causal effects involving high-dimensional variables such as images. It can generate high-dimensional interventional samples given that we know the causal graph and the causal query is identifiable from observational data. 

