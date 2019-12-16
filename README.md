# Age-Estimation-Using-Deep-Transfer-Learning

In this project, the feature learning capabilities of deep convolutional neural networks (CNNs) are leveraged and a deep framework for domain adaptation in regression is used. 
 
Dataset: 
Program file for preprocessing and data collection:  term-progs-Singh_download_flicker_images.py term-progs-Singh_download_reddit_images.py term-progs-Singh_metadata_wiki_clean.py 
Two type of datasets for two different regression problems are used here. 
Problem 1: Predicting Image Popularity Dataset used:  
1.	Source Domain: The first data set consists of images from Reddit for which we used the Image dataset by Deza et al. [1]. After removing broken links and corrupted images, 9,977 labeled(with popularity score available) unique images were left. The popularity count was obtained by log normalized upvotes. This is a setting where images are posted anonymously and does not get influenced by the popularity of the person who posted the images. 
 
2.	Target Domain: The second dataset consists of 12,440 Flickr images obtained through the 
Flickr API belonging to a period of2months. Out of 12440, 6940 were labeled and 5500 were 
unlabeled. Out of 6940, we kep 1440 for training and 5500 for testing of the network. Popularity score was obtained through log normalized number of views [2]. This represents the setting where popularity of images is also influenced by the popularity of the person who posted the images. 
 
Problem 2: Predicting Age from Images 
Program file for preprocessing and data collection:  
Dataset used: 
1.	Source Domain: Female photographs from Wikipedia dataset[3]. This consisted of 3480 labeled images. Ages were available in years and were normalized by dividing by 101(assumed maximum age) to make the majority of labels fall in range 0-1.  
2.	Target Domain: Male photographs from Wikipedia dataset. This consisted of 4870 labeled and 2610 unlabeled images. Out of 4870, 870 were kept to train and remaining 4000 to test the network. Age is normalized in same manner as in source domain. 
 
Why is this important? 
To compare the results I obtained results using following baselines: 
1.	No Source: term-progs-Singh_no source.py 
 This shows the results when no source domain is used and network is trained using only labeled samples from target domain.  
2.	No Domain Adaptation: term-progs-Singh_NO_DA.py 
This shows the results when source labeled and target labeled samples are used to train without any MMD or smoothing loss term. 
 
Creating Model: 
term-progs-Singh_age.py term-progs-Singh_popularity.py term_progs_Singh_losses_mmd.py Baseline model consists of 7 Convolutional layer with “Relu” as activation function in all of them and one output layer with sigmoid activation function and 1 neuron(to obtain regression value) 
