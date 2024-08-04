# The core code of MFCFN-SMC
This is the core code of a grade prediction model for foam flotation industrial process. In the article, we named it MFCFN-SMC.<br>
First of all, you need to extract the *transformer.rar*.<br>
Next, in your main function, `from VideoTransNet import VTN`. You can call the network structure from VideoTransNet.py.<br>
It should be noted that the hyperparameters in the network model need to be modified based on the input data and requirements of their own experiments.<br>
We provide a reference for hyperparameters in *config.yaml*.<br>
Lastly, The *FISM.py* file contains the entire process of FISM calculation in this article.
