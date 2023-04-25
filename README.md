# GDF
[IJCAI'23] Unbiased Risk Estimator to Multi-Labeled Complementary Label Learning (the appendix is shown in the .pdf file)

This code gives the implementation  of the paper "Unbiased Risk Estimator to Multi-Labeled Complementary Label Learning".

Requirements
- Python >=3.6
- PyTorch >=1.9

---
main.py
  >This is main function. After running the code, you should see a text file with the results saved in the same directory. The results will have seven columns: epoch number, training loss, hamming loss of test data, one error of test data, coverage of test data, ranking loss of test data and average precision of test data.

generate.py
  >This is used to generate complementary labels. After running, you should see a .csv file of complementary labels for a dataset in the vector form. If you have prepared the training data and its complemenatry labels, please ignore it. 
  
  
## Running

python main.py --lo \<method name\> --dataset \<dataset name\>

**Methods and models**

In main.py, specify the method argument to choose one of the 2 methods available:
- GDF: GDF loss function is defined by Equation(12) in the paper
- unbiase: BCE loss is used to MLCLL and derives an unbiased risk estimator, which is defined by Equation(9) in the paper

Specify the dataset argument:
- scene: scene dataset
- yeast: yeast dataset
