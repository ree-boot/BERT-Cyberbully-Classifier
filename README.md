# BERT-Cyberbully-Classifier using Deep Learning

The project uses PyTorch to set up a Deep Learning environment for BERT.

## Description

The project is based on the Bidirectional Encoder Representation for Transformers which is a Deep Learning model pre-trained by Google. This project finetunes the model with the aim to detect whether a comment/tweet is a cyberbully or not and if yes, then which category it belongs to.

## Getting Started

### Installing

* Download and setup PyTorch from official website.
* pip install pandas
* pip install numpy
* pip install scikit-learn
* pip install regex
* pip install transformers
* pip install streamlit

* Modifications:
  Original dataset has 6 classes of cyberbullying, in the Dataset CSV file, labels are numbered as:

0: non cyberbully  
1: age  
2: ethnicity  
3: gender  
4: other  
5: religion

I have not used the 4th (other) category as it has many duplicates and removing duplicates causes a huge imbalance in the dataset, many of these tweets are also ambiguous with non cyberbully ones.

### Executing program

* Step-by-step bullets
```
1: Run the Bert_classifier.py file.
2: Run the app.py file once the above has been completely executed.
```

## Help

Any advise for common problems or issues.
```
If you change the Bert_Classifier class in the program, then do not forget to make same chanes in the app.py file.
```


## License

This project is licensed under the MIT License - see the LICENSE.md file for details
