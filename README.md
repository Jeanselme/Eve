# Gradient Descent & Eve
Compare gradient descent algorithm to Eve.

## References

### Article
Eve algorithm is based on the article : [Improving Stochastic Gradient Descent With Feedback](https://arxiv.org/pdf/1611.01505v2.pdf) by Jayanth Koushik & Hiroaki Hayashi.

## Execution
To download data (at the first use), execute :  
```
python3.5 dataExtraction.py
```

To compute a comparison of the different algorithm, execute :  
```
python3.5 performances.py
```

All the needed algorithm are present in the gradientDescent.py file.

## Results
We compute the comparison on the dataset : [Ionosphere Data Set](http://archive.ics.uci.edu/ml/datasets/Ionosphere). The goal is to predict the class of the radar.  

## Libraries
Needs numpy, scipy and sys. Compiled with python3.5
