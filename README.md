## Regression Models
- Least Squares 
- Ridge Regression
- Lasso
- PCR
- PLS

## Task
- Classifying Email as Spam or Non-Spam

## Data

- Spambase from UCI
- UCI explanation:
Our collection of spam e-mails came from our postmaster and individuals who had filed spam.  Our collection of non-spam e-mails came from filed work and personal e-mails, and hence the word 'george' and the area code '650' are indicators of non-spam.  These are useful when constructing a personalized spam filter.  One would either have to blind such non-spam indicators or get a very wide collection of non-spam to generate a general purpose spam filter.

For background on spam: Cranor, Lorrie F., LaMacchia, Brian A.  Spam!, Communications of the ACM, 41(8):74-83, 1998.

Typical performance is around ~7% misclassification error. False positives (marking good mail as spam) are very undesirable.If we insist on zero false positives in the training/testing set, 20-25% of the spam passed through the filter. See also Hewlett-Packard Internal-only Technical Report. External version forthcoming. 

## Results:
- Cross-fold validation is done. In the program, the mean and standard deviation for all the mentioned algorithms are calculated for all the folds. The results of the program are as follows:
- LS Mean = 0.12 , Std Dev = 0.016
- Ridge Regresson Mean = 0.12 , Std Dev = 0.02
- Lasso Mean = 0.11 , Std Dev = 0.012
- PLS Mean = 0.11 , Std Dev = 0.012
- PCR Mean = 0.32 , Std Dev = 0.03
