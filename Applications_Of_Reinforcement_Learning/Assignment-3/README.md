### Assignment 3

#### Task 1:

- On April 10, 1912, the RMS Titanic sank after colliding with an iceberg on its maiden voyage. The accident killed 1502 of the 2224 passengers and crew on board.
- The dataset has the response: whether a passenger survived the tragedy (Survived). Additional information includes the sex of the passenger (1 if male, 0 otherwise)(Sexmale), age (Age), the number of siblings/spouse aboard (SibSp), the number of parents/children aboard (Parch) and the passengers fare (Fare) (in British pounds). The training data-set contains around 880 observations. Train your classification model using this data (train.csv) and calculate percentage accuracy of your model on test data set (test.csv).

#### Task 2:

- `Ghooster`(Arpit) was 20 years old when the tragedy happened. He boarded the ship with no family or spouse and paid 7.5 British pounds for his ticket.`Shreya nigwal` was 19 years old on the tragic day. She boarded the ship with her fiance (treat this as a spouse) and her mother. She paid 512 British pounds for her ticket. What are the estimated probabilities of survival for Kumar Arpit and Shreya Nigwal as per the model you have trained above?

#### Information about data sets:

1. train.csv - This data set is labeled with label name survived. You need  train your model using label  "survivied"
2. test.csv - You need make predictions using this data set.
3. gender_submission.csv - Compare the labels with the above predictions to calculate the accuracy.