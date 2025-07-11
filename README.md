Summary: 

This project builds a machine learning pipeline to classify whether a social media comment was written by a human or artificial intelligence. The classifier leverages multiple sub-models that each focus on different aspects of human communication — such as emotion, sentiment, theme, and keyword structure — and combines them using an ensemble learning approach to produce a final prediction and confidence score. Initial data preprocessing and transformation were done on a subset of 100k human-written comments from Huggingface. 

Models used:    

Libraries used:

Goals:

Main - Detect and classify social media comments as either human-written or AI-generated
secondary: Build models that understand different, nuanced aspects of communication. (Sentiment, Emotion, Theme, and Keyword association)

Data:

Human Written comments:

Original size - 65M rows
subset used for training - 100k rows

Full dataset link - https://www.google.com/url?q=https://www.google.com/url?q%3Dhttps://huggingface.co/datasets/Exorde/exorde-social-media-december-2024-week1/viewer/default/train%26amp;sa%3DD%26amp;source%3Deditors%26amp;ust%3D1751584459308112%26amp;usg%3DAOvVaw1NJnDkDPV_Qb0LC-Jiy20I&sa=D&source=docs&ust=1751584459318454&usg=AOvVaw1rCjvYsMmVoXkXbNBFxSQi


AI-generated comments were synthesized using large language models

Pre-Processing: 

- Removed any non-English comments
- Dropped columns of date, userhash, URL, and language.

Sentiment Model:

Used a Ridge regression Cross Validation model to train. This model works well because the sentiment scores given to comments are on a scale of -1 to 1, Ridge does well with the continuous nature of the tasks, and also does a good job at adjusting weights for things like punctuation and emojis. (trained on 80k tested on 20k)

Mean Square Error: 0.0371
R² Score: 0.7799

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/d6ffc9dc-d952-492b-ab5f-ea3e6e4823aa" />

Emotion Pipeline Model: 

When training the emotion classifier model, I was having trouble with the model overfitting to the neutral class. To help with this, I had the idea to make an emotional pipeline. The first model decides if a comment is neutral or emotional, using a logistic regression model, since this is now a binary classification problem; Then the emotional comments go to the next model to classify their fine emotion, which uses an XGBoost Classifier because it is good with multiclass problems and also does well handling weights of underrepresented classes. 

Accuracy: 77%
Macro F1: 0.68
Weighted F1: 0.76


<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/a0edab54-ca65-44c0-b0ce-65f73294fa6e" />



Primary Theme Model: 

For this model again I go back to the XGBoost classifier, because of the multiclass nature of the problem, and the favorable results from the last model. Despite the drop in accurracy compared to the previous models, this model I still believe to be the best choice for my circumstances. The overlap between many themes likely made it difficult for the model to accurately decipher between them. 


Accuracy: 68%
Macro F1: 0.59
Weighted F1: 0.67


<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/a3574aef-0005-4b22-8d5b-f3de5c1a2ea5" />


Secondary Theme Model: 

This model solves a multi-label classification problem, each comment can be given 0, 1, or more labels for their secondary theme. For this model I used a one-vs-rest XGB classifier, creating a binary prediction for each individual label. This model struggles with under represented labels. 

Micro F1 score: 0.68
Macro F1 score: 0.47






Methodology:
The independent models for sentiment, main emotion, primary theme, and secondary theme were all trained, tested, and implemented, all while only seeing human-written comments. I made this decision because the purpose of the individual models is to excel at scoring a key aspect of the human language for a comment. The way to make sure that this is accurate is by giving it data that can be and has been scored on these important features, which are human-originated comments.   



