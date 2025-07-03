Summary: 
This project builds a machine learning pipeline to classify whether a social media comment was written by a human or artificial intelligence. The classifier leverages multiple sub-models that each focus on different aspects of human communication — such as emotion, sentiment, theme, and keyword structure — and combines them using an ensemble learning approach to produce a final prediction and confidence score. Initial data preprocessing and transformation was done on a subset of 100k human written comments, from huggingface. Models used are:    Libraries used are:
Goals:
Main: Detect and classify social media comments as either human-written or AI-generated
secondary: Build models that understance different nuanced aspects of communication. (Sentiment, Emotion, Theme and Keyword association)
Data:
Human Written comments:
Original size: 65M rows
subset used for training: 100k rows
Full dataset link: https://www.google.com/url?q=https://www.google.com/url?q%3Dhttps://huggingface.co/datasets/Exorde/exorde-social-media-december-2024-week1/viewer/default/train%26amp;sa%3DD%26amp;source%3Deditors%26amp;ust%3D1751584459308112%26amp;usg%3DAOvVaw1NJnDkDPV_Qb0LC-Jiy20I&sa=D&source=docs&ust=1751584459318454&usg=AOvVaw1rCjvYsMmVoXkXbNBFxSQi
AI generated comments: were synthesized using large language models


