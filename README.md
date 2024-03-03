# NLP Assignment 5 (AIT - DSAI)

- [Student Information](#student-information)
- [Files Structure](#files-structure)
- [Task 1 - Training BERT from Scratch](#task-1---training-bert-from-scratch)
- [Task 2 - Sentence BERT](#task-2---sentence-bert)
- [Task 3 - Evaluation and Analysis](#task-3---evaluation-and-analysis)
- [Task 4 - Web Application](#task-4---web-application)
    - [How to run](#how-to-run)
    - [Usage](#usage)

## Student Information
 - Name: Myo Thiha
 - ID: st123783

## Files Structure
 - In the training folder, The Jupytor notebook files (training) can be located.
 - The 'app' folder include 
    - `app.py` file for the entry point of the web application
    - Note: I tried to use docker but not working because of the resource problem. 

## Task 1 - Training BERT from Scratch

- Dataset: I used the NLTK brown corpus's government category to train my BERT model.
- Save Model: Then, I save model weights which can be located in the `app/models/bert-from-scratch.pt`


## Task 2 - Sentence BERT

- I trained two models for sentence BERT: using pretrained BERT and BERT from scratch that I trained previously.


## Task 3 - Evaluation and Analysis

| Attentions          | Training Loss | Accuracy | Cosine Similarity | 
|----------------|-------------|---------------|---------------|--------------------|
| Setence BERT - Pretrained       |    1.14      |      34.60 % |       0.77        |
| Setence BERT - Scratch       |    1.33      |      35.30  |       0.996        |

## Task 4 - Web Application

### How to run?
 - First of all, I tried wiht docker, it runs everything except for the model inference. Docker cannot handle resource consumpiton of the model inference and stop working. 
 - Therefore, you have to run the app without docker.
 - How to run: go inside the app folder using terminal and run `python app.py`. Note that your python environment must contains the dependencies that are described in the requirements.txt.
 - Then, the application can be accessed on http://localhost:8000
 - You will directly land on the "Home" page.

### Usage:
- Input: There are two input fields where you can enter two sentences to compare.
- Output: afte you click the 'Check' button, the cosine similarity score can be seen.