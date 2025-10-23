## This project is for my research study - <br><i>"Explainable Gene Importance Modeling in Breast Cancer Subtypes Using Logistic Regression and Attention-Based Deep Learning"</i></br>

This project INcludes two Machine learning models and Two Deep learning models .<br></br>

<table>
  <tr>
    <th>Machine Learning (ML)</th>
    <th>Deep Learning (DL)</th>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>Fully Connected Neural Network</td>
  </tr>
  <tr>
    <td>Random Forest / Gradient Boosting</td>
    <td>Attention-Based Deep Learning Model</td>
  </tr>
</table>

## Key points :-<br></br>

-ML models for baseline prediction<br></br>
-DL models for advanced prediction and gene importance<br></br>
-Explainable via SHAP and attention weights<br></br>

## Dataset used - ( METABRIC_RNA_Mutation)

To use this repo follow normal initializing steps and then install requirements from requirements.txt , then to run py script for ml run -
<code>
python src/ml.py
</code>

this will run and two graphs will pop up on the screen . these graphs are sns graphs close them to save the log reg and rf model.

Do the same for shap analysis. this time no graphs will pop up they will be directly saved.
<code>
python src/shap_analysis.py
</code>

Below are Graphs and images of confusion matrix of Ml models<br></br>

<img width="695" height="522" alt="image" src="https://github.com/user-attachments/assets/f958ec8e-43ef-4186-ae3c-a3938f3e6df8" />
<img width="1123" height="897" alt="image" src="https://github.com/user-attachments/assets/e60b928d-cd0e-45bd-88ab-8e1d4ba0523b" />
<img width="998" height="695" alt="graph 1" src="https://github.com/user-attachments/assets/ad5552dd-95de-465e-8666-d76704aac72f" />
<img width="1002" height="700" alt="graph 2" src="https://github.com/user-attachments/assets/abcd4a92-9ad5-47fc-bb1b-cd5f27dbe575" />
<img width="501" height="306" alt="image" src="https://github.com/user-attachments/assets/bca93e4a-e884-46be-a48d-99db01a70ecf" />        
<img width="464" height="232" alt="image" src="https://github.com/user-attachments/assets/971220d1-e2ea-4e99-ba6a-c910288ae90b" />
