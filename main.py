import streamlit as st
import pandas as pd 
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


from submodule.load_data import load_data
from submodule.plot_graph import plot_graph
from submodule.train_model import train_model

def main():
    st.title("Heart Attack Analysis & Prediction Dashboard")
    data = load_data()
    page = st.sidebar.selectbox("Select a page:", ["Data", "Exploration", "Modelling"])

    if page == "Data":
        st.header("Understanding Data")
        st.text("1. Dataset Preview on the first 5 rows of the data:")
        st.write(data.head())
    
        st.text("1.1 Data Dictionary")
        data_dictionary = """
        <ul style='margin-left: 30px;'>
            <li><strong>age</strong>: Age of the patient</li>
            <li><strong>sex</strong>: Sex of the patient</li>
            <li><strong>cp</strong>: Chest pain type
                <ul>
                    <li>0 = Typical Angina</li>
                    <li>1 = Atypical Angina</li>
                    <li>2 = Non-anginal Pain</li>
                    <li>3 = Asymptomatic</li>
                </ul>
            </li>
            <li><strong>trtbps</strong>: Resting blood pressure (in mm Hg)</li>
            <li><strong>chol</strong>: Cholesterol in mg/dl fetched via BMI sensor</li>
            <li><strong>fbs</strong>: (fasting blood sugar > 120 mg/dl)
                <ul>
                    <li>1 = True</li>
                    <li>0 = False</li>
                </ul>
            </li>
            <li><strong>restecg</strong>: Resting electrocardiographic results
                <ul>
                    <li>0 = Normal</li>
                    <li>1 = ST-T wave normality</li>
                    <li>2 = Left ventricular hypertrophy</li>
                </ul>
            </li>
            <li><strong>thalachh</strong>: Maximum heart rate achieved</li>
            <li><strong>oldpeak</strong>: Previous peak</li>
            <li><strong>slp</strong>: Slope</li>
            <li><strong>caa</strong>: Number of major vessels</li>
            <li><strong>thall</strong>: Thalium Stress Test result (0 to 3 </li>
            <li><strong>exng</strong>: Exercise induced angina
                <ul>
                    <li>1 = Yes</li>
                    <li>0 = No</li>
                </ul>
            </li>
            <li><strong>output</strong>: Target variable</li>
        </ul>
        """
        st.markdown(data_dictionary, unsafe_allow_html=True)

        st.write("")
        st.write("")
        st.text("2. To understand the shape of the data.")
        st.write("The shape of the dataset is: ", data.shape)
        st.write("Rows    : ", data.shape[0])
        st.write("Columns : ", data.shape[1])

        st.write("")
        st.write("")
        st.text("3. To know the data type.")
        buffer = io.StringIO()
        data.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
        
        st.write("")
        st.write("")
        st.text("4. Checking on the number of unique values in each column.")
        # Create a dictionary of unique counts for each column
        unique_counts = {col: data[col].nunique() for col in data.columns}
        # Convert to DataFrame and transpose for better readability
        unique_counts_df = pd.DataFrame(unique_counts, index=["Unique Count"]).transpose()
        # Display the DataFrame in Streamlit
        st.dataframe(unique_counts_df, width=200)

        st.write("")
        st.write("")
        st.text("5. Checking on missing values.")
        miss_val = data.isnull().sum()
        # Display in Streamlit
        st.dataframe(miss_val, width = 200)

        st.write("")
        st.write("")
        st.text("6. Separating the columns in categorical and continuous.")
        st.write("The categorial cols are : sex, exng, caa, cp, fbs, restecg, slp, thall")
        st.write("The continuous cols are : age, trtbps, chol, thalachh, oldpeak")
        st.write("The target variable is :  output")

        
        st.write("")
        st.write("")
        st.text("7. To view the summary statistics for continouous colums.")
        con_cols = ["age","trtbps","chol","thalachh","oldpeak"]
        sum_stat = data[con_cols].describe().transpose()
        st.dataframe(sum_stat)

    elif page == "Exploration":
            st.title("Exploratory Data Analysis")
        
            # Call the plot_graph function to get all figures
            fig1, fig2, fig3, fig4, fig5, fig6, fig7 = plot_graph(data)
        
            # Display each figure in Streamlit
            st.subheader("Fig 1: Count plot for Categorical Features")
            st.pyplot(fig1)
        
            st.subheader("Fig 2: Boxen plot for Continuous Features")
            st.pyplot(fig2)
        
            st.subheader("Fig 3: Target Count Distribution")
            st.pyplot(fig3)
        
            st.subheader("Fig 4: Correlation Matrix Heatmap")
            st.pyplot(fig4)
        
            st.subheader("Fig 5: Scatterplot Heatmap")
            st.pyplot(fig5)
        
            st.subheader("Fig 6: KDE Plots for Continuous Features by Target Variable")
            st.pyplot(fig6)
        
            st.subheader("Fig 7: Additional Distribution Plots")
            st.pyplot(fig7)
    
    elif page == "Modelling":
            st.title("Model Training")

            # Creating centered layout using columns
            col1, col2, col3 = st.columns([2, 1, 1])  # col1 and col3 will act as margins
        
            with col1:  # Center content in col2
                # Model selection dropdown
                model_type = st.selectbox("Select a model type", ["SVM", "LogisticRegression", "DecisionTree", "RandomForest", "GradientBoosting"])
        
                # Display option to choose a tuned model only for SVM
                return_tuned = False  # Default for non-SVM models
                if model_type == "SVM":
                    st.write("### Use Tuned Model?")
                    use_tuned_radio = st.radio("Use Tuned Model?", ["Yes", "No"], index=1)
                    return_tuned = use_tuned_radio == "Yes"
        
                # Add a button to train the model
                train_button = st.button("Train Model")
        
            if train_button:
                # Train and evaluate the model based on the selection
                if model_type == "SVM":
                    # Call `train_model` with `use_tuned` argument for SVM
                    model, accuracy, report, fig, fig_roc = train_model(model_type, data, use_tuned=return_tuned)
                else:
                    # For other models, no tuning is done
                    model, accuracy, report, fig, fig_roc = train_model(model_type, data)
        
                # Display results
                st.write(f"Test Accuracy: {accuracy:.2f}")
                report_df = pd.DataFrame(report).transpose()
                # Move the accuracy to its own row with f1-score and support populated
                accuracy_value = report_df.loc["accuracy", "f1-score"]  # Get the accuracy value
                total_support = report_df.loc["0", "support"] + report_df.loc["1", "support"]  # Sum of actual class supports
                
                # Create a new row for accuracy with empty precision and recall, and set f1-score and support
                report_df.loc["accuracy"] = {
                    "precision": None,  # Leave precision empty
                    "recall": None,     # Leave recall empty
                    "f1-score": accuracy_value,  # Accuracy in f1-score column
                    "support": total_support  # Accuracy support value
                }
                
                # Now, display this properly formatted classification report in Streamlit
                st.write("Classification Report:")
                
                # We will render the DataFrame using Streamlit's `st.dataframe`
                st.dataframe(report_df.style.format(precision=3)) 
                st.pyplot(fig)  # Display the confusion matrix plot
        
                # Display ROC Curve only for Logistic Regression
                if model_type == "LogisticRegression" and fig_roc is not None:
                    st.pyplot(fig_roc)
                    
if __name__ == '__main__':
    main()
