# To run it open cmd and write "streamlit run diabetes_app.py" in directory where the file is located
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import time
# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #FF4B4B; text-align: center;}
    .section-header {font-size: 2rem; color: #1F77B4; border-bottom: 2px solid #1F77B4; padding-bottom: 0.3rem;}
    .subheader {font-size: 1.5rem; color: #2CA02C;}
    .info-text {background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">Diabetes Prediction App 🩸🩺🧪</h1>', unsafe_allow_html=True)
st.markdown("""
This app allows you to predict diabetes based on patient health metrics. Upload your dataset or use the default one, 
explore the data, train machine learning models, and make predictions.
""")

# Load default dataset
# @st.cache_data
# def load_default_data():
#     return pd.read_csv('diabetes.csv')

# Initialize session state for storing data and models
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'results' not in st.session_state:
    st.session_state.results = {}

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Dataset Upload", 
    "Data Exploration", 
    "Data Cleaning", 
    "Visualization", 
    "Model Training", 
    "Model Comparison", 
    "Prediction"
])

# Dataset Upload Section
if section == "Dataset Upload":
    st.markdown('<h2 class="section-header">Dataset Upload</h2>', unsafe_allow_html=True)
    
    # upload_option = st.radio("Choose data source:", 
    #                        ("Use default dataset", "Upload your own dataset"))
    upload_option = "Use default dataset"
    # if upload_option == "Use default dataset":
    #     df = load_default_data()
    #     st.session_state.df = df
    #     st.success("Default dataset loaded successfully!")
    # else:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("Dataset uploaded successfully!")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Display dataset info
        st.markdown('<h3 class="subheader">Dataset Overview</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**First 5 rows:**")
            st.dataframe(df.head())
        
        with col2:
            st.write("**Dataset Shape:**")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            
            st.write("**Null Values:**")
            null_counts = df.isnull().sum()
            st.dataframe(null_counts[null_counts > 0])
            
            st.write("**Duplicate Rows:**")
            st.write(f"{df.duplicated().sum()} duplicate rows found")
        
        # Summary statistics
        st.write("**Summary Statistics:**")
        st.dataframe(df.describe())
        
        st.write("**🎨 Boxplots Before Handling Outliers (Colored):**")

        fig, axes = plt.subplots(3, 3, figsize=(14, 10))
        axes = axes.flatten()

        numeric_cols = df.select_dtypes(include=[np.number]).columns  # include Outcome too

        # Use same Set2 color palette
        palette = sns.color_palette("Set2", len(numeric_cols))

        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                sns.boxplot(
                    y=df[col],
                    ax=axes[i],
                    color=palette[i % len(palette)]  # keep consistent palette
                )
                axes[i].set_title(col, fontsize=12, fontweight="bold", color="#333333")
                axes[i].set_ylabel("")

        plt.tight_layout()
        st.pyplot(fig)

# Data Exploration Section 
elif section == "Data Exploration": 
    st.markdown('<h2 class="section-header">Data Exploration</h2>', unsafe_allow_html=True) 
     
    if st.session_state.df is None: 
        st.warning("Please upload a dataset first!") 
    else: 
        df = st.session_state.df 
         
        # Dataset Info
        st.markdown("---")  
        st.subheader("📊 Dataset Info")
        buffer = io.StringIO() 
        df.info(buf=buffer) 
        s = buffer.getvalue() 
        st.text_area("Dataset Information:",s, height=500)  # wide + scrollable
         
        # Column Details
        st.markdown("---")  
        st.subheader("📋 Column Details") 
        col_details = pd.DataFrame({ 
            'Data Type': df.dtypes, 
            'Non-Null Count': df.count(), 
            'Null Count': df.isnull().sum() 
        }) 
        st.dataframe(col_details, use_container_width=True)  # full-width table
        
# ----------STOP HERE-----------------------------------
# Data Cleaning Section
elif section == "Data Cleaning":
    st.markdown('<h2 class="section-header">Data Cleaning & Preprocessing</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first!")
    else:
        df = st.session_state.df.copy()
        
        # Handle missing values
        # st.write("**Handling Missing Values:**")
        st.subheader("Handling Missing Values:")
        if df.isnull().sum().sum() > 0:
            df.fillna(df.mean().round(2), inplace=True)
            st.write("Missing values filled with column means")
        else:
            st.write("No missing values found")
        cols = ['Glucose', 'BloodPressure', 'BMI']
        for col in cols:
            df[col]=df[col].replace(0,df[col].median())
        df['SkinThickness']=df['SkinThickness'].replace(99,df['SkinThickness'].median())
        # Remove duplicates
        # st.write("**Removing Duplicates:**")
        st.subheader("Removing Duplicates:")
        initial_count = df.shape[0]
        df.drop_duplicates(inplace=True)
        final_count = df.shape[0]
        st.write(f"Removed {initial_count - final_count} duplicate rows")
        
        # Store cleaned dataframe
        st.session_state.df_cleaned = df
        
        # Display cleaned dataset info
        # st.write("**Cleaned Dataset Info:**")
        st.subheader("Cleaned Dataset:")
        st.dataframe(df)
        
        st.write(f"**New Shape:** Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        # Boxplot after handling outliers 
        st.subheader("🎨 Boxplots After Handling Outliers (Colored):")
        st.write("**After handling outliers, any remaining values are considered real and valid data points, not errors, and therefore should not be removed.**") 
        fig, axes = plt.subplots(3, 3, figsize=(14, 10)) 
        axes = axes.flatten() 

        numeric_cols = df.select_dtypes(include=[np.number]).columns 

        # Define a color palette (pick as many as you want)
        palette = sns.color_palette("Set2", len(numeric_cols))  

        for i, col in enumerate(numeric_cols): 
            if i < len(axes): 
                sns.boxplot(y=df[col], ax=axes[i], color=palette[i % len(palette)])  
                axes[i].set_title(col, fontsize=12, fontweight="bold", color="#E0AFAF") 
                axes[i].set_ylabel("")  # remove extra label for cleaner look

        plt.tight_layout() 
        st.pyplot(fig)

# Visualization Section
elif section == "Visualization":
    st.markdown('<h2 class="section-header">📊 Data Visualization</h2>', unsafe_allow_html=True)

    if st.session_state.df_cleaned is None:
        st.warning("⚠️ Please clean the dataset first!")
    else:
        df = st.session_state.df_cleaned.copy()

        # Consistent outcome colors
        outcome_colors = {0: "#124ae2", 1: "#ca611b"}  # Green = no diabetes, Red = diabetes

        # Available visualization options
        visualization_options = [
            "Class balance (Outcome 0 vs 1) / pie chart",
            "Outcome vs Age (Violin plot)",
            "Outcome vs BMI (Violin plot)",
            "Age vs Glucose (scatter plot)",
            "BMI vs Glucose (Scatter plot)",
            "Pregnancies vs Age (scatter plot)",
            "Correlation between features → heatmap",
            "Pairplot for dataset",
            "Histogram for all features"
        ]

        # Single / Multi select toggle
        multi_view = st.checkbox("Show multiple visualizations", value=False)

        if multi_view:
            selected_visualizations = st.multiselect(
                "Choose visualizations to display:",
                visualization_options,
                default=[visualization_options[0]]
            )
        else:
            selected_visualizations = [st.selectbox("Choose a visualization:", visualization_options)]

        # Loop through selections
        for selected_visualization in selected_visualizations:

            # --- PIE CHART: Class balance ---
            if selected_visualization == "Class balance (Outcome 0 vs 1) / pie chart":
                st.subheader("📊 Diabetes Outcome Distribution")

                counts = df['Outcome'].value_counts()
                fig = px.pie(
                    values=counts.values,
                    names=counts.index.map({0: "No Diabetes", 1: "Diabetes"}),
                    title="Diabetes Outcome Distribution",
                    color=counts.index,
                    color_discrete_map=outcome_colors,
                    hole=0.4
                )
                fig.update_traces(textinfo="percent+label+value", pull=[0, 0.1])
                st.plotly_chart(fig, use_container_width=True)

                st.write("**📈 Outcome Statistics:**")
                total = len(df)
                stats_df = pd.DataFrame({
                    "Metric": ["Total Samples", "No Diabetes", "Diabetes", "Diabetes Prevalence"],
                    "Value": [total, counts.get(0, 0), counts.get(1, 0), f"{(counts.get(1, 0)/total)*100:.1f}%"]
                })
                st.dataframe(stats_df)

            # --- VIOLIN: Age ---
            elif selected_visualization == "Outcome vs Age (Violin plot)":
                st.subheader("🎻 Age Distribution by Outcome")
                fig = px.violin(df, x="Outcome", y="Age", color="Outcome",
                                box=True, points="all", color_discrete_map=outcome_colors)
                fig.update_layout(xaxis=dict(tickvals=[0,1], ticktext=["No Diabetes","Diabetes"]))
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(df.groupby("Outcome")["Age"].describe().round(1))

            # --- VIOLIN: BMI ---
            elif selected_visualization == "Outcome vs BMI (Violin plot)":
                st.subheader("🎻 BMI Distribution by Outcome")
                fig = px.violin(df, x="Outcome", y="BMI", color="Outcome",
                                box=True, points="all", color_discrete_map=outcome_colors)
                fig.update_layout(xaxis=dict(tickvals=[0,1], ticktext=["No Diabetes","Diabetes"]))
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(df.groupby("Outcome")["BMI"].describe().round(1))

            # Custom color map for Outcome (0=Blue, 1=Orange)

            # --- SCATTER: Age vs Glucose --- 
            elif selected_visualization == "Age vs Glucose (scatter plot)": 
                fig = px.scatter(
                    df, x="Age", y="Glucose", color="Outcome", 
                    color_discrete_map={'0': "blue", '1': "orange"}, trendline="lowess", 
                    title="Age vs Glucose by Outcome", opacity=0.7
                ) 
                st.plotly_chart(fig, use_container_width=True) 
                st.write(f"Correlation: {df['Age'].corr(df['Glucose']):.3f}") 

            # --- SCATTER: BMI vs Glucose --- 
            elif selected_visualization == "BMI vs Glucose (Scatter plot)": 
                fig = px.scatter(
                    df, x="BMI", y="Glucose", color="Outcome", 
                    color_discrete_map={0: "blue", 1: "orange"}, trendline="lowess", 
                    title="BMI vs Glucose by Outcome", opacity=0.7
                ) 
                st.plotly_chart(fig, use_container_width=True) 
                st.write(f"Correlation: {df['BMI'].corr(df['Glucose']):.3f}") 

            # --- SCATTER: Pregnancies vs Age --- 
            elif selected_visualization == "Pregnancies vs Age (scatter plot)": 
                fig = px.scatter(
                    df, x="Age", y="Pregnancies", color="Outcome", 
                    color_discrete_map={0: "blue", 1: "orange"}, 
                    title="Pregnancies by Age and Outcome", opacity=0.7
                ) 
                st.plotly_chart(fig, use_container_width=True) 

            # --- HEATMAP: Correlation ---
            elif selected_visualization == "Correlation between features → heatmap":
                corr = df.corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto",
                                title="Feature Correlation Heatmap",
                                color_continuous_scale="Balance", zmin=-1, zmax=1)
                st.plotly_chart(fig, use_container_width=True)

                outcome_corr = corr['Outcome'].drop("Outcome").sort_values(ascending=False)
                st.dataframe(outcome_corr.to_frame("Correlation").round(3))

            # --- PAIRPLOT --- 
            elif selected_visualization == "Pairplot for dataset":
                import seaborn as sns
                import matplotlib.pyplot as plt
                
                features = st.multiselect(
                    "Select features:",
                    df.columns.tolist(),
                    default=["Glucose", "BMI", "Age", "Outcome"]
                )

                if len(features) > 1:
                    # Customize pairplot
                    fig = sns.pairplot(
                        df[features],
                        hue="Outcome",
                        diag_kind="kde",              # KDE on diagonal
                        palette="Set2",               # Color palette (try 'Set1', 'coolwarm', 'viridis')
                        height=3,                     # Size of each subplot (default 2.5)
                        plot_kws={'alpha': 0.7, 's': 20}  # scatter style: transparency & point size
                    )
                    st.pyplot(fig)
                else:
                    st.warning("Please select at least 2 features.")

            # --- HISTOGRAM: All features ---
            elif selected_visualization == "Histogram for all features":
                feature = st.selectbox("Select feature:", df.select_dtypes(include=[np.number]).columns)
                fig = px.histogram(df, x=feature, color="Outcome",
                                   marginal="box", nbins=30,
                                   color_discrete_map=outcome_colors, opacity=0.7)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df[feature].describe().round(2).to_frame().T)

# Model Training Section 
elif section == "Model Training": 
    st.markdown('<h2 class="section-header">Model Training</h2>', unsafe_allow_html=True) 
     
    if st.session_state.df_cleaned is None: 
        st.warning("Please clean the dataset first!") 
    else: 
        df = st.session_state.df_cleaned 
         
        # Prepare data for modeling 
        X = df.drop('Outcome', axis=1) 
        y = df['Outcome'] 
         
        # Train-test split 
        test_size = st.slider("Test set size:", 0.1, 0.4, 0.2, 0.05) 
        X_train, X_test, y_train, y_test = train_test_split( 
            X, y, test_size=test_size, random_state=42
        ) 
         
        # Store in session state 
        st.session_state.X_train = X_train 
        st.session_state.X_test = X_test 
        st.session_state.y_train = y_train 
        st.session_state.y_test = y_test 
         
        # Scaling options 
        st.write("**Feature Scaling:**") 
        scale_data = st.checkbox("Apply Standard Scaler", value=True) 
         
        if scale_data: 
            scaler = StandardScaler() 
            X_train_scaled = scaler.fit_transform(X_train) 
            X_test_scaled = scaler.transform(X_test) 
            st.session_state.scaler = scaler 
             
            st.write("**Scaled Training Data (first 5 rows):**") 
            st.dataframe(pd.DataFrame(X_train_scaled, columns=X.columns).head()) 
        else: 
            X_train_scaled = X_train 
            X_test_scaled = X_test 
            st.session_state.scaler = None 
         
        # Model selection 
        st.write("**Select a Model to Train:**") 
        model_option = st.selectbox( 
            "Choose a model:", 
            ["Logistic Regression", "Neural Network", "Random Forest",  
             "K-Nearest Neighbors", "Support Vector Machine", "Naive Bayes"] 
        ) 
         
        # Model parameters 
        if model_option == "Logistic Regression": 
            C = st.slider("Regularization parameter (C):", 0.01, 10.0, 1.0) 
            model = LogisticRegression(multi_class='multinomial',C=C,random_state=42) 
         
        elif model_option == "Neural Network": 
            hidden_layer_sizes = st.selectbox("Hidden layers:", [(16,8), (32,16), (64,32)]) 
            model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                                  activation="relu", solver="adam", 
                                  max_iter=500, random_state=42) 
         
        elif model_option == "Random Forest": 
            n_estimators = st.slider("Number of trees:", 10, 200, 20) 
            max_depth = st.slider("Max depth:", 1, 20, 10) 
            model = RandomForestClassifier(n_estimators=n_estimators,  
                                         max_depth=max_depth, random_state=42) 
         
        elif model_option == "K-Nearest Neighbors": 
            n_neighbors = st.slider("Number of neighbors:", 1, 20, 5) 
            model = KNeighborsClassifier(n_neighbors=n_neighbors) 
         
        elif model_option == "Support Vector Machine": 
            # C = st.slider("Regularization parameter (C):", 0.01, 10.0, 1.0) 
            kernel = st.selectbox("Kernel:", ["linear", "rbf", "poly"]) 
            model = SVC( kernel=kernel, random_state=42, probability=True) 
         
        elif model_option == "Naive Bayes": 
            model = GaussianNB() 
         
        # Train model 
        if st.button("Train Model"): 
            with st.spinner(f"Training {model_option}..."): 
                model.fit(X_train_scaled, y_train) 
                y_pred = model.predict(X_test_scaled) 
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None 
                 
                # Calculate metrics (only accuracy for simplicity) 
                accuracy = accuracy_score(y_test, y_pred) 
                 
                # Store model and results 
                st.session_state.models[model_option] = model 
                st.session_state.results[model_option] = { 
                    'accuracy': accuracy, 
                    'y_pred': y_pred, 
                    'y_pred_proba': y_pred_proba 
                } 
                 
                # Display results 
                st.success(f"{model_option} trained successfully!") 
                st.metric("Accuracy", f"{accuracy:.4f}") 
                 
                # Confusion matrix 
                st.write("**Confusion Matrix:**") 
                cm = confusion_matrix(y_test, y_pred) 
                fig = px.imshow(cm, text_auto=True,  
                              labels=dict(x="Predicted", y="Actual", color="Count"), 
                              x=['No Diabetes', 'Diabetes'], 
                              y=['No Diabetes', 'Diabetes'], 
                              title="Confusion Matrix") 
                st.plotly_chart(fig, use_container_width=True) 
                 
                # ROC Curve (if model supports probabilities) 
                if y_pred_proba is not None: 
                    st.write("**ROC Curve:**") 
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba) 
                    roc_auc = auc(fpr, tpr) 
                     
                    fig = go.Figure() 
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',  
                                           name=f'ROC curve (AUC = {roc_auc:.2f})')) 
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',  
                                           name='Random classifier', line=dict(dash='dash'))) 
                     
                    fig.update_layout( 
                        title="ROC Curve", 
                        xaxis_title="False Positive Rate", 
                        yaxis_title="True Positive Rate", 
                        width=700, height=500 
                    ) 
                    st.plotly_chart(fig, use_container_width=True) 

# Model Comparison Section 
elif section == "Model Comparison": 
    st.markdown('<h2 class="section-header">Model Comparison</h2>', unsafe_allow_html=True) 
     
    if not st.session_state.results: 
        st.warning("Please train at least one model first!") 
    else: 
        # Create comparison dataframe (only accuracy) 
        comparison_data = [] 
        for model_name, metrics in st.session_state.results.items(): 
            comparison_data.append({ 
                'Model': model_name, 
                'Accuracy': metrics['accuracy'] 
            }) 
         
        comparison_df = pd.DataFrame(comparison_data) 
         
        st.write("**Model Accuracy Comparison:**") 
        st.dataframe(comparison_df) 
         
        # Visualization of model comparison 
        fig = go.Figure() 
        fig.add_trace(go.Bar( 
            x=comparison_df['Model'], 
            y=comparison_df['Accuracy'], 
            name="Accuracy" 
        )) 
         
        fig.update_layout( 
            title="Model Accuracy Comparison", 
            xaxis_title="Models", 
            yaxis_title="Accuracy", 
            yaxis_range=[0, 1] 
        ) 
        st.plotly_chart(fig, use_container_width=True) 


# Prediction Section
elif section == "Prediction":
    st.markdown('<h2 class="section-header">Diabetes Prediction</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models:
        st.warning("Please train at least one model first!")
    else:
        # Select model for prediction
        selected_model = st.selectbox(
            "Select a trained model for prediction:",
            list(st.session_state.models.keys())
        )
        
        model = st.session_state.models[selected_model]
        scaler = st.session_state.scaler
        
        # Input features
        st.write("**Enter Patient Details:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Glucose", min_value=0.0, max_value=200.0, value=100.0)
            blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
        
        with col2:
            skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
            insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        
        with col3:
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
        
        # Create feature array
        input_features = np.array([[pregnancies, glucose, blood_pressure, 
                                  skin_thickness, insulin, bmi, dpf, age]])
        
        # Scale features if scaler exists
        if scaler is not None:
            input_features = scaler.transform(input_features)
        
        # Make prediction
        if st.button("Predict Diabetes"):
            prediction = model.predict(input_features)
            prediction_proba = model.predict_proba(input_features)
            
            st.markdown("---")
            st.markdown('<h3 class="subheader">Prediction Result</h3>', unsafe_allow_html=True)
            
            if prediction[0] == 1:
                st.error(f"**Prediction:** The model predicts this patient has diabetes")
                        # f"(confidence: {prediction_proba[0][1]*100:.2f}%)")
            else:
                st.success(f"**Prediction:** The model predicts this patient does not have diabetes")
                        #   f"(confidence: {prediction_proba[0][0]*100:.2f}%)")
            
            # Show probability distribution
            # fig = px.bar(x=['No Diabetes', 'Diabetes'], 
            #            y=prediction_proba[0],
            #            labels={'x': 'Outcome', 'y': 'Probability'},
            #            title="Prediction Probability Distribution",
            #            color=['No Diabetes', 'Diabetes'],
            #            color_discrete_map={'No Diabetes': 'green', 'Diabetes': 'red'})
            
            # fig.update_layout(yaxis_range=[0, 1])
            # st.plotly_chart(fig, use_container_width=True)

# Footer 
st.markdown("---") 
st.markdown(""" 
<div style='text-align: center; font-size:20px; color:white;'> 
    <p><b>Diabetes Prediction App 🩸🩺🧪</b></p> 
    <p>This application is designed for educational and research purposes only, 
    helping users explore data, build models, and understand diabetes prediction insights.</p> 
    <p>Developed by <b>Ahmed Ashraf</b> | &copy; 2025 All Rights Reserved</p> 
</div> 
""", unsafe_allow_html=True) 
