import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import io

# --- 1. Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    df_orig = pd.read_csv('laptop_data_cleaned.csv')
    df = df_orig.copy() # Work on a copy

    # Sort as in notebook
    df = df.sort_values(by='TypeName').reset_index(drop=True)

    # Fill Ppi with mean value
    impute_ppi = SimpleImputer(strategy='mean')
    df['Ppi'] = impute_ppi.fit_transform(df[['Ppi']])
    df['Ppi'] = df['Ppi'].round(2)

    # Fill Weight with KNNImputer
    impute_weight = KNNImputer(n_neighbors=3)
    df['Weight'] = impute_weight.fit_transform(df[['Weight']])
    df['Weight'] = df['Weight'].round(2)

    # Store original categorical values for display purposes if needed later
    # For now, we'll encode directly for models.
    
    df_encoded = df.copy()
    # Encode categorical columns
    label_encoders = {}
    categorical_cols = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'Os']
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le # Store for potential inverse transform if needed for display

    return df_orig, df_encoded, label_encoders

# Load the data
df_original, df_processed, encoders = load_and_preprocess_data()
df_display = df_original.copy() # For EDA displays that prefer original labels

# --- 2. Sidebar Navigation ---
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", [
    "Data Overview",
    "Visualizations",
    "Regression Model (Price Prediction)",
    "Classification Model (TouchScreen Prediction)",
    "Unsupervised Model (KMeans Clustering)"
])

# --- 3. Page Content ---

# --- Section 1: Data Overview ---
if selection == "Data Overview":
    st.title("Laptop Data Overview 💻🧐")
    st.header("Raw Data")
    st.dataframe(df_original.head())

    st.header("Data Info")
    buffer = io.StringIO()
    df_original.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.header("Descriptive Statistics")
    st.dataframe(df_original.describe().T)

    st.header("Null Values Count")
    # Re-calculate nulls on original before imputation for accurate display
    df_null_check = pd.read_csv('laptop_data_cleaned.csv')
    st.dataframe(df_null_check.isnull().sum().rename("Null Values"))

# --- Section 2: Visualizations ---
elif selection == "Visualizations":
    st.title("Data Visualizations 📊")

    plot_choice = st.selectbox(
        "Choose a visualization:",
        [
            "Distribution of Laptop Prices",
            "Price vs RAM",
            "Weight vs Price",
            "TouchScreen vs Price",
            "SSD Size vs Price",
            "CPU Brand vs Price",
            "Operating System vs Price",
            "Feature Pairplot (colored by TypeName)",
            "Correlation Heatmap"
        ]
    )

    if plot_choice == "Distribution of Laptop Prices":
        st.subheader("Distribution of Laptop Prices")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df_display['Price'], kde=True, ax=ax)
        ax.set_title("Distribution of Laptop Prices")
        ax.set_xlabel("Price (Log Transformed)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    elif plot_choice == "Price vs RAM":
        st.subheader("Price vs RAM")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='Ram', y='Price', data=df_display, ax=ax)
        ax.set_title("Price vs RAM")
        st.pyplot(fig)

    elif plot_choice == "Weight vs Price":
        st.subheader("Weight vs Price")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x='Weight', y='Price', data=df_display, ax=ax)
        ax.set_title("Weight vs Price")
        st.pyplot(fig)
    
    elif plot_choice == "TouchScreen vs Price":
        st.subheader("TouchScreen vs Price")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='TouchScreen', y='Price', data=df_display, ax=ax)
        ax.set_title("TouchScreen vs Price")
        st.pyplot(fig)

    elif plot_choice == "SSD Size vs Price":
        st.subheader("SSD Size vs Price")
        # For better visualization, let's treat SSD as categorical for boxplot if many unique values
        df_ssd_plot = df_display.copy()
        if df_ssd_plot['SSD'].nunique() > 10: # Heuristic
            df_ssd_plot['SSD_str'] = df_ssd_plot['SSD'].astype(str) + 'GB'
            x_col = 'SSD_str'
        else:
            x_col = 'SSD'
        
        fig, ax = plt.subplots(figsize=(10, 6)) # Increased size for better readability
        sns.boxplot(x=x_col, y='Price', data=df_ssd_plot, ax=ax)
        ax.set_title("SSD Size vs Price")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    elif plot_choice == "CPU Brand vs Price":
        st.subheader("CPU Brand vs Price")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Cpu_brand', y='Price', data=df_display, ax=ax)
        ax.set_title("CPU Brand vs Price")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    elif plot_choice == "Operating System vs Price":
        st.subheader("Operating System vs Price")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='Os', y='Price', data=df_display, ax=ax)
        ax.set_title("Operating System vs Price")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    elif plot_choice == "Feature Pairplot (colored by TypeName)":
        st.subheader("Feature Pairplot (colored by TypeName)")
        st.info("This plot might take a moment to generate.")
        pair_fig = sns.pairplot(df_display,
                                vars=['Ram', 'HDD', 'SSD', 'Weight', 'Price'],
                                hue='TypeName')
        st.pyplot(pair_fig)

    elif plot_choice == "Correlation Heatmap":
        st.subheader("Correlation Heatmap of Numerical Features")
        # Use the processed dataframe for Ppi and Weight as they have filled NaNs
        # For heatmap, only numerical columns are useful. Original df is fine for most.
        numeric_cols_for_heatmap = ['Ram', 'Weight', 'Price', 'Ppi', 'HDD', 'SSD', 'Ips', 'TouchScreen']
        
        # Ensure all columns are numeric, Ppi and Weight from df_processed
        df_heatmap = df_original[numeric_cols_for_heatmap].copy()
        df_heatmap['Ppi'] = df_processed['Ppi'] # Use imputed Ppi
        df_heatmap['Weight'] = df_processed['Weight'] # Use imputed Weight
        
        corr = df_heatmap.corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=True, cmap='viridis', fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)


# --- Section 3: Regression Model (Neural Network for Price Prediction) ---
elif selection == "Regression Model (Price Prediction)":
    st.title("Regression Model: Laptop Price Prediction")
    st.write("This model predicts the laptop price using a Neural Network.")

    @st.cache_resource # Cache the model and its training history
    def train_and_evaluate_regression_model(data):
        X = data.drop('Price', axis=1)
        y = data['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=0.95)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        # Build Model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_pca.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1) 
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history = model.fit(X_train_pca, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0) # verbose=0 for streamlit
        
        y_pred = model.predict(X_test_pca)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        loss, mae = model.evaluate(X_test_pca, y_test, verbose=0)
        
        return mse, r2, loss, mae, history, y_test, y_pred

    mse, r2, test_loss, test_mae, history, y_test_reg, y_pred_reg = train_and_evaluate_regression_model(df_processed)

    st.subheader("Model Evaluation Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error (MSE) on Test Set", f"{mse:.2f}")
    col2.metric("R² Score on Test Set", f"{r2:.2f}")
    col1.metric("Test Loss (Keras MSE)", f"{test_loss:.2f}")
    col2.metric("Test Mean Absolute Error (MAE)", f"{test_mae:.2f}")

    st.subheader("Training History")
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(history.history['loss'], label='Train Loss')
    ax_loss.plot(history.history['val_loss'], label='Validation Loss')
    ax_loss.set_title('Model Loss Over Epochs')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss (MSE)')
    ax_loss.legend()
    st.pyplot(fig_loss)

    fig_mae, ax_mae = plt.subplots()
    ax_mae.plot(history.history['mae'], label='Train MAE')
    ax_mae.plot(history.history['val_mae'], label='Validation MAE')
    ax_mae.set_title('Model MAE Over Epochs')
    ax_mae.set_xlabel('Epoch')
    ax_mae.set_ylabel('Mean Absolute Error')
    ax_mae.legend()
    st.pyplot(fig_mae)
    
    st.subheader("Actual vs. Predicted Prices (Sample)")
    comparison_df = pd.DataFrame({'Actual Price': y_test_reg.flatten(), 'Predicted Price': y_pred_reg.flatten()})
    st.dataframe(comparison_df.head(10))


# --- Section 4: Classification Model (Neural Network for TouchScreen) ---
elif selection == "Classification Model (TouchScreen Prediction)":
    st.title("Classification Model: TouchScreen Prediction")
    st.write("This model predicts whether a laptop has a TouchScreen using a Neural Network.")

    @st.cache_resource
    def train_and_evaluate_classification_model(data):
        X = data.drop('TouchScreen', axis=1)
        y = data['TouchScreen']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=0.95)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        model = Sequential([
            Dense(32, activation='relu', input_shape=(X_train_pca.shape[1],)),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(X_train_pca, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        
        loss, accuracy = model.evaluate(X_test_pca, y_test, verbose=0)
        y_pred_proba = model.predict(X_test_pca)
        y_pred_class = (y_pred_proba > 0.5).astype(int) # Convert probabilities to class labels

        return accuracy, history, y_test, y_pred_class

    accuracy, history_cls, y_test_cls, y_pred_cls = train_and_evaluate_classification_model(df_processed)

    st.subheader("Model Evaluation Metrics")
    st.metric("Accuracy on Test Set", f"{accuracy:.2%}")
    
    st.subheader("Training History")
    fig_acc_cls, ax_acc_cls = plt.subplots()
    ax_acc_cls.plot(history_cls.history['accuracy'], label='Train Accuracy')
    ax_acc_cls.plot(history_cls.history['val_accuracy'], label='Validation Accuracy')
    ax_acc_cls.set_title('Model Accuracy Over Epochs')
    ax_acc_cls.set_xlabel('Epoch')
    ax_acc_cls.set_ylabel('Accuracy')
    ax_acc_cls.legend()
    st.pyplot(fig_acc_cls)

    fig_loss_cls, ax_loss_cls = plt.subplots()
    ax_loss_cls.plot(history_cls.history['loss'], label='Train Loss')
    ax_loss_cls.plot(history_cls.history['val_loss'], label='Validation Loss')
    ax_loss_cls.set_title('Model Loss Over Epochs')
    ax_loss_cls.set_xlabel('Epoch')
    ax_loss_cls.set_ylabel('Loss (Binary Crossentropy)')
    ax_loss_cls.legend()
    st.pyplot(fig_loss_cls)
    
    st.subheader("Actual vs. Predicted TouchScreen (Sample)")
    comparison_cls_df = pd.DataFrame({'Actual TouchScreen': y_test_cls.flatten(), 'Predicted TouchScreen': y_pred_cls.flatten()})
    st.dataframe(comparison_cls_df.head(10))


# --- Section 5: Unsupervised Model (KMeans Clustering) ---
elif selection == "Unsupervised Model (KMeans Clustering)":
    st.title("Unsupervised Model: KMeans Clustering")
    st.write("This model groups laptops into 5 clusters based on RAM, Weight, HDD, SSD, and TouchScreen features.")

    @st.cache_data
    def run_kmeans_clustering(data):
        # Use the original df before encoding for these specific columns,
        # as the notebook did not scale/PCA them for K-Means
        # However, TouchScreen is already 0/1. Others are numeric.
        # Ram, Weight, HDD, SSD are available in df_processed (after imputation of Weight)
        # TouchScreen is also there.
        
        # The notebook used the df *after* encoding, so we should use df_processed for consistency
        # But the selected features for KMeans were 'Ram', 'Weight', 'HDD', 'SSD', 'TouchScreen'
        # Let's select these from the df_processed (which has imputations and encodings)
        unsupervised_features = data[['Ram', 'Weight', 'HDD', 'SSD', 'TouchScreen']].copy()

        kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto').fit(unsupervised_features) # n_init to suppress warning
        
        # Create a copy to avoid modifying the cached df_processed directly if it's used elsewhere
        df_clustered = data.copy() 
        df_clustered['Cluster_Evaluation'] = kmeans.labels_
        
        # Select only the relevant columns for display as per notebook's unSupervised_df
        display_cols = ['Ram', 'Weight', 'HDD', 'SSD', 'TouchScreen', 'Cluster_Evaluation']
        return df_clustered[display_cols]

    unsupervised_df_with_clusters = run_kmeans_clustering(df_processed)

    st.subheader("Clustered Data (with Evaluation column)")
    st.dataframe(unsupervised_df_with_clusters)

    st.subheader("Cluster Distribution")
    fig_cluster, ax_cluster = plt.subplots()
    sns.countplot(x='Cluster_Evaluation', data=unsupervised_df_with_clusters, ax=ax)
    ax_cluster.set_title("Distribution of Laptops Across Clusters")
    st.pyplot(fig_cluster)