from data_handler import dataSetup
import streamlit as st
import pandas as pd
from pycaret.classification import *
from pycaret.regression import *



st.title("Build Your Model Using PyCaret")
st.markdown("### Upload Dataset")
st.markdown("Please upload your dataset in CSV, Excel (xls/xlsx), or JSON format.")
dataset = st.file_uploader("Upload Dataset", type=["csv", "xls", "xlsx", "json"])
if dataset is not None:
    if dataset.name.endswith("csv"):
        data = pd.read_csv(dataset)
    elif dataset.name.endswith("xls") or dataset.name.endswith("xlsx"):
        data = pd.read_excel(dataset)
    elif dataset.name.endswith("json"):
        data = pd.read_json(dataset)
    st.markdown("### Dataset Preview")
    st.write(data.head())
    st.markdown("### Dataset Description")
    st.write(data.shape)
    column_info = pd.DataFrame({
    'Column': data.columns,
    'Data Type': data.dtypes.values,
    'Non-Null Count': data.notnull().sum().values
    })
    st.write(column_info)
    st.write(data.describe())

    st.markdown("### Imputation Options")
    numeric_imputation = st.selectbox(
        "Numeric Imputation",
        [
            "mean",
            "median",
            "mode",
            "knn",
            "enter the value you want to write in place of missing numeric values",
        ],
    )
    if (
        numeric_imputation
        == "enter the value you want to write in place of missing numeric values"
    ):
        numeric_imputation = st.number_input(
            "Enter the value you want to write in place of missing numeric values", 0
        )
    categorical_imputation = st.selectbox(
        "Categorical Imputation",
        [
            "mode",
            "drop",
            "enter the value you want to write in place of missing categorical values",
        ],
    )
    if (
        categorical_imputation
        == "enter the value you want to write in place of missing categorical values"
    ):
        categorical_imputation = st.text_input(
            "Enter the value you want to write in place of missing categorical values",
            "missing",
        )

    st.markdown("### Features you want to change their datatypes")
    numeric_features = st.multiselect("Numeric Features", data.columns)
    categorical_features = st.multiselect(
        "Categorical Features", data.columns.difference(numeric_features)
    )
    date_features = st.multiselect(
        "Date Features",
        data.columns.difference(numeric_features).difference(categorical_features),
    )
    st.markdown("### Target Column")
    target = st.selectbox("Target Column", data.columns)

    if data[target].dtype == "object":
        problem_type = "Classification"
    else:
        problem_type = "Regression"

    st.markdown("### Features you want to ignore")
    ignore_features = st.multiselect("Ignore Features", data.columns)
    st.markdown("### Features you want to keep")
    keep_features = st.multiselect(
        "Keep Features", data.columns.difference(ignore_features)
    )
    st.markdown("### Features you want to encode as ordinal")
    ordinal_features = st.multiselect("Ordinal Features", data.columns)
    max_encoding_ohe = st.number_input("Max Encoding OHE", 25)
    st.markdown("### Remove Outliers Option")
    remove_outliers = st.checkbox("Remove Outliers")
    if remove_outliers:
        outliers_method = st.selectbox("Outliers Method", ["iforest", "ee", "lof"])
        outliers_threshold = st.number_input("Outliers Threshold", 0.05)
    else:
        outliers_method = "iforest"
        outliers_threshold = 0.05

    if st.button("Setup Experiment"):
        if problem_type == "Classification":
            s = ClassificationExperiment()
        elif problem_type == "Regression":
            s = RegressionExperiment()
        s= dataSetup(
            data=data,
            target=target,
            problem_type=problem_type,
            numeric_imputation=numeric_imputation,
            categorical_imputation=categorical_imputation,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            date_features=date_features,
            ignore_features=ignore_features,
            keep_features=keep_features,
            ordinal_features=ordinal_features,
            max_encoding_ohe=max_encoding_ohe,
            remove_outliers=remove_outliers,
            outliers_method=outliers_method,
            outliers_threshold=outliers_threshold,
        )
        st.success("Experiment Setup Complete")
        st.markdown("### Experiment Details")
        st.table(s.pull())
        st.write("Target Column: ", target)
        st.write("Problem Type: ", problem_type)
        st.markdown("### Compare Models")
        top3 = s.compare_models(n_select=3)
        st.table(s.pull())
        st.markdown("Choose the metric you want to use for model optimization")
        optimizer = st.selectbox("Optimizer", ["Accuracy", "AUC", "Recall", "Precision", "F1", "MSE", "RMSE", "MAE", "R2", "MAPE", "RMSLE", "QWK", "Kappa", "MCC"])
        tuned_top3 = [s.tune_model(i) for i in top3]
        blender = s.blend_models(tuned_top3)
        stacker = s.stack_models(tuned_top3)
        best_model = s.automl(optimize = optimizer)
        st.write("Best Model: ", best_model)
        st.markdown("### Save Model")
        save = st.button("Save Model")
        if save:
            s.save_model(best_model, "best_model")
            st.success("Model Saved")
            if st.button('Download Model'):
                st.download_button(label='Download Model', data=best_model, file_name='model.pkl')
