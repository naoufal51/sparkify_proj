import os
import shutil
import joblib
import wandb
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Dict, Optional
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType
from sparkify_proj.feature_engineering import FeatureEngineering
from sparkify_proj.pipeline import DataPipeline
from sparkify_proj.preprocessing import Preprocessing
from fastapi.middleware.cors import CORSMiddleware
from threading import Lock
import shap
from fastapi.staticfiles import StaticFiles
from fastapi import APIRouter
from starlette.responses import FileResponse


# Config for wandb
wandb_key = os.environ.get("WANDB_API_KEY")
wandb.login(key=wandb_key)

wandb_project = "sparkify_naoufal"

pipeline_name = "pipeline_model"
pipeline_version = "v0"
pipeline_name_version = f"{pipeline_name}:{pipeline_version}"

classifier_name = "XGBClassifier.pkl"
classifier_version = "v0"
classifier = f"{classifier_name}:{classifier_version}"


spark_lock = Lock()

app = FastAPI()

api_router = APIRouter()


# define data schema
schema = StructType(
    [
        StructField("artist", StringType(), True),
        StructField("auth", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("ts", LongType(), True),
        StructField("userId", StringType(), True),
        StructField("sessionId", LongType(), True),
        StructField("page", StringType(), True),
        StructField("method", StringType(), True),
        StructField("status", LongType(), True),
        StructField("level", StringType(), True),
        StructField("itemInSession", LongType(), True),
        StructField("location", StringType(), True),
        StructField("userAgent", StringType(), True),
        StructField("lastName", StringType(), True),
        StructField("firstName", StringType(), True),
        StructField("Song", StringType(), True),
        StructField("registration", LongType(), True),
        StructField("length", DoubleType(), True),
    ]
)


def remove_colon_from_dir_path(dir_path: str):
    if ":" in dir_path:
        new_dir_path = dir_path.replace(":", "_")
        if os.path.exists(new_dir_path):
            shutil.rmtree(new_dir_path)
        shutil.move(dir_path, new_dir_path)
    else:
        new_dir_path = dir_path
    return new_dir_path


def load_artifacts():
    with wandb.init(project=wandb_project) as run:
        pipeline_model_path = run.use_artifact(
            pipeline_name_version, type="pipeline_model"
        ).download()
        pipeline_model_path = remove_colon_from_dir_path(pipeline_model_path)
        pipeline_model = PipelineModel.load(pipeline_model_path)

        model_path = run.use_artifact(classifier, type="model").download()
        model = joblib.load(model_path + f"/{classifier_name}")

    return pipeline_model, model


def create_spark_session():
    import uuid

    app_name = f"Sparkify_{uuid.uuid4()}"
    spark = SparkSession.builder.master("local").appName(app_name).getOrCreate()

    return spark


@app.on_event("startup")
async def startup_event():
    global pipeline_model, model, spark
    pipeline_model, model = load_artifacts()
    spark = create_spark_session()


@api_router.post("/predict")
async def predict(user_data_file: UploadFile = File(...)):
    """
    Serves the model inference endpoint.

    Args:
        user_data_file (UploadFile): The user data file.

    Returns:
        Dict: A dictionary containing:
            - userId: user identification
            - churn_probability: User churn probability
            - SHAP_values: Get the importances of each feature that lead to the decision.

    Exception:
        HTTPException: If the user_data_file is not supported.


    """
    try:
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            temp_file.write(await user_data_file.read())
            temp_file.flush()

            spark = create_spark_session()

            user_data = spark.read.json(temp_file.name, schema=schema)

            # Preprocess the data
            preprocessor = Preprocessing(user_data)
            preprocessed_data = preprocessor.preprocess_data()

            # Feature engineering
            feature_engineer = FeatureEngineering(preprocessed_data)
            feature_engineered_data = feature_engineer.feature_engineering()

            # Run data through the pipeline
            data_pipeline = DataPipeline(feature_engineered_data, "./data")
            data_pipeline_data = data_pipeline.run_inference_pipeline(pipeline_model)

            # Predict the churn probability
            user_id = data_pipeline_data["userId"]
            data_pipeline_data = data_pipeline_data.drop("userId", axis=1)
            predictions = model.predict_proba(data_pipeline_data)

            # Compute SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data_pipeline_data)

            # Returns churn probability and SHAP values for each user in a dictionnary
            churn_probabilities = [float(x[1]) for x in predictions]
            shap_values_list = shap_values.tolist()
            prediction_and_explanation = []
            for uid, prob, shap_values in zip(
                user_id, churn_probabilities, shap_values_list
            ):
                contributions = dict(zip(data_pipeline_data.columns, shap_values))
                prediction_and_explanation.append(
                    {
                        "userId": uid,
                        "churn_probability": prob,
                        "contributions": contributions,
                    }
                )

            spark.stop()

            return {"prediction": prediction_and_explanation}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


app.include_router(api_router, prefix="/api")

app.mount(
    "/dashboard/static",
    StaticFiles(directory="dashboard/dist", html=True),
    name="static",
)


@app.get("/dashboard")
async def serve_index():
    return FileResponse("dashboard/dist/index.html")
