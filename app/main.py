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


# threading lock
spark_lock = Lock()

wandb_key = os.environ.get('WANDB_API_KEY')
wandb.login(key=wandb_key)

# define app
app = FastAPI()

api_router = APIRouter()




# CORS configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )



# define data schema
schema = StructType([
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
    StructField("length", DoubleType(), True)
])


def remove_colon_from_dir_path(dir_path: str):
    if ':' in dir_path:
        new_dir_path = dir_path.replace(':', '_')
        if os.path.exists(new_dir_path):
            shutil.rmtree(new_dir_path)
        shutil.move(dir_path, new_dir_path)
    else:
        new_dir_path = dir_path
    return new_dir_path


def load_artifacts():
    with wandb.init(project='sparkify_v2') as run:
        pipeline_model_path = run.use_artifact(
            'pipeline_model:v0', type='pipeline_model').download()
        pipeline_model_path = remove_colon_from_dir_path(pipeline_model_path)
        pipeline_model = PipelineModel.load(pipeline_model_path)

        model_path = run.use_artifact(
            'XGBClassifier.pkl:v8', type='model').download()
        model = joblib.load(model_path + '/XGBClassifier.pkl')

    return pipeline_model, model



def create_spark_session():
    import uuid
    # Generate a unique app name
    app_name = f"Sparkify_{uuid.uuid4()}"
    spark = SparkSession.builder.master(
        "local").appName(app_name).getOrCreate()

    return spark


@app.on_event("startup")
async def startup_event():
    global pipeline_model, model, spark
    pipeline_model, model = load_artifacts()
    spark = create_spark_session()

@api_router.post("/testpost")
def test_post():
    return {"message": "Post successful!"}

@api_router.post("/predict")
async def predict(user_data_file: UploadFile = File(...)):
    try:

        # Creating a temporary file
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            temp_file.write(await user_data_file.read())
            temp_file.flush()

            # Create a new spark session
            spark = create_spark_session()

            # Read user data into a dataframe
            user_data = spark.read.json(temp_file.name, schema=schema)

            # Preprocess the data
            preprocessor = Preprocessing(user_data)
            preprocessed_data = preprocessor.preprocess_data()

            # Feature engineering
            feature_engineer = FeatureEngineering(preprocessed_data)
            feature_engineered_data = feature_engineer.feature_engineering()

            # Run data through the pipeline
            data_pipeline = DataPipeline(feature_engineered_data, './data')
            data_pipeline_data = data_pipeline.run_inference_pipeline(
                pipeline_model)

            # Separate userId from other features
            user_id = data_pipeline_data['userId']
            data_pipeline_data = data_pipeline_data.drop('userId', axis=1)

            # Predict the churn probability
            predictions = model.predict_proba(data_pipeline_data)

            # Compute SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data_pipeline_data)

            # Extract the probability of churn
            churn_probabilities = [float(x[1]) for x in predictions]

            # Convert SHAP values to list of lists for JSON serialization
            shap_values_list = shap_values.tolist()

            # Create a dictionary with userIds as keys and (churn_probability, SHAP_values) as values
            prediction_and_explanation = []
            for uid, prob, shap_values in zip(user_id, churn_probabilities, shap_values_list):
                # We convert the feature contributions into a dictionary to make it easier to interpret
                contributions = dict(
                    zip(data_pipeline_data.columns, shap_values))
                prediction_and_explanation.append(
                    {"userId": uid, "churn_probability": prob, "contributions": contributions})

            spark.stop()

            return {'prediction': prediction_and_explanation}

    except Exception as e:
        # If any error occurred, stop the spark session and raise the exception
        raise HTTPException(status_code=400, detail=str(e))

# serve dashboard (static file dist)
app.include_router(api_router, prefix="/api")

# app.mount("/", StaticFiles(directory="dashboard/dist", html=True), name="static")
app.mount("/dashboard/static", StaticFiles(directory="dashboard/dist", html=True), name="static")


@app.get("/dashboard")
async def serve_index():
    return FileResponse("dashboard/dist/index.html")


# @app.get("/")
# async def serve_index():
#     return FileResponse("dashboard/dist/index.html")


# uvicorn app.main:app --host 127.0.0.1 --port 8080
# npm run dev