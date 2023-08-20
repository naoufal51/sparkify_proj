# Sparkify Churn Detection Project

Sparkify Churn Detection is a project designed to predict user churn on the fictitious Sparkify digital music service platform. The model would be crucial for the stakeholders at Sparkify to devise appropriate user retention strategies. The project features an interactive web dashboard and provides the capability to train the churn detection model using predefined configurations.

https://github.com/naoufal51/disaster_response/assets/15954923/9297c5d9-cb7f-46af-88f0-fa7d5172f1cd


## Getting Started

### Web Application Setup:
Follow the steps below to set up and run the web dashboard:

1. Export the Weights & Biases (W&B) API key:
```sh
export WANDB_API_KEY=YOUR_WANDB_API_KEY
```

2. Set up the dashboard:
```sh
cd app/dashboard
echo "VITE_API_URL=http://127.0.0.1:8080" > .env
npm install
npm run build
cd ..
```
3. Start the application:
```sh
uvicorn main:app --host 127.0.0.1 --port 8080
```



## Model Training:
For a detailed walkthrough of model training, check the dedicated [Kaggle notebook](https://www.kaggle.com/naoufal51/sparkify-project-run)

Alternatively, set up the training environment using the following steps:

1. Clone the repository and setup the env:
``` sh 
git clone https://github.com/naoufal51/sparkify_proj.git
pip install -r sparkify_proj/requirements.txt
```
2. Update the config.toml file with your API keys and custom conf. For instance:
```sh
[model_experiment]
max_evals = 20

[wandb]
key = "your_wandb_key"
project = "sparkify_naoufal"

[prefect]
key = "your_prefect_key"

[raw_data_s3]
file_name = "sparkify/sparkify_event_data.json"

```
3. Train the model:
``` sh 
cd sparkify_proj/
python main.py

```

## Acknowledgements
- [Udacity](https://www.udacity.com/) for providing the Sparkify dataset

