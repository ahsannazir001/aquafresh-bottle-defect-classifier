# Maintenance Guide — AquaFresh Bottle Defect Classifier

## 1. Project Components

- `api/main.py` — FastAPI endpoints and model inference.
- `models/best_tuned_resnet18.pt` — selected trained model checkpoint.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — container build and startup configuration.
- `src/` — training and evaluation scripts.
- `tests/` — automated API tests.
- `reports/` — training, evaluation, and error-analysis outputs.

## 2. Run Locally

Install the dependencies in a suitable Python environment:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` to inspect and test the API.

## 3. Run with Docker

Build the image:

```bash
docker build -t aquafresh-api:task4 .
```

Run the container:

```bash
docker run --rm -p 8000:8000 --name aquafresh-api-task4 aquafresh-api:task4
```

The API documentation should then be available at `http://localhost:8000/docs`.

## 4. Verify the Deployment

- Open the live `/health` endpoint and confirm the healthy status.
- Open `/docs` and submit an image to `POST /predict`.
- Confirm the endpoint returns a valid JSON prediction.
- Check Render deployment logs if the service fails to start or respond.
- Remember that a free Render service may take longer to respond after inactivity.

## 5. Model Update Procedure

1. Collect new bottle images with reliable ground-truth labels.
2. Keep images from the same original source in only one dataset split.
3. Retrain candidate models and record their configurations.
4. Evaluate the candidate on a held-out test set.
5. Review confusion matrices and misclassified examples.
6. Test the candidate locally through the API and Docker.
7. Replace the deployed checkpoint only after validating the new model.
8. Redeploy and repeat the live health and prediction tests.

## 6. Troubleshooting

### API does not start

Check the terminal output, installed dependencies, Python version, and module path.

### Model checkpoint cannot be loaded

Confirm that `models/best_tuned_resnet18.pt` exists and matches the checkpoint format expected by `api/main.py`.

### Prediction returns HTTP 422

Confirm that the request includes an image in the expected multipart form field named `file`.

### Render deployment fails

Review the build and runtime logs. Confirm that the Dockerfile starts Uvicorn and listens on the port provided by Render.

### Predictions appear incorrect

Review the image quality, class labels, preprocessing, and saved checkpoint. Compare predictions with verified labels before drawing conclusions.

## 7. Security and Operational Notes

- Never commit API secrets, credentials, or private environment files.
- Validate uploaded image files and handle invalid inputs safely.
- Keep dependencies and deployment configuration under version control.
- Document changes to model weights, preprocessing, and API behaviour.
