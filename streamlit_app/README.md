# Streamlit USL App (Hosted Setup)

This package is ready for Streamlit Cloud deployment with the same UI as your current app and compressed distilled model as the default inference model.

## Folder contents

- app.py
- model.pkl
- requirements.txt

## Input modes in app

- Upload Video (best fidelity for this model)
- Upload Image
- Use Webcam Snapshot

## Streamlit Cloud deployment steps

1. Push this `streamlit_app` folder to GitHub.
2. Open Streamlit Cloud and sign in with GitHub.
3. Click New app.
4. Select your repository and branch.
5. Set Main file path to `streamlit_app/app.py`.
6. Click Deploy.

## Notes

- Default behavior now loads `models_dataset1/csv_models/artifacts/distilled_student.joblib` first (compressed model).
- If that file is unavailable, app falls back to `streamlit_app/model.pkl`, then `models_dataset1/csv_models/artifacts/best_model.joblib`.
- You can still override with `MODEL_ARTIFACT_PATH` in Streamlit Cloud secrets/environment.
- To deploy compressed model with safety fallback, set:
  - `MODEL_ARTIFACT_PATH=models_dataset1/csv_models/artifacts/distilled_student.joblib`
  - `FALLBACK_MODEL_ARTIFACT_PATH=models_dataset1/csv_models/artifacts/best_model.joblib`
  - `ENABLE_FALLBACK=true`
  - `FALLBACK_CONFIDENCE_THRESHOLD=0.75`

If fallback is enabled and the compressed model confidence is below the threshold, the app serves the teacher prediction for that request while keeping the same interface.
