# Streamlit USL App (Hosted Setup)

This package is ready for Streamlit Cloud deployment with the 338-feature deployed SVM-RBF model.

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

- `model.pkl` is already expected in the same folder as `app.py`.
- If you want a custom path, set `MODEL_ARTIFACT_PATH` in Streamlit Cloud secrets/environment.
