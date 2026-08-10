# DL GenAI Project 23f2000775

The Streamlit app lives in `deployment/app.py`.

Run it directly from `deployment/`:

```powershell
cd deployment
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If you prefer to stay at the repository root, use:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run deployment/app.py
```

For the CNN deployment, place the trained files next to `deployment/app.py` or set these environment variables:

- `CNN_MODEL_PATH` for the saved NumPy checkpoint, for example `cnn_model.npz`
- `CNN_VOCAB_PATH` for the saved vocabulary JSON, for example `cnn_vocab.json`
