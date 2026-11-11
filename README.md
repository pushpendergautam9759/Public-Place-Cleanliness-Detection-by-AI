# Public Place Cleanliness Classification Using CNN

This project classifies a public-place image into three categories:
- Clean
- Moderate
- Dirty

## Tools
Python, TensorFlow, Keras, CNN and Streamlit.

## Dataset folders
Keep images in this structure:

    dataset/
        Clean/
        Moderate/
        Dirty/

Use different real images in each folder. Do not use duplicate copies.

## Run the project

    pip install -r requirements.txt
    python train.py
    streamlit run app.py

The trained model is saved in `models/cleanliness_model.keras`.

Note: A very small dataset is useful only for checking the code. It is not enough to train a reliable classifier. Add more varied images to all three classes before training and evaluating the model.
