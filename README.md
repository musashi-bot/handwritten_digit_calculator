# Handwritten Math Expression Calculator (Streamlit App)

This is a Streamlit web app that lets users **draw digits and mathematical operators** on a canvas and evaluates the expression using a trained PyTorch CNN model. The model takes in grayscale 56 X 56 images as input . 

## Demo

[Click here to try the app!](https://your-streamlit-cloud-url](https://handwrittendigitcalculator-ntvbdflgmrhdvexh42lcbz.streamlit.app/)

## How It Works

- Three drawable canvases:
  - First digit (0–9)
  - Operator (`+`, `-`, `*`, `/`)
  - Second digit (0–9)
- A CNN model classifies each drawn image
- The expression is evaluated in real-time using Python


