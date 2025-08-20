# Crimson X

## Description

This project is a web application hosted on Render, providing a front-end interface to make predictions using a Convolutional Neural Network (CNN) model.

## Table of Contents

1. [Deployment](#deployment)
2. [Using the Application](#using-the-application)
3. [License](#license)

## Deployment

The front-end application is deployed on Render, and all deployments happen automatically whenever changes are pushed to the repository:

1. Push code to the repository.
2. Render will automatically build and deploy the front-end application.
3. The web app will be live at a Render-generated URL, where users can interact with the model.

Render will handle auto-scaling, so the application can efficiently manage traffic.

## Using the Application

Once deployed, the application can be accessed via the public URL provided by Render. Users can:

1. Upload an image (or provide other input data depending on the model).
2. The web app will send the data to the backend model endpoint for prediction.
3. Results will be returned and displayed to the user.

## License

The model was created using https://www.kaggle.com/datasets/rajumavinmar/finger-print-based-blood-group-dataset , which has no license.
