Embedding a Disasters Image Classification Model into Restful APIs
==============================
This is a supplementary submission of **final paper** for the **CIS726** course.

It contains the code necessary to host a local restful API that utilizes CNN model to predict
incoming request values.

The model has the following hyperparameters:

* 50 Epochs
* Learning Rate of 0.001
* Adam optimizer
* ResNet50 has been selected as the pretrained model

The expected returned values are:

* Lime image annotation in `Base64` format
* Grad-CAM image annotation in `Base64` format
* Grad-CAM++ image annotation in `Base64` format
* The predicted label

> The weights of the model has been imported, rather than the whole architecture and configuration.

> Due to hosting limitations, Lime `num_samples` attribute has been reduced from 1000 to 10 only.

Such hyperparameters returned the best results.

Getting Started
------------
Clone the project from GitHub

`$ git clone https://github.com/tariqshaban/disaster-classification-with-xai-server.git`

It is encouraged to refer to [FastAPI](https://fastapi.tiangolo.com/tutorial/) documentation.

You may need to configure the Python interpreter (depending on the used IDE).

You may encounter problem concerning CORS policy when the server is improperly hosted.

No further configuration is required.

Usage
------------
Execute the `uvicorn main:app` command in the console, ensure that the port 8000 is not occupied, if need be, add
the `--port *YOUR_PORT*` flag.

You can also issue direct API request using [Heroku](https://www.heroku.com),
[example](https://disaster-classification-server.herokuapp.com/classify_image); [Postman](https://www.postman.com/)
should be used for the image to be uploaded.

--------