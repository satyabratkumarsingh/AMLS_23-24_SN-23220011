# AMLS_23-24_SN-23220011
Assignment

This project contains following 4 python files

 - main.py
 - utils/base_logger.py
 - utils/base_logger.py
 - A/binary_classification.py
 - B/multi_classication.py
 - B/convolutional_neural_network.py

And the requiremnts file
 - requirements.txt

To run the application, first create a python virtaul env
and then install dependecies using following command
  > pip install -r requirements.txt
  This should be run from the project root folder

To run the application do 
    > python main.py
   
The logs for the application can be checked from app.log file

The Binary classification takes around 10 mins to run 
Multi classifcation takes 30-40 mins to run.

The best way to run them is to run them individually 

by commenting one of these
    # binary_classification.train_models()
or 
    #multi_class.train_model()
    # multi_class.test_model()
and then running main.py
