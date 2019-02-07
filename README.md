# red-barrel-challenge

### Environment
 - Python 2.7

### Virtual Environment
 - Create virtual environment
 ```sh
 $ virtualenv -p python venv
 ```
 - Activate virtual environment
 ```sh
 $ source ./venv/bin/activate
 ```
 - Install all dependencies
 ```sh
 $ pip install -r requirements.txt
 ```
 - Exit virtual environment
 ```sh
 $ deactivate
 ```
 
### Image Labeling
 - Place the training image folder *2019Proj1_train/* inside the project folder
 ```sh
 $ python labeling_train.py
 ```


### Model Training
 - Open notbook.ipynb in with jupter notebook, and run all cells.  (Training data: *2019Proj1_train/* and *labeled_train/* are required)
 ```sh
 $ jupyter notebook
 ```

### Image Labeling
 - Place the test image folder *Test_Set/* inside the project folder
 ```sh
 $ python main.py
 ```