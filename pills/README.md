# Business-Card-Detector
Detect if an image is a business card or not!


### Steps:
1. Clone the repo
```
git clone https://github.com/kumar-shridhar/Business-Card-Detector.git
```

2. Create a conda env
```
conda create -n businesscarddetect python=3.6
```

3. Activate conda env
```
conda activate businesscarddetect
```

4. Install dependencies
```
pip install -r requirements.txt
```

5. Install torch-cpu
```
conda install pytorch-cpu torchvision-cpu -c pytorch
```

6. Run the app
```
python app.py
```

7. Make a request
```
curl -X POST -F image=@path-to-image.jpg 'http://localhost:5000/predict'
```

