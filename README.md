# Automated object counting on riverbanks

This version uses the yolov5 model as detection (multiclass).

## Installation

## Dev Branch - Installation


Follow these steps in that order exactly:
```shell
git clone https://github.com/surfriderfoundationeurope/surfnet.git <folder-for-surfnet> -b release
conda create -n surfnet pytorch torchvision -c pytorch
conda activate surfnet
cd <folder-for-surfnet>
pip install -r requirements.txt
```
## Downloading pretrained models

You can download MobileNetV3 model with the following script:
```shell
cd models
sh download_pretrained_base.sh
cd ..
```
The file will be downloaded into [models](models).

## Validation videos

If you want to download the 3 test videos on the 3 portions of the Auterrive riverbank, run:

```
cd data
sh download_validation_videos.sh
cd ..
```

This will download the 3 videos in distinct folders of [data/validation_videos](data/validation_videos).

## Run

If you have custom videos, add them to [data/validation_videos](data/validation_videos) in a new subfolder. Then:

### Development
Setting up the server and testing: from surfnet/ directory, you may run a local flask developement server with the following command:

```shell
export FLASK_APP=src/serving/app.py
flask run
```

### Production
Setting up the server and testing: from surfnet/ directory, you may run a local wsgi gunicorn production server with the following command:

```shell
PYTHONPATH=./src gunicorn -w 5 --threads 2 --bind 0.0.0.0:8001 --chdir ./src/serving/ wsgi:app
```

### Test surfnet API
Then, in order to test your local dev server, you may run:
```shell
curl -X POST http://127.0.0.1:5000/ -F 'file=@/path/to/video.mp4' # flask
```
Change port 5000 to 8001 to test on gunicorn or 8000 to test with Docker and gunicorn.

### Docker
You can build and run the surfnet AI API within a Docker container.

Docker Build:
```shell
docker build -t surfnet/surfnet:latest .
```

Docker Run:
```shell
docker run --env PYTHONPATH=/src -p 8000:8000 --name surfnetapi surfnet/surfnet:latest
```

### Makefile
You can use the makefile for convenience purpose to launch the surfnet API:
```shell
make surfnet-dev-local # with flask
make surfnet-prod-local # with gunicorn
make surfnet-prod-build-docker # docker build
make surfnet-prod-run-docker # docker run
```

## Configuration

`src/serving/inference.py` contains a Configuration dictionary that you may change:
- `skip_frames` : `3` number of frames to skip. Increase to make the process faster and less accurate.
- `kappa`: `7` the moving average window. `1` prevents the average, avoid `2` which is ill-defined.
- `tau`: `4` the number of consecutive observations necessary to keep a track. If you increase `skip_frames`, you should lower `tau`.


## Datasets and Training

Consider other branches for that!
