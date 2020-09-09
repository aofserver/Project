# Project
make by sarawut nacwijit

# Require
```
$ pip3 install -r requirements.txt
```
if can't install dbilb try 
```
$ python -m pip install https://files.pythonhosted.org/packages/0e/ce/f8a3cff33ac03a8219768f0694c5d703c8e037e6aba2e865f9bae22ed63c/dlib-19.8.1-cp36-cp36m-win_amd64.whl#sha256=794994fa2c54e7776659fddb148363a5556468a6d5d46be8dad311722d54bfcf
```
or follow metthod install dlib in Video [![YouTube](https://s.ytimg.com/yts/img/favicon-vfl8qSV2F.ico)](https://www.youtube.com/watch?v=HqjcqpCNiZg)


# Setup data
save image person in folder `/facedata/` and rename picture file `name_number`
![](https://github.com/aofserver/Project/blob/master/Face%20Recognition%20Dlib/etc/1.png)

# Recognition
run `facetrainer.py` for train data set when end process you get file `trainset.pk` is file trainer
```
$ python facetrainer.py
```
# Detection
run `facedetection.py` for use 'trainset.pk' to recognition input image
```
$ python facedetection.py
```
press `esc` to exit

# Result
![](https://github.com/aofserver/Project/blob/master/Face%20Recognition%20Dlib/etc/2.png)


# Reference
[YouTube](https://www.youtube.com/watch?v=gT3uELrVpOs&t=2s)



