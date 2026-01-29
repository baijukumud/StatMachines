!pip install -U ultralytics  # torch.reqired

!yolo detect predict imgsz=h,w source='<link/path-to-mage>' model='<model-path>'
!ls /content/runs/detect/predict2/  # Find complete file name

from IPython.display import Image
Image('<imag-path>')
