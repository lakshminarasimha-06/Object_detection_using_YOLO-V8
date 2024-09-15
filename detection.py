import os
import xml.etree.ElementTree as et
from glob import glob
import pandas as pd
import cv2
def object_info(filename):
  tree=et.parse(filename)
  root=tree.getroot()
  xml_data=[]
  image_title=root.find("filename").text
  width,height=int(root.find('size').find('width').text),int(root.find('size').find('height').text)
  objects=root.findall('object')
  for o in objects:
    name=o.find('name').text
    xmin=int(o.find('bndbox').find('xmin').text)
    xmax=int(o.find('bndbox').find('xmax').text)
    ymin=int(o.find('bndbox').find('ymin').text)
    ymax=int(o.find('bndbox').find('ymax').text)
    xml_data.append([image_title,width,height,name,xmin,xmax,ymin,ymax])
  return xml_data
train_xml_list=glob(os.path.join(train_path,'*.xml'))
test_xml_list=glob(os.path.join(test_path,"*.xml"))
train_data=[]
test_data=[]
for i in map(object_info,train_xml_list):
  train_data.extend(i)
for i in map(object_info,test_xml_list):
  test_data.extend(i)
  train_df=pd.DataFrame(train_data,columns=['filename','width','height','name','xmin','xmax','ymin','ymax'])
  test_df=pd.DataFrame(test_data,columns=['filename','width','height','name','xmin','xmax','ymin','ymax'])
for df in [train_df,test_df]:
  df['center_x']=((df['xmin']+df['xmax'])/2)/df['width']
  df['center_y']=((df['ymin']+df['ymax'])/2)/df['height']
  df['w']=(df['xmax']-df['xmin'])/df['width']
  df['h']=(df['ymax']-df['ymin'])/df['height']
  labels={'cat':0,'dog':1}
  df['id']=df['name'].map(labels)
def save_labels(df,folderpath):
  for filename,group in df.groupby('filename'):
    text_filename=os.path.join(folderpath,os.path.splitext(filename)[0]+'.txt')
    group[['id','center_x','center_y','w','h']].to_csv(text_filename,sep=' ',index=False,header=False)
save_labels(train_df,train_path)
save_labels(test_df,test_path)
from ultralytics import YOLO
model=YOLO('yolov8s.yaml')
model.train(data= 'data.yaml',epochs=30,batch=8,name='Model')
image_path='/content/drive/MyDrive/pets_detection/download (3).jpeg'
image=cv2.imread(image_path)
result=model.predict(source=image,save=True,save_txt=True)
def draw_bounding_box_with_label(image, label, bbox, color=(200, 10, 0), font=cv2.FONT_HERSHEY_SIMPLEX):
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

    # Define font parameters
    font_scale = 0.2  # Adjust the font scale as necessary
    font_thickness = 1 # Adjust thickness for better readability
    text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Position the text within the bounding box
    text_x = xmin
    text_y = max(ymin - 2, text_size[1] + 2)  # Adjust text position

    # Draw text background for better visibility
    cv2.rectangle(image, (text_x, text_y - text_size[1] - 5),
                  (text_x + text_size[0], text_y + 5), color, -1)

    # Draw the text with anti-aliasing
    cv2.putText(image, label, (text_x, text_y), font, font_scale,
                (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)
for result in result:
    for box in result.boxes:
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])  # Bounding box coordinates
        label = result.names[int(box.cls)]  # Object label (e.g., 'cat', 'dog')
        confidence = box.conf[0]  # Confidence score

        # Label with confidence score
        label_with_conf = f"{label} {confidence:.2f}"

        # Draw bounding box and label on the image
        draw_bounding_box_with_label(image, label_with_conf, (xmin, ymin, xmax, ymax), color=(200, 200, 100))
# Display the final image
from google.colab.patches import cv2_imshow

cv2_imshow(image)
cv2.waitKey(0)
cv2.destroyAllWindows()