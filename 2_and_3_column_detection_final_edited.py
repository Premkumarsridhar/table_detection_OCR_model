from PIL import Image, ImageDraw
import json
import cv2
import os
from utility import createFolder

list_id = ['53bf60d2-177d-4107-b573-f9e7754ed1ca', 'fa051534-3c49-41be-8041-5c617ce22fa9', '94635cac-f555-4e5a-90fa-49700d955a76', 'd2d8cd5b-83cf-4620-a695-dd7df30dcef5', '473730a9-3e99-4fb1-87c3-0684e315dc88', '31d830da-a9ef-4b20-b8b6-7f29c9287b7e', '6c6e5599-aa0a-491a-a874-4415b1168010', '51ec4f4a-51d8-4032-8968-06bacd853e78', '2a66748a-e9cd-4acc-91e6-2dad0877a89e', '5970cda9-08ab-44e8-9344-e8fc56aaf614', '25655d06-3651-45ef-88c4-4d6a45cede01', '942e7789-edc9-4198-b735-455b7937a50c', 'bc2307a3-cdc6-46b7-a8b0-41a430c56697', '048f2148-baa0-4153-8ddd-687c412b6fa6', 'de7f2a18-8451-425b-a1ad-b51f3b881b5a', '97caea02-de1a-477b-8413-7667df5ec97a', '5a2ea89c-3c9f-4db0-8ce7-f21458e111d5', 'c9ef1a8f-9c45-4ae2-bff9-f62f1b235aa4', '3d4f85e8-559a-439c-b0ac-2dc8c53f3221', 'feaa7b62-127c-4995-a051-69f84f01af73', '64bbec4b-2a46-4915-9f56-5c1e4817aab5', '6351deca-36cc-4296-9370-1f3edeaa8bca', '608a7b44-c711-453b-9b43-c219f37a603c', '1a610fc9-f6d3-4af4-aa0a-4aed8105ecd3', '72bdb4fc-08af-4023-81d0-727519e82243', '7f82ad70-079f-4439-a685-c154ad0df62c', '5f86619a-4985-4407-8b1f-82091058e3a5', '5f05943b-f84f-45fd-808e-d6af9e95f4b5', '927e1e10-11a6-457b-9c40-5019d98e5718', '0c694038-2fc6-4ce8-809a-80c489156aea', 'd03e73cb-3c60-45be-95c3-a11e5f158b82', 'be4a5a28-31a2-46fe-a436-1056dd588418', '579f689d-e7e9-4be8-8df1-92d6d0535585', '27752030-6d08-4f8a-bdb9-3424c510d092']

def detectingColumn(child_id_list, blocks, height, width, path):

  child_ids = child_id_list

  data_response = blocks

  #-----------------------------------------------
  height = int(height)
  width = int(width)

  tabular_word_blocks = []

  for i in data_response['annotations']:
    if i['BlockType'] == 'WORD' and i['Id'] in child_ids:
      tabular_word_blocks.append(i) 

  save_path = os.path.join(path, "ceatedImageForDetectingWordsInColumn")
  try:
    if not os.path.exists(save_path):
      os.mkdir(save_path)
  except:
    None

  image = Image.new('RGB', (width, height), (255, 255, 255))
  draw = ImageDraw.Draw(image)
  for k in tabular_word_blocks:
    x1 = k['Geometry']['BoundingBox']['Left'] * width
    y1 = k['Geometry']['BoundingBox']['Top'] * height
    x4 = x1 + k['Geometry']['BoundingBox']['Width'] * width
    y4 = y1 + k['Geometry']['BoundingBox']['Height'] * height
    shape = [(x1, y1), (x4, y4)]
    draw.rectangle(shape, fill = "#000000", outline = "black")
  pathToSaveImg = save_path + "/created_image.png"
  image.save(pathToSaveImg)

  # read input image
  img = cv2.imread(pathToSaveImg)

  # define border color
  lower = (0, 80, 110)
  upper = (0, 120, 150)

  # threshold on border color
  mask = cv2.inRange(img, lower, upper)

  # dilate threshold
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 100))
  mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

  # recolor border to white
  img[mask==255] = (255,255,255)

  # convert img to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # otsu threshold
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU )[1] 

  # apply morphology open
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,200))
  morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
  morph = 255 - morph

  # find contours and bounding boxes
  bboxes = []
  bboxes_img = img.copy()
  contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1]
  for cntr in contours:
      x,y,w,h = cv2.boundingRect(cntr)
      cv2.rectangle(bboxes_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
      bboxes.append((x,y,w,h))

  # get largest width of bboxes
  maxwidth = max(bboxes)[2]

  # sort bboxes on x coordinate
  def takeFirst(elem):
      return elem[0]

  bboxes.sort(key=takeFirst)

  datalists = dict()
  if len(bboxes) == 3:
    datalists[1] = [bboxes[1]]
    datalists[2] = [bboxes[2]]
  elif len(bboxes) > 3:
    templist = bboxes[1:]
    temp_col_ind = []
    for i in range(len(templist)):
      for j in range(len(templist[i])):
        if j == 2:
          if templist[i][j] >=100 and templist[i][j + 1] >= 200:
            temp_col_ind.append(i)
            break
    if len(temp_col_ind) == 2:
      datalists[1] = [templist[temp_col_ind[0]]]
      datalists[2] = [templist[temp_col_ind[1]]]
    elif len(temp_col_ind) == 3:
      datalists[1] = [templist[temp_col_ind[0]]]
      datalists[2] = [templist[temp_col_ind[1]]]
      datalists[3] = [templist[temp_col_ind[2]]]
    elif len(temp_col_ind) > 3:
      width_array = []
      for i in range(len(bboxes)):
        for j in range(len(bboxes[i])):
          if j == 2:
            width_array.append(bboxes[i][j])
      # -----------------------------------------
      width_mx = max(width_array[0], width_array[1])
      width_secondmax = min(width_array[0], width_array[1])
      n = len(width_array)
      for i in range(2,n):
        if width_array[i] > width_mx:
          width_secondmax = width_mx
          width_mx = width_array[i]
        elif width_array[i] > width_secondmax and \
          width_mx != width_array[i]:
          width_secondmax = width_array[i]
        elif width_mx == width_secondmax and \
          width_secondmax != width_array[i]:
          width_secondmax = width_array[i]
      # -----------------------------------------
      ind_max_and_smax = []
      for i in range(len(width_array)):
        if width_array[i] == width_mx or width_array[i] == width_secondmax:
          ind_max_and_smax.append(i)
      ind_max_and_smax = sorted(ind_max_and_smax)
      # -----------------------------------------
      datalists[1] = [bboxes[ind_max_and_smax[0]]]
      datalists[2] = [bboxes[ind_max_and_smax[1]]]
    
  elif len(bboxes) == 2:
    datalists[1] = [bboxes[0]]
    datalists[2] = [bboxes[1]] 

  columns = []
  for i in datalists.values():
    for j in i:
      columns.append(list(j))

  avg_column_coordinates_ac = []
  if len(columns) == 2:
    x1_1 = columns[0][0]
    y1_1 = columns[0][1]
    x2_1 = x1_1 + columns[0][2]
    y2_1 = y1_1
    x3_1 = x1_1
    y3_1 = y1_1 + columns[0][3] 
    x4_1 = x1_1 + columns[0][2]
    y4_1 = y3_1
    # -----------------------------------
    x1_2 = columns[1][0]
    y1_2 = columns[1][1]
    x2_2 = x1_2 + columns[1][2]
    y2_2 = y1_2
    x3_2 = x1_2
    y3_2 = y1_2 + columns[1][3] 
    x4_2 = x1_2 + columns[1][2]
    y4_2 = y3_2
    # -----------------------------------
    x_avg1 = (x2_1 + x1_2) / 2  
    y_avg1 = (y2_1 + y1_2) / 2
    # -----------------------------------
    avg_column_coordinates_ac.append(x_avg1/width)
  elif len(columns) == 3:
    x1_1 = columns[0][0]
    y1_1 = columns[0][1]
    x2_1 = x1_1 + columns[0][2]
    y2_1 = y1_1
    x3_1 = x1_1
    y3_1 = y1_1 + columns[0][3] 
    x4_1 = x1_1 + columns[0][2]
    y4_1 = y3_1
    # -----------------------------------
    x1_2 = columns[1][0]
    y1_2 = columns[1][1]
    x2_2 = x1_2 + columns[1][2]
    y2_2 = y1_2
    x3_2 = x1_2
    y3_2 = y1_2 + columns[1][3] 
    x4_2 = x1_2 + columns[1][2]
    y4_2 = y3_2
    # -----------------------------------
    x1_3 = columns[2][0]
    y1_3 = columns[2][1]
    x2_3 = x1_3 + columns[2][2]
    y2_3 = y1_3
    x3_3 = x1_3
    y3_3 = y1_3 + columns[2][3] 
    x4_3 = x1_3 + columns[2][2]
    y4_3 = y3_3
    # -----------------------------------
    x_avg1 = (x2_1 + x1_2) / 2
    x_avg2 = (x2_2 + x1_3) / 2
    y_avg1 = (y2_1 + y1_2) / 2
    y_avg2 = (y2_2 + y1_3) / 2
    # -----------------------------------
    avg_column_coordinates_ac.append(x_avg1/width)
    avg_column_coordinates_ac.append(x_avg2/width)

  segregated_column_data = dict()
  if len(avg_column_coordinates_ac) > 1:
    for item in range(len(avg_column_coordinates_ac)):
      segregated_column_data[item] = ''
    segregated_column_data[len(avg_column_coordinates_ac)] = ''
  elif len(avg_column_coordinates_ac) == 1:
    segregated_column_data[0] = ''
    segregated_column_data[1] = ''

  if len(segregated_column_data) == 2:
    for i in range(len(tabular_word_blocks)):
      count = 0
      for j in range(len(tabular_word_blocks[i]['Geometry']['Polygon'])):
        if tabular_word_blocks[i]['Geometry']['Polygon'][j]['X'] < avg_column_coordinates_ac[0]:
          count += 1
      if count == 4:
        segregated_column_data[0] += " " + data_response['annotations'][i]['Text'] 
      else:
        segregated_column_data[1] += " " + data_response['annotations'][i]['Text']
  elif len(segregated_column_data) == 3:
    for i in range(len(tabular_word_blocks)):
      fcount = 0
      scount = 0
      tcount = 0
      for j in range(len(tabular_word_blocks[i]['Geometry']['Polygon'])):
        if tabular_word_blocks[i]['Geometry']['Polygon'][j]['X'] < avg_column_coordinates_ac[0]:
          fcount += 1 
        elif tabular_word_blocks[i]['Geometry']['Polygon'][j]['X'] > avg_column_coordinates_ac[0] and tabular_word_blocks[i]['Geometry']['Polygon'][j]['X'] < avg_column_coordinates_ac[1]:
          scount += 1
        elif tabular_word_blocks[i]['Geometry']['Polygon'][j]['X'] > avg_column_coordinates_ac[1]:
          tcount += 1
      if fcount == 4:
        segregated_column_data[0] += " " + tabular_word_blocks[i]['Text'] 
      elif scount == 4:
        segregated_column_data[1] += " " + tabular_word_blocks[i]['Text']
      elif tcount == 4:
        segregated_column_data[2] += " " + tabular_word_blocks[i]['Text']

  count_empty = 0
  for i in range(len(segregated_column_data)):
    if segregated_column_data[i] == '':
      count_empty += 1   
      print("something wrong")
      break
    else:
      print(segregated_column_data[i])

    