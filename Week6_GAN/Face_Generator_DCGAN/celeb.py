"""
    MsCeleb file to individual image files
    
"""

import os
import base64
import struct

"""
File format: text files, each line is an image record containing 7 columns, delimited by TAB.
Column1: Freebase MID
Column2: ImageSearchRank
Column3: ImageURL
Column4: PageURL
Column5: FaceID
Column6: FaceRectangle_Base64Encoded (four floats, relative coordinates of UpperLeft and BottomRight corner)
Column7: FaceData_Base64Encoded d Data](https://msceleb.blob.core.windows.net/ms-celeb-v1-samples/MsCelebV1-Faces-Aligned.Samples.zip): 14 entities, 32MBceleb.blob.core.windows.net/ms-celeb-v1-samples/MsCelebV1-Faces-Aligned.Samples.zip): 14 entities, 32MB
"""

file_in = open('./MS-Celeb-FaceImageCroppedWithAlignment.tsv', 'r', encoding='utf8')
base_path = './MsCeleb'
bbox_file = open(base_path + '/bboxes.txt', 'w', encoding='utf8')
while True:
    line = file_in.readline()
    if line:
        data_info = line.split('\t')
        filename = data_info[0] + "/" + data_info[1] + "_" + data_info[4] + ".jpg"
        bbox = struct.unpack('ffff', base64.b64decode(data_info[5]))
        bbox_file.write(filename + " " + (" ".join(str(bbox_value) for bbox_value in bbox)) + "\n")

        img_data = base64.b64decode(data_info[6])
        output_file_path = base_path + '/' + filename
        output_path = os.path.dirname(output_file_path)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        img_file = open(output_file_path, 'wb')
        img_file.write(img_data)
        img_file.close()
    else:
        break

bbox_file.close()
file_in.close()
