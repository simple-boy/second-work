import os
import cv2
import json
import shutil
import random
import pandas as pd
import json


# 计算任意多边形的面积，顶点按照顺时针或者逆时针方向排列
def compute_polygon_area(points):
    point_num = len(points)
    if(point_num < 3): return 0.0
    s = points[0][1] * (points[point_num-1][0] - points[1][0])
    #for i in range(point_num): # (int i = 1 i < point_num ++i):
    for i in range(1, point_num): # 有小伙伴发现一个bug，这里做了修改，但是没有测试，需要使用的亲请测试下，以免结果不正确。
        s += points[i][1] * (points[i-1][0] - points[(i+1)%point_num][0])
    return abs(s/2.0)


def main():
    for phase in ['train', 'val']:
        path = 'coco/' + phase
        if not os.path.exists(path):
            os.makedirs(path)
        dataset = {}
        dataset['info'] = {"description": "COCO Dataset"}
        dataset['categories'] = []
        dataset['images'] = []
        dataset['annotations'] = []
        print(phase)
        f = open(os.path.join(phase + '/', 'via_region_data.json'), 'r')
        content = f.read()
        a = json.loads(content)
        f.close()
        index = 1
        cate_id = 1
        id = 1
        for i in a:
            segmentation = []
            image = cv2.imread(os.path.join(phase + '/', a[i]['filename']))
            dataset['images'].append({
                'file_name': a[i]['filename'],
                'id': index,
                'width': image.shape[1],
                'height': image.shape[0],
            })
            shutil.move(phase + '/' + a[i]['filename'], 'coco/' + phase)
            index = index + 1
            for n in a[i]['regions']:
                area = []
                if dataset['categories'] == []:
                    dataset['categories'].append(
                        {'id': cate_id, 'name': a[i]['regions'][n]['shape_attributes']['name'], 'supercategory': 'Ball'})
                    cate_id = cate_id + 1
                elif a[i]['regions'][n]['shape_attributes']['name'] != 'polygon':
                    dataset['categories'].append(
                        {'id': cate_id, 'name': a[i]['regions'][n]['shape_attributes']['name'], 'supercategory': 'Ball'})
                    cate_id = cate_id + 1

                for x, y in zip(a[i]['regions'][n]['shape_attributes']['all_points_x'], a[i]['regions'][n]['shape_attributes']['all_points_y']):
                    segmentation.append(x)
                    segmentation.append(y)
                    area.append([x, y])
                x_max = max(a[i]['regions'][n]['shape_attributes']['all_points_x'])
                x_min = min(a[i]['regions'][n]['shape_attributes']['all_points_x'])
                y_max = max(a[i]['regions'][n]['shape_attributes']['all_points_y'])
                y_min = min(a[i]['regions'][n]['shape_attributes']['all_points_y'])
                area_sum = compute_polygon_area(area)
                dataset['annotations'].append({
                    'area': area_sum,
                    'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
                    'category_id': 0,
                    'id': id,
                    'image_id': index,
                    'iscrowd': 0,
                    'segmentation': [segmentation]
                })
                id = id + 1
        folder = 'coco/annotations'
        if not os.path.exists(folder):
            os.makedirs(folder)
        json_name = os.path.join('', 'coco/annotations/instances_{}.json'.format(phase))
        with open(json_name, 'w') as f:
            json.dump(dataset, f, ensure_ascii=False)
        print('Success!')

if __name__ == '__main__':
    main()

