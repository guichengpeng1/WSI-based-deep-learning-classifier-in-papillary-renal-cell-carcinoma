#!/usr/bin/env python
# coding: utf-8

import openslide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
import cv2
import skimage
from lxml import etree
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import gc
import time
import sys
import getopt
import pandas as pd

def get_slide(slide_path):
    slide = openslide.OpenSlide(slide_path)
    return slide

def normalize_dynamic_range(image, percentile = 95):
    """
    Normalize the dynamic range of an RGB image to 0~255. If the dynamic ranges of patches 
    from a dataset differ, apply this function before feeding images to VahadaneNormalizer,
    e.g. hema slides.
    :param image: A RGB image in np.ndarray with the shape [..., 3].
    :param percentile: Percentile to get the max value.
    """
    max_rgb = []
    for i in range(3):
        value_max = np.percentile(image[..., i], percentile)
        max_rgb.append(value_max)
    max_rgb = np.array(max_rgb)

    new_image = (np.minimum(image.astype(np.float32) * (255.0 / max_rgb), 255.0)).astype(np.uint8)
    
    return new_image

# filter blank
def filter_blank(image, threshold = 80):
    image_lab = skimage.color.rgb2lab(np.array(image))
    image_mask = np.zeros(image.shape).astype(np.uint8)
    image_mask[np.where(image_lab[:, :, 0] < threshold)] = 1
    image_filter = np.multiply(image, image_mask)
    percent = ((image_filter != np.array([0,0,0])).astype(float).sum(axis=2) != 0).sum() / (image_filter.shape[0]**2)

    return percent


def AnnotationParser(path):
    assert Path(path).exists(), "This annotation file does not exist."
    tree = etree.parse(path)
    annotations = tree.xpath("/ASAP_Annotations/Annotations/Annotation")
    annotation_groups = tree.xpath("/ASAP_Annotations/AnnotationGroups/Group")
    classes = [group.attrib["Name"] for group in annotation_groups]
  
    def read_mask_coord(cls):
        for annotation in annotations:
            if annotation.attrib["PartOfGroup"] == cls:
                contour = []
                for coord in annotation.xpath("Coordinates/Coordinate"):
                    x = np.float(coord.attrib["X"])
                    y = np.float(coord.attrib["Y"])
                    contour.append([round(float(x)),round(float(y))])
                #mask_coords[cls].extend(contour)
                mask_coords[cls].append(contour)
    
    def read_mask_coords(classes):
        for cls in classes:
            read_mask_coord(cls)
        return mask_coords            
    mask_coords = {}
    for cls in classes:
        mask_coords[cls] = []
    mask_coords = read_mask_coords(classes)
    return mask_coords,classes


def Annotation(slide,path,save_path=None,rule=False,save=False):
    
    wsi_width,wsi_height = slide.level_dimensions[0]
    masks = {}
    contours = {}
    mask_coords, classes = AnnotationParser(path)
    
    def base_mask(cls,wsi_height,wsi_width):
        masks[cls] = np.zeros((wsi_height,wsi_width),dtype=np.uint8)
    def base_masks(wsi_height,wsi_width):
        for cls in classes:
            base_mask(cls,wsi_height,wsi_width)
        return masks
    
    def main_masks(classes,mask_coords,masks):
        for cls in classes:
            contours = np.array(mask_coords[cls])
            #contours = mask_coords[cls]
            for contour in contours:
                #print(f"cls:{cls},\ncontour:{contour},\ntype:{type(contour)}")
                masks[cls] = cv2.drawContours(masks[cls],[np.int32(contour)],0,True,thickness=cv2.FILLED)
        return masks
   
    def export_mask(save_path,cls):
        assert Path(save_path).is_dir()
        cv2.imwrite(str(Path(save_path)/"{}.tiff".format(cls)),masks[cls],(cv2.IMWRITE_PXM_BINARY,1))
    def export_masks(save_path):
        for cls in masks.keys():
            export_mask(save_path,cls)
    def exclude_masks(masks,rule,classes):
        #masks_exclude = masks.copy()
        masks_exclude = masks
        for cls in classes:
            for exclude in rule[cls]["excludes"]:
                if exclude in masks:
                    overlap_area = cv2.bitwise_and(masks[cls],masks[exclude])
                    masks_exclude[cls] = cv2.bitwise_xor(masks[cls],overlap_area)
        #masks = masks_exclude
        return masks_exclude
                    
    masks = base_masks(wsi_height,wsi_width)
    masks = main_masks(classes,mask_coords,masks)
    if rule:
        classes = list(set(classes) & set(rule.keys()))
        masks = exclude_masks(masks,rule,classes)
    if save:
        export_masks(save_path)
    if "artifact" not in classes:
        masks["artifact"] = np.zeros((wsi_height,wsi_width),dtype=np.uint8)
    if "mark" not in classes:
        masks["mark"] = np.zeros((wsi_height,wsi_width),dtype=np.uint8)
    return masks 

def show_thumb_mask(mask,size=512):
   
    height, width = mask.shape
    scale = max(size / height, size / width)
    mask_resized = cv2.resize(mask, dsize=None, fx=scale, fy=scale)
    mask_scaled = mask_resized * 255
    plt.imshow(mask_scaled)
    return mask_scaled

def get_mask_slide(masks):
    tumor_slide = openslide.ImageSlide(Image.fromarray(masks["tumor"]))
    return tumor_slide

def get_tiles(slide,tumor_slide,tile_size=512,overlap=False,limit_bounds=False):
    slide_tiles = DeepZoomGenerator(slide,tile_size,overlap=overlap,limit_bounds=limit_bounds)
    tumor_tiles = DeepZoomGenerator(tumor_slide,tile_size,overlap=overlap,limit_bounds=limit_bounds)
    return slide_tiles,tumor_tiles

def get_tile_masked(slide_tile,tumor_tile): ####version_update: To save tile_masked, use this function
    x = slide_tile.shape
    y = tumor_tile.shape
    if not x == y:
        h = np.min([x[0],y[0]])
        w = np.min([x[1],y[1]])
        tumor_tile = tumor_tile[:h,:w,:]
        slide_tile = slide_tile[:h,:w,:]
    tile_masked = np.multiply(slide_tile,tumor_tile)
    percent = np.mean(tumor_tile)
    tile_masked[np.all(tile_masked==0)]=255
    return tile_masked,percent

def extract_patches(levels,scales):
    
    for i,level in enumerate(levels):
        
        print(f'processing ---level {scales[i]}')
        print(tile_path)
        tiledir = Path(tile_path)/str(scales[i])
        if not Path(tiledir).exists():
            os.makedirs(tiledir)
        assert slide_tiles.level_tiles[level] == tumor_tiles.level_tiles[level]
        cols,rows = slide_tiles.level_tiles[level]
        for row in range(rows):
            for col in range(cols):
                tilename = os.path.join(tiledir,'%d_%d.%s'%(col,row,"tiff"))
               
                if not Path(tilename).exists():
                    slide_tile = np.array(slide_tiles.get_tile(level,(col,row)))
                    tumor_tile = np.array(tumor_tiles.get_tile(level,(col,row)))
                    tile_masked,percent_2 = get_tile_masked(slide_tile,tumor_tile) # percent of annotated area 
                    percent_1 = filter_blank(tile_masked) # percent of tissue area
                    if all((percent_1 >= 0.75,percent_2 >= 0.75)):                      
                        Image.fromarray(np.uint8(tile_masked)).save(tilename)
                       
                    else:
                        pass
        print("Done!")
    print("All levels processed!!")
    

INDEX= 0
n = 5
argv = sys.argv[1:]
try:
    opts,args = getopt.getopt(argv,"s:n:i:x:")
except:
    print("Error")
for opt,arg in opts:
    if opt in ['-n']:
        n = int(arg)
    elif opt in ['-s']:
        subset = arg
    elif opt in ['-i']:
        i = int(arg)
    elif opt in ['-x']:
        INDEX = int(arg)


classes = ["nonrecurrence","recurrence"]
OVERLAP =0
LIMIT = False
rule = {"stroma":{"excludes":["blood","artifact","mark"]}}
scales = ['5X','10X','20X','40X']
df = pd.read_csv(None",encoding="GB2312") # df containing svs_path and labels
svs_paths = None
svs_labels = None

TILE_SIZE = None
patch_path = None

len(svs_paths)

number = len(svs_paths)

if n*i < number:
    svs_paths = svs_paths[n*(i-1):n*i]
    labels = svs_labels[n*(i-1):n*i]
if n*i >= number:
    svs_paths = svs_paths[n*(i-1):]
    labels = svs_labels[n*(i-1):]


extracted_case = []
un_extracted_case = []
for i,svs in enumerate(svs_paths):
    start = time.time()
    totol_num = len(svs_paths)
    print(f"processing  {i+1}/{totol_num}:------{svs}")
    label = labels[i]
    xml_path = str(Path(svs).with_suffix(".xml"))
    center_name = Path(svs).parent.name
   
    case_name = Path(svs).stem
    
    tile_path = Path(patch_path)/f"{center_name}_{TILE_SIZE}"/classes[label]/case_name
    slide = get_slide(str(svs))
    try:
        masks = Annotation(slide,path=str(xml_path))
        print(f"masks groups includes :{list(masks.keys())}")
        tumor_slide = get_mask_slide(masks) 
        slide_tiles,tumor_tiles = get_tiles(slide,tumor_slide,tile_size=TILE_SIZE,overlap=OVERLAP,limit_bounds=LIMIT)
        del slide
        del masks
        del tumor_slide
        gc.collect()
        level_count = slide_tiles.level_count
        
        levels=[level_count-4,level_count-3,level_count-2,level_count-1]

        try:
            extract_patches(levels,scales)
            extracted_case.append(svs)
        except Exception as e:
            un_extracted_case.append(svs)
            print("something is wrong when extracting")
            print("ERROR!",e)
            continue
    except Exception as e:
        print("something is wrong when parsing")
        print("ERROR!",e)
        continue
    end = time.time()
    print(f"Time consumed : {(end-start)/60} min")
    print(f"******{len(un_extracted_case)}/{len(svs_paths)} remain unextract******")


