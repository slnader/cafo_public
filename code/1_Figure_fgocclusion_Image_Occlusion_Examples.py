# ==============================================================================
# Figure 1: Image Occlusion Examples
# ==============================================================================

#Required modules (condatensorflow environment)
import os
import psycopg2
import pandas as pd
import numpy as np
import getpass
from PIL import Image as PILImage
from PIL import ImageDraw
from PIL import ImageOps
from PIL import ImageFont

def main():

    # ==============================================================================
    #0. File paths to image names
    # ==============================================================================

    #Image location
    image_dir = '../data/images/fig1'

    #Image paths
    occluded_poultry = os.path.join(image_dir, 'north-carolina_duplin_2643_287_6_1_0_17_916_13409.jpeg')
    full_poultry = os.path.join(image_dir, 'north-carolina_sampson_19269_287_6_1_0_17_858_13469.jpeg')
    occluded_swine = os.path.join(image_dir, 'north-carolina_jones_4172_287_6_1_0_18_-784_13568.jpeg')
    full_swine = os.path.join(image_dir, 'north-carolina_duplin_18401_287_6_1_0_18_-906_13564.jpeg')

    #Figure directory
    fig_dir = '../figures'

    #PIL fonts
    font = ImageFont.truetype(os.path.join('../data/fonts/DejaVuSans-Bold.ttf'), 16)

    # ==============================================================================
    #1. Build figure
    # ==============================================================================

    print("Creating image occlusion example figure...")

    #Image dims
    col_inc = 299
    offset = 18

    #Create new image template
    new_im = PILImage.new('RGB', ((col_inc*2)+offset, (col_inc*2)+offset), color = (255,255,255))
    draw_image = ImageDraw.Draw(new_im)

    #Column labels
    draw_image.text((col_inc/3,0), 'Occluded', 'black', font = font)
    draw_image.text((col_inc+offset + (col_inc/4),0), 'Non-Occluded', 'black', font = font)

    #Images
    new_im.paste(PILImage.open(occluded_poultry).convert('RGB') , ((0*col_inc)+offset, (0*col_inc)+offset))
    new_im.paste(PILImage.open(full_poultry).convert('RGB') , ((1*col_inc)+offset, (0*col_inc)+offset))
    new_im.paste(PILImage.open(occluded_swine).convert('RGB') , ((0*col_inc)+offset, (1*col_inc)+offset))
    new_im.paste(PILImage.open(full_swine).convert('RGB') , ((1*col_inc)+offset, (1*col_inc)+offset))

    #Row labels (poultry)
    txt=PILImage.new('RGB', (col_inc,offset), color = (255,255,255))
    d = ImageDraw.Draw(txt)
    d.text( (col_inc/3, 0), "Poultry",  font=font, fill='black')
    w=txt.rotate(90)
    new_im.paste(w, (0,offset))

    #Row labels (swine)
    txt=PILImage.new('RGB', (col_inc,offset), color = '#ffffff')
    d = ImageDraw.Draw(txt)
    d.text( (col_inc/2.5, 0), "Swine",  font=font, fill='black')
    w=txt.rotate(90)
    new_im.paste(w, (0,col_inc+offset))

    #Image labels
    draw_image.text((offset+2,offset), 'a', 'white', font=font)
    draw_image.text(((1*col_inc)+offset+2, (0*col_inc)+offset), 'c', 'white', font=font)
    draw_image.text(((0*col_inc)+offset+2, (1*col_inc)+offset), 'b', 'white', font=font)
    draw_image.text(((1*col_inc)+offset+2, (1*col_inc)+offset), 'd', 'white', font=font)

    #Save
    new_im.save(os.path.join(fig_dir, '1_Figure_fgocclusion_Image_Occlusion_Examples.png'), dpi=(300,300))
    
    print("Done.")

#Add arguments
if __name__ == '__main__':
    main() 
