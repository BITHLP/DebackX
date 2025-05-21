# image size: 512 x 512
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms
import os

def render_subtitle(img, text, font_path):
    drawPIL = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 20, encoding="utf-8")
    text_list = split_subtitle(drawPIL, text, font)
    multi_line_text = "\n".join(text_list)
    text_position = get_text_position(drawPIL, multi_line_text, font)
    img = draw_gray_rectangle(img, text_position, (128, 128, 128), 0.5)
    drawPIL = ImageDraw.Draw(img)
    x, y, _, _ = text_position
    drawPIL.multiline_text((x, y), multi_line_text, font=font, align="center")
    return img, text_position


def split_subtitle(drawPIL, sentence, font):
    result = []
    max_len = 50
    while len(sentence) > max_len:
        split_index = max_len
        while sentence[split_index] != ' ':
            split_index -= 1
            if split_index <= 0:
                split_index = max_len
                break
        result.append(sentence[:split_index].strip())
        sentence = sentence[split_index:].strip()

    result.append(sentence.strip())
    # print(result)
    return result


def get_subtitle_length(drawPIL, text, font):
    return drawPIL.textlength(text, font)


def get_text_position(drawPIL, text, font): 
    x1, y1, x2, y2 = drawPIL.multiline_textbbox((0, 0), text , font=font, align="center")
    y_offset = 42 - y2
    x_offset = (512 - x2) // 2
    x1 += x_offset
    y1 += y_offset
    x2 += x_offset
    y2 += y_offset
    y2 += 5
    return (x1, y1, x2, y2)


def draw_gray_rectangle(image, position, color, alpha):
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(position, fill=color+(int(alpha*255), ))
    image = Image.alpha_composite(image.convert('RGBA'), overlay)
    return image.convert("RGB")


def build_image(src_text_file_path, tgt_text_file_path, output_src_subtitle_file, output_tgt_subtitle_file, src_text_img_dir, tgt_text_img_dir):
    """
    src_text_file_path: file contains subtitle texts (source language)
    tgt_text_file_path: file contains subtitle texts (target language)
    output_src_subtitle_file: output file contains subtitle in source language
    output_tgt_subtitle_file: output file contains subtitle in target language
    output_src_text_img_dir: 
    output_tgt_text_img_dir: 
    """
    src_text_list = []
    tgt_text_list = []
    src_text_file = open(src_text_file_path, "r")
    tgt_text_file = open(tgt_text_file_path, "r")

    for l in src_text_file:
        src_text_list.append(l.strip())
    for l in tgt_text_file:
        tgt_text_list.append(l.strip())
    
    if not os.path.exists(src_text_img_dir):
        os.makedirs(src_text_img_dir)
    if not os.path.exists(tgt_text_img_dir):
        os.makedirs(tgt_text_img_dir)

    src_subtitle_file = open(output_src_subtitle_file, "w")
    tgt_subtitle_file = open(output_tgt_subtitle_file, "w")

    idx = 1
    # for font_path in ["/data2/yztian/IIMT30k/TimesNewRoman.ttf", "/data2/yztian/IIMT30k/Arial.ttf"]:
    for font_path in ["/data2/yztian/IIMT30k/TimesNewRoman.ttf"]:
        for s, t in zip(src_text_list, tgt_text_list):
            empty_image = Image.new('RGB', (512, 48), (0, 0, 0))
            src_text_img, src_position = render_subtitle(empty_image, s, font_path)
            empty_image = Image.new('RGB', (512, 48), (0, 0, 0))
            tgt_text_img, tgt_position = render_subtitle(empty_image, t, font_path)

            if src_position[3] - src_position[1] <= 48 and  tgt_position[3] - tgt_position[1] <= 48:
                src_subtitle_file.write(s+"\n")
                tgt_subtitle_file.write(t+"\n")
                src_text_img.save(os.path.join(src_text_img_dir, str(idx)+".jpg"))
                tgt_text_img.save(os.path.join(tgt_text_img_dir, str(idx)+".jpg"))
                idx += 1

if __name__ == "__main__":
    # build_image("/data1/yztian/en-de/data/train/txt/train.de", "/data1/yztian/en-de/data/train/txt/train.en",
    #             "/data2/yztian/IIMT30k/train-text-img-iwslt/de/subtitle.txt", "/data2/yztian/IIMT30k/train-text-img-iwslt/en/subtitle.txt",
    #             "/data2/yztian/IIMT30k/train-text-img-iwslt/de/text_img", "/data2/yztian/IIMT30k/train-text-img-iwslt/en/text_img")

    build_image("/data1/yztian/wmt14/train.de", "/data1/yztian/wmt14/train.en",
                "/data2/yztian/IIMT30k/train-text-img-wmt14/de/subtitle.txt", "/data2/yztian/IIMT30k/train-text-img-wmt14/en/subtitle.txt",
                "/data2/yztian/IIMT30k/train-text-img-wmt14/de/text_img", "/data2/yztian/IIMT30k/train-text-img-wmt14/en/text_img")