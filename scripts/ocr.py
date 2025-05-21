import easyocr
import os


def merge_ocr_results(results):
    sorted_results = sorted(results, key=lambda x: (x[0][0][0], x[0][0][1]))
    merged_text = ' '.join(result[1] for result in sorted_results)

    return merged_text

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

reader = easyocr.Reader(['de','en']) # this needs to run only once to load the model into memory

img_dir = "/data1/yztian/ACL2025/DebackX/results/translation/val/de/text"
result_file = "/data1/yztian/ACL2025/DebackX/results/translation/val/ocr.de"

img_list = sorted(os.listdir(img_dir), key=lambda x: int(x.replace(".jpg", "")))
result_file = open(result_file, "w")
for img in img_list:
    result = reader.readtext(os.path.join(img_dir, img))
    result_file.write(merge_ocr_results(result))
    result_file.write("\n")
    result_file.flush()