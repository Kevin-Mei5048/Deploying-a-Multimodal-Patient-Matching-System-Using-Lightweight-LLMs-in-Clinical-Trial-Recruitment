import os
import glob
from paddleocr import PaddleOCR  # OCR模块
import fitz  # PyMuPDF


def process_files_path(directory_path, output_txt_file):
    """
    处理给定目录下的所有文件，自动化处理 PDF 转图像、图像转文本、文本文件读取等任务，
    并将最终的结果写入到指定的文本文件中。
    
    参数：
    - directory_path: 要处理的文件夹路径。
    - output_txt_file: 最终生成的文本文件路径。
    """
    final_content = ""  # 用于存储所有提取的文本内容
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        print(f"Processing file: {filename}")

        # 获取文件扩展名
        extension = os.path.splitext(filename)[1].lower()

        if extension == '.pdf':
            image_paths = pdf_to_image(file_path)
            print(f"Generated images: {image_paths}")
            for image_path in image_paths:
                extracted_text = image_to_text(image_path)
                final_content += extracted_text + "\n**********\n"
        
        elif extension in ['.png', '.jpg', '.jpeg']:
            extracted_text = image_to_text(file_path)
            final_content += extracted_text + "\n**********\n"
        
        elif extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as txt_file:
                txt_content = txt_file.read()
                final_content += txt_content + "\n**********\n"
    
    # 将所有提取的文本内容写入到输出文本文件
    with open(output_txt_file, 'w', encoding='utf-8') as output_file:
        output_file.write(final_content)
    
    print(f"Final content saved to: {output_txt_file}")


def pdf_to_image(pdf_file):
    """
    将 PDF 文件的每一页转换为图像，并保存在 PDF 同级目录下的 PDF_Image 文件夹中。
    """
    
    # 1. 初始化: 获取 PDF 所在路径及生成图片存放目录
    print(f"Converting PDF to images: {pdf_file}")
    
    # 提取 PDF 文件所在目录的父目录名，并创建图片保存目录
    pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]  # 获取 PDF 文件名（不含扩展名）
    image_directory = os.path.join(os.path.dirname(pdf_file), "PDF_Image")
    
    # 如果目标文件夹不存在，则创建它
    os.makedirs(image_directory, exist_ok=True)

    # 2. 打开 PDF 文件
    pdf_document = fitz.open(pdf_file)
    
    # 设置缩放因子，提高输出图像分辨率
    zoom = 4.0  # 可调节的缩放因子
    mat = fitz.Matrix(zoom, zoom)

    # 3. 遍历每一页，将其转换为图片
    image_paths = []
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        image = page.get_pixmap(matrix=mat)
        
        # 生成并保存图片文件
        image_path = os.path.join(image_directory, f"{pdf_name}_output_image_{page_number}.png")
        image.save(image_path)
        
        # 打印日志并将路径保存到列表中
        print(f"Saved image: {image_path}")
        image_paths.append(image_path)
    
    # 4. 返回所有生成图片的路径列表
    return image_paths


def get_line_number(coordinate):
    """
    计算文本块的行号(Y 轴的平均值)。
    """
    return sum(y for _, y in coordinate) / len(coordinate)


def restore_text(ocr_data):
    """
    将 OCR 识别出的文本按行号进行分组和排序，还原为有序文本。
    """
    
    # 1. 初始化一个空字典，用于按行存储文本块
    lines = {}

    # 2. 遍历 OCR 数据，将文本块按行进行分组
    for item_list in ocr_data:
        for coordinate, (text, _) in item_list:
            line_number = get_line_number(coordinate)
            
            # 根据行号将文本块分组，使用阈值来判断是否属于同一行
            threshold = 30
            found = False
            for key in lines.keys():
                if abs(line_number - key) < threshold:
                    lines[key].append((coordinate, text))
                    found = True
                    break
            if not found:
                lines[line_number] = [(coordinate, text)]
    # 此时lines储存格式为：
    # lines = {
    # 150: [((100, 150), "Hello"), ((200, 150), "world!")],
    # 250: [((100, 250), "This"), ((200, 250), "is"), ((300, 250), "OCR")]
    # }  

    
    # 3. 按行号排序并拼接文本，返回完整的有序文本
    restored_text = ''
    for key in sorted(lines.keys()):
        line = lines[key]
        # 按 X 坐标排序文本块，以确保一行内的文本顺序正确
        line.sort(key=lambda x: min(coord[0] for coord in x[0]))
        restored_text += ' '.join(text for _, text in line) + '\n'
    
    return restored_text

    # 此时restored_text返回格式应为：
    # Hello world!
    # This is OCR



def image_to_text(image_path):
    """
    使用 PaddleOCR 对单个图像文件进行 OCR 识别，并返回识别出的文本。
    """
    # 初始化 OCR 模型
    ocr = PaddleOCR(use_angle_cls=True)

    # 打印文件信息，并进行 OCR 识别
    print(f"Processing image: {image_path}")
    ocr_data = ocr.ocr(image_path, cls=True)
    print(f"OCR Data: {ocr_data}")

    # test
    print(f"OCR Data for {image_path}: {ocr_data}")
    # 将 OCR 识别结果恢复为有序文本
    text = restore_text(ocr_data)

    # 返回识别出的文本
    return text
