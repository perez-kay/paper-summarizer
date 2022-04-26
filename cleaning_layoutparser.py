from pdf2image import convert_from_path
import numpy as np
import layoutparser as lp
import re

model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config', 
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

ocr_agent = lp.TesseractAgent(languages='eng')

pdf_images = convert_from_path("test papers/textrank algorithm paper.pdf")

def sort_textblock_ids(page_img):
    """
    Sorts the text block IDs in order as they appear on the page (i.e the
    first paragraph is ID 0, next is ID 1, etc).

    Parameters
    ----------
    page_img : np.array
        The image of the current page, converted into a NumPy array
    
    Returns
    -------
    lp.Layout
        A Layout object containing the sorted text blocks
    """
    
    layout = model.detect(page_img)
    # filter out text blocks
    text_blocks = lp.Layout([block for block in layout if block.type == 'Text'])

    # filter out title blocks
    #title_blocks = lp.Layout([block for block in layout if block.type == "Title"])

    image_width = len(page_img[0])

    # Sort element ID of the left column based on y1 coordinate
    left_interval = lp.Interval(0, image_width / 2, axis='x')\
                    .put_on_canvas(page_img)

    # returns a list of layout elements on the left of the page?
    left_blocks = text_blocks.filter_by(left_interval, center=True)._blocks

    # sort by their order number
    left_blocks.sort(key = lambda block : block.coordinates[1])

    # Sort element ID of the right column based on y1 coordinate
    right_blocks = [b for b in text_blocks if b not in left_blocks]
    right_blocks.sort(key = lambda b:b.coordinates[1])

    # Sort the overall element ID starts from left column
    text_blocks = lp.Layout([b.set(id = idx) for idx, b \
                    in enumerate(left_blocks + right_blocks)])
   
    return text_blocks

def clean_text(text):
    """
    Cleans up the raw text from the PDF extraction by merging words that get 
    split by line breaks, removing newline characters, and any other noise from 
    extraction.
    
    Parameters
    ----------
    text : str
        The raw text from the PDF extraction
    
    Returns
    -------
    str
        The cleaned up text, with no newlines, split words, or additional noise
    """

    # removes newline characters
    text = re.sub("\n", " ", text)

    # removes the occurence of "\x0c" which I believe is a tab character? 
    # not sure, but it's not needed
    text = re.sub("\\x0c", "", text)

    # merges together words that get split due to line breaks (e.g sen- tence)
    text = re.sub("([a-z])-\s", r"\1", text)
    return text


def convert_single_page(page_img):
    """
    Converts a single PDF page into plain text.

    Parameters
    ----------
    page_img : np.array
        The image of the current page, converted into a NumPy array
    
    Returns
    -------
    list
        A list containing the page's text. Each element in the list is a
        paragraph of text.
    """    

    text_blocks = sort_textblock_ids(page_img)
    
    for block in text_blocks:

        # Crop image around the detected layout
        segment_image = (block.pad(left=15, right=15, top=5, bottom=5)\
                        .crop_image(page_img))

        # Perform OCR (Optical Character Recognition)
        text = ocr_agent.detect(segment_image)

        # Save OCR result
        block.set(text=text, inplace=True)
   
    page_text = []
    for block in text_blocks:
        text = clean_text(block.text)
        page_text.append(text)
        
    return page_text


def convert_all_pages(pdf_path):
    """
    Converts all pages of a PDF into plain text. This method only grabs text
    and ignores any titles, figures, tables, images, etc.

    Parameters
    ----------
    pdf_path : str
        The relative path to the PDF

    Returns
    -------
    list
        A list containing the entire PDF's text. 
    """
    pdf_images = convert_from_path(pdf_path)
    all_text = []
    for page in pdf_images:
        page_image = np.asarray(page)
        all_text += convert_single_page(page_image)
    return all_text