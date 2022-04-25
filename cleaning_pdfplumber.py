import re
import pdfplumber

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
    text = re.sub(r"\n", " ", text)

    # removes the occurence of "(cid:[number])". This seems to be how pdfplumber
    # spits out LaTeX, so I removed it as it's just noise
    text = re.sub(r"\(cid:\d*\)", "", text)

    # merges together words that get split due to line breaks (e.g sen- tence)
    text = re.sub(r"([a-z])-\s", r'\1', text)
    return text


def convert_first_page(pdf_path):
    """
    Converts the first page of an academic paper into plain text. This function 
    is under the assumtion that the first page has some sort of title header, 
    with the main body text starting inches away from the top of the page.
    
    Parameters
    ----------
    pdf_path : str
        The relative file path to the PDF
    
    Returns
    -------
    str
        The clean text from the first page of the PDF.
    """
    
    with pdfplumber.open(pdf_path) as pdf:
        # get the first page
        page = pdf.pages[0]

        # crop the left and right columns, excluding the top header 
        # (which is exclusive to the first page)
        left = page.crop((0, 0.21 *  page.height, 0.5 * float(page.width), \
                        0.9 * float(page.height)))
        right = page.crop((0.5 * page.width, 0.21 * page.height, page.width, \
                        0.93 * page.height))

        # extract the text from the left and right columns
        l_text = left.extract_text(x_tolerance=1.5)
        r_text = right.extract_text(x_tolerance=1.5)

        # merge them together
        text = l_text + " " + r_text
        text = clean_text(text)
        return text
    
def convert_all_pages(pdf_path):
    """
    Converts all pages of a PDF into plain text.
    
    Parameters
    ----------
    pdf_path : str
        The relative file path to the PDF
        
    Returns
    -------
    list
        A list containing the entirety of the text from the PDF, processed in
        order. Each element in the list contains a single page's text. (i.e
        list[0] is the first page's text)
    """
    
    all_text = []
    first_page = convert_first_page(pdf_path)
    all_text.append(first_page)

    with pdfplumber.open(pdf_path) as pdf:    
        for i in range(1, len(pdf.pages), 1):
            # get the first page
            page = pdf.pages[i]

            # crop the left and right columns
            left = page.crop((0, 0, 0.5 * float(page.width), \
                            0.9 * float(page.height)))
            right = page.crop((0.5 * page.width, 0.21 * page.height, \
                            page.width, page.height))

            # extract the text from the left and right columns
            l_text = left.extract_text(x_tolerance=0.5)
            r_text = right.extract_text(x_tolerance=1.5)

            # merge them together
            text = l_text + " " + r_text
            text = clean_text(text)
            all_text.append(text)
            
    return all_text

