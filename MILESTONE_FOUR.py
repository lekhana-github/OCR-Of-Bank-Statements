
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from paddleocr import PaddleOCR
import tempfile
import fitz  # PyMuPDF for PDF processing
import os
from groq import Groq
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
import cloudinary
import cloudinary.uploader
import cloudinary.api
import cloudinary.utils
import requests

def download_image(url):
    response = requests.get(url)
    file_path = "temp_image.jpg"
    with open(file_path, "wb") as file:
        file.write(response.content)
    return file_path


# Set up the Groq API client
os.environ["GROQ_API_KEY"] = "API KEY"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Set page configuration
st.set_page_config(page_title="PaddleOCR Text Detection", layout="wide")

# Cloudinary Configuration
cloudinary.config(
    cloud_name="CLOUD NAME", 
    api_key="API KEY", 
    api_secret="SECRET KEY", 
    secure=True
)

def load_lottiefile(filepath:str):
    with open(filepath,"r",encoding="utf-8") as f:
        return json.load(f)
    
def load_lottieurl(url:str):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

# Function to upload an image to Cloudinary
def upload_image_to_cloudinary(image, folder_name):
    try:
        upload_result = cloudinary.uploader.upload(image, folder=folder_name)
        return upload_result["secure_url"]
    except Exception as e:
        st.error(f"Error uploading image: {e}")
        return None
# Initialize session state for folder_name
if "folder_name" not in st.session_state:
    st.session_state.folder_name = None

def cloud():
    img = Image.open("CLOUD.png")

        # Resize the image
    resized_img = img.resize((img.width, 300))  # Adjust width and height as needed

        # Display the resized image
    st.image(resized_img)
    # Step 1: User selects a category
    st.write("Please click a button to select a category:")
    col1, col2, col3 = st.columns(3)

    if col1.button("💼 Salary Slip", key="salary", use_container_width=True):
        st.session_state.folder_name = "SalarySlip"
    elif col2.button("📝 Transaction History", key="trans", use_container_width=True):
        st.session_state.folder_name = "Transaction History"
    elif col3.button("📊 Profit and Loss", key="profit", use_container_width=True):
        st.session_state.folder_name = "Profit and Loss"

    # Step 2: Allow the user to upload an image if a category is selected
    if st.session_state.folder_name:
        st.write(f"Upload an image for {st.session_state.folder_name}:")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "pdf"])

        if uploaded_file is not None:
            if st.button("Upload to Cloudinary"):
                with st.spinner("Uploading image..."):
                    image_url = upload_image_to_cloudinary(uploaded_file, st.session_state.folder_name)
                    if image_url:
                        st.success(f"Successfully uploaded to the cloud: {image_url}")
                        st.image(image_url, caption="Uploaded Image", width=600)


@st.cache_data
def convert_df(in_df):

    """
    Converts a DataFrame to a CSV format encoded in UTF-8.

    Args:
        in_df (pd.DataFrame): The input DataFrame to be converted.

    Returns:
        bytes: CSV-encoded bytes of the DataFrame.
    """

    return in_df.to_csv().encode('utf-8')

@st.cache_resource
def init_ppocr(lang="en", params=None):

    """
    Initializes the PaddleOCR reader with specified parameters.

    Args:
        lang (str): Language to be used for OCR.
        params (dict, optional): Parameters for PaddleOCR configuration. Defaults to None.

    Returns:
        PaddleOCR: Initialized PaddleOCR reader.
    """

    if params is None:
        params = {
            "det_db_thresh": 0.3,
            "det_db_box_thresh": 0.6,
            "det_db_unclip_ratio": 1.6,
            "use_gpu": False,
            "max_text_length": 25,
            "drop_score": 0.5,
            "rec_algorithm": "CRNN",
            "rec_char_type": "en",
            "rec_batch_num": 6,
            "rec_image_shape": "3, 32, 320",
            "use_angle_cls": True  # Enable angle classification
        }
    return PaddleOCR(lang=lang, **params)


@st.cache_data
def ppocr_detect(_in_reader, in_image_path):

    
    # Performs OCR detection on the provided image using the PaddleOCR reader.

    # Args:
    #     _in_reader (PaddleOCR): The initialized PaddleOCR reader.
    #     in_image_path (str): Path to the input image for OCR detection.

    # Returns:
    #     tuple: OCR results and status (either 'OK' or an error message).
    
     
    try:
        out_ppocr_results = _in_reader.ocr(in_image_path, rec=True)
        out_status = 'OK'
    except Exception as e:
        out_ppocr_results = []
        out_status = e

    return out_ppocr_results, out_status

def save_uploaded_file(uploadedfile):

    """
    Saves the uploaded file temporarily for processing.

    Args:
        uploadedfile (UploadedFile): The uploaded file from Streamlit.

    Returns:
        str: Path to the saved temporary file.
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf" if uploadedfile.type == "application/pdf" else ".jpg") as temp_file:
        temp_file.write(uploadedfile.read())
        return temp_file.name

def pdf_to_images(pdf_path):

    """
    Converts a PDF file into a list of images, one for each page.

    Args:
        pdf_path (str): Path to the input PDF file.

    Returns:
        list: List of PIL.Image objects representing pages of the PDF.
    """

    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        img = page.get_pixmap()
        pil_img = Image.frombytes("RGB", [img.width, img.height], img.samples)
        images.append(pil_img)
    return images

def generate_prompt_response(recognized_text):

     
    # Generates an AI response based on the recognized text and user prompt.

    # Args:
    #     recognized_text (str): The text recognized by OCR.

    # Displays:
    #     str: AI-generated response to the user prompt.
    
     
    user_prompt = st.text_input("Enter your prompt related to the recognized text:")
    if user_prompt:
        with st.spinner("Generating response..."):
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "The user provided OCR-detected text."},
                    {"role": "user", "content": recognized_text},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama3-8b-8192"
            )
        st.markdown("### AI Response:")
        st.markdown(
            f'<div style="max-height: 400px; overflow-y: scroll; padding: 10px; border: 1px solid #ccc; background-color: #f9f9f9;">'
            f'{response.choices[0].message.content}</div>',
            unsafe_allow_html=True
        )


def summary(recognized_text):
     
    sum_prompt = "Please summarize the content based on the recognized text in the single paragraph with simple points."
    if sum_prompt:
        with st.spinner("Generating response..."):
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "The user provided OCR-detected text."},
                    {"role": "user", "content": recognized_text},
                    {"role": "user", "content": sum_prompt}
                ],
                model="llama3-8b-8192"
            )
        st.markdown("SUMMARY OF BANK STATEMENT")
        # st.markdown(response.choices[0].message.content)
        st.markdown(
            f'<div style="max-height: 400px; overflow-y: scroll; padding: 10px; border: 1px solid #ccc; background-color: #f9f9f9;">'
            f'{response.choices[0].message.content}</div>',
            unsafe_allow_html=True
        )

def datatable(recognized_text):
     
    table_prompt = "display the data in the form of single table from the recognised text"
    if table_prompt:
        with st.spinner("Generating response..."):
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "The user provided OCR-detected text."},
                    {"role": "user", "content": recognized_text},
                    {"role": "user", "content": table_prompt}
                ],
                model="llama3-8b-8192"
            )
        st.markdown("ANALYSED DATA")
        # st.markdown(response.choices[0].message.content)
        st.markdown(
            f'<div style="max-height: 400px; overflow-y: scroll; padding: 10px; border: 1px solid #ccc; background-color: #f9f9f9;">'
            f'{response.choices[0].message.content}</div>',
            unsafe_allow_html=True
        )

def datajson(recognized_text):
     
    json_prompt = "display the data in the form of single and simple json from the recognised text"
    if json_prompt:
        with st.spinner("Generating response..."):
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "The user provided OCR-detected text."},
                    {"role": "user", "content": recognized_text},
                    {"role": "user", "content": json_prompt}
                ],
                model="llama3-8b-8192"
            )
        st.markdown("ANALYSED DATA")
        # st.markdown(response.choices[0].message.content)
        st.markdown(
            f'<div style="max-height: 400px; overflow-y: scroll; padding: 10px; border: 1px solid #ccc; background-color: #f9f9f9;">'
            f'{response.choices[0].message.content}</div>',
            unsafe_allow_html=True
        )


def rename_files(files):

    """
    Allows users to rename uploaded files through Streamlit inputs.

    Args:
        files (list): List of uploaded files.

    Returns:
        dict: Mapping of new file names to file objects.
    """
    
    renamed_files = {}
    for file in files:
        new_name = st.text_input(f"Rename {file.name}", value=file.name)
        renamed_files[new_name] = file
    return renamed_files

def app1():

    """
    Main application function to handle file uploads, OCR detection, and AI prompts.

    Displays:
        Streamlit interface for uploading files, performing OCR, and interacting with AI.
    """

    st.title("PaddleOCR Text Detection and Recognition with AI Prompts")

    st.markdown("#### Upload Images or PDFs for OCR:")
    uploaded_files = st.file_uploader("Upload images or PDFs", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

    if uploaded_files:
        renamed_files = rename_files(uploaded_files)
        categories = ["Transaction History", "Salary Slip", "Profit and Loss"]

        col1, col2, col3, col4 = st.columns(4)
        category = None

        with col1:
            if st.button("📝 Transaction History", key="trans", use_container_width=True):
                category = "Transaction History"
        with col2:
            if st.button("💼 Salary Slip", key="salary", use_container_width=True):
                category = "Salary Slip"
        with col3:
            if st.button("📊 Profit and Loss", key="profit", use_container_width=True):
                category = "Profit and Loss"
        with col4:
            if st.button("📂 Other", key="other", use_container_width=True):
                category = "Other"

        if category:
            st.markdown(f"**Selected Category: {category}**")
            selected_file = next((file for name, file in renamed_files.items() if name == category), None)
        else:
            selected_file = next((file for name, file in renamed_files.items() if name not in categories), None)

        if selected_file:
            file_path = save_uploaded_file(selected_file)

            ocr_reader = init_ppocr(lang="en", params={
                "det_db_thresh": 0.3,
                "det_db_box_thresh": 0.6,
                "det_db_unclip_ratio": 1.6,
                "use_gpu": False,
                "max_text_length": 25,
                "drop_score": 0.5,
                "rec_algorithm": "CRNN",
                "rec_char_type": "en",
                "rec_batch_num": 6,
                "rec_image_shape": "3, 32, 320"
            })

            st.markdown(f"### File: {selected_file.name}")

            if file_path.endswith(".pdf"):
                images = pdf_to_images(file_path)
                st.markdown("#### OCR on PDF Pages:")
                for i, img in enumerate(images):
                    st.markdown(f"### Page {i + 1}")

                    col1, col2 = st.columns([3, 3])

                    with col1:
                        st.image(img, caption=f"Page {i + 1}", use_container_width=True)

                    img_path = tempfile.mktemp(suffix=".jpg")
                    img.save(img_path)
                    results, status = ppocr_detect(ocr_reader, img_path)

                    if status == "OK":
                        recognized_text = "".join(
                            f"Text: {line[1][0]} (Confidence: {line[1][1] * 100:.2f}%)\n\n"
                            for result in results for line in result
                        )

                        with col2:
                            st.markdown("### Recognized Text:")
                            st.markdown(
                                f"""
                                <div style="max-height: 260px; overflow-y: scroll; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
                                    <pre>{recognized_text}</pre>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            # generate_prompt_response(recognized_text)
                            
                    else:
                        st.error(f"OCR Detection Error on Page {i + 1}: {status}")
                    summary(recognized_text)
                    datatable(recognized_text)
                    datajson(recognized_text)
                    generate_prompt_response(recognized_text)
            else:
                img = Image.open(file_path)

                col1, col2 = st.columns([3, 3])

                with col1:
                    resized_img = img.resize((900, 500))
                    st.image(resized_img, caption="Uploaded Image", use_container_width=False)

                results, status = ppocr_detect(ocr_reader, file_path)

                if status == "OK":
                    recognized_text = "".join(
                        f"Text: {line[1][0]} (Confidence: {line[1][1] * 100:.2f}%)\n\n"
                        for result in results for line in result
                    )

                    with col2:
                        st.markdown("### Recognized Text:")
                        st.markdown(
                            f"""
                            <div style="max-height: 260px; overflow-y: scroll; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
                                <pre>{recognized_text}</pre>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        # generate_prompt_response(recognized_text)
                        
                else:
                    st.error(f"OCR Detection Error: {status}")
                summary(recognized_text)
                datatable(recognized_text)
                datajson(recognized_text)
                generate_prompt_response(recognized_text)

def app2():
    """
    Fetches an image from Cloudinary based on the user's selected folder,
    displays it with resizing, performs OCR detection, and executes further operations
    like text recognition, AI-based prompts, summary generation, and table display.
    """
    st.title("Cloudinary-Based OCR Text Detection and Analysis")

    # Step 1: User selects a folder (category)
    st.markdown("### Select a Category:")
    col1, col2, col3 = st.columns(3)

    folder_name = None
    if col1.button("💼 Salary Slip"):
        folder_name = "SalarySlip"
    elif col2.button("📝 Transaction History"):
        folder_name = "TransactionHistory"
    elif col3.button("📊 Profit and Loss"):
        folder_name = "ProfitAndLoss"

    if folder_name:
        st.success(f"Selected Category: {folder_name}")

        # Step 2: Fetch a random image from the selected folder
        with st.spinner("Fetching a random image from Cloudinary..."):
            try:
                resources = cloudinary.api.resources(type="upload", prefix=folder_name, max_results=50)
                if resources["resources"]:
                    random_image = np.random.choice(resources["resources"])
                    image_url = random_image["secure_url"]
                    file_path = download_image(image_url)  # Function to download the image locally
                else:
                    st.error("No images found in the selected folder.")
                    return
            except Exception as e:
                st.error(f"Error fetching images: {e}")
                return

        # Step 3: Display the image with resizing
        st.markdown("### Image Preview and OCR Results:")
        img = Image.open(file_path)
        col1, col2 = st.columns([3, 3])

        with col1:
            resized_img = img.resize((900, 500))
            st.image(resized_img, caption="Fetched Image", use_container_width=False)

        # Step 4: Perform OCR detection
        with st.spinner("Performing OCR..."):
            ocr_reader = init_ppocr()
            results, status = ppocr_detect(ocr_reader, file_path)

            if status == "OK":
                recognized_text = "".join(
                    f"Text: {line[1][0]} (Confidence: {line[1][1] * 100:.2f}%)\n\n"
                    for result in results for line in result
                )

                with col2:
                    st.markdown("### Recognized Text:")
                    st.markdown(
                        f"""
                        <div style="max-height: 260px; overflow-y: scroll; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
                            <pre>{recognized_text}</pre>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.error(f"OCR Detection Error: {status}")
                return

        # Step 5: Additional operations
        summary(recognized_text)
        datatable(recognized_text)
        datajson(recognized_text)
        generate_prompt_response(recognized_text)




def home():
    img = Image.open("OCRI.png")

    # Resize the image
    resized_img = img.resize((img.width, 300))  # Adjust width and height as needed

    # Display the resized image
    st.image(resized_img)
    page_bg_img = """
<style>
"""
    # Create two columns: text on the left, animation on the right
    col1, col2,col3 = st.columns([3,2, 1])  # Adjust the proportions as needed

    with col1:
        st.markdown(
    """
    ## OCR (Optical Character Recognition) 📜🔍
    OCR is a computer vision task that involves detecting text areas and recognizing characters within images or scanned documents. 
    It is a critical technology for automating data extraction from financial documents like bank statements. 💼💡

    ### Allows User: 🧑‍💻💬
    This application allows you to upload financial documents, perform OCR to extract the recognized text, and analyze the extracted information. 📈💳

    Upload your bank statements 📂 and see how OCR helps automate the extraction of valuable financial data! 🚀💻
    """
)

    with col2:
        lottie_coding = load_lottiefile("bank.json")
        st_lottie(
            lottie_coding,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=300,  # Adjust the height if needed
            width=None,
            key="coding",
        )
        
    with col3:
        lottie_another  = load_lottiefile("ocr.json")
        # Display the second Lottie animation
        st_lottie(
            lottie_another,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=300,
            key="another",
        )


def about():
    st.markdown(
    """
    ## About This Application 📑

    This application leverages cutting-edge technologies to process financial documents like bank statements and analyze their contents effectively. 🤖

    ### Features and Technologies Used: 🛠️
    - **OCR with PaddleOCR**: Utilized for accurate text detection and recognition from uploaded financial documents. 📸
    - **Large Language Model Analysis with Groq API Keys**: Enables advanced analysis and structuring of extracted text, providing deeper insights. 🧠
    - **Cloudinary Integration**: Allows fetching files directly from the cloud for seamless processing. ☁️
    - **Local File Upload**: Supports uploading files from your device for on-the-go analysis. 💻

    ### How It Works: ⚙️
    1. **Upload Documents**: Users can upload bank statements from local storage or fetch them from Cloudinary. 📤
    2. **Perform OCR**: PaddleOCR extracts text from the uploaded documents. 🔍
    3. **Analyze Text**: The extracted text is analyzed using a large language model, enhancing the understanding of financial data. 📊

    ### User Prompt: ✍️
    You can provide a custom prompt to tailor the analysis. This allows you to focus on specific aspects of the bank statement, such as transactions, balances, or dates.

    Example: 
    - "Analyze the recent transactions and highlight any unusual activity." 💸
    - "Summarize the total expenditure for the month." 🧾

    This application simplifies the process of handling financial documents, making it efficient and user-friendly. ✅
    """
)





def main():

    choice = option_menu(
    menu_title=None,  # No title
    options=["HOME","ABOUT","CLOUD","APP1","APP2"],
    icons=["house", "info","cloud","laptop","laptop"],  # Icons for each option
    orientation="horizontal",  # Horizontal menu
    default_index=0,  # Default selection
    styles={
        "container": {
            "padding": "0!important", 
            "background-color": "#f5f5f5"
        },
        "icon": {
            "color": "blue", 
            "font-size": "25px"
        },
        "nav-link": {
            "font-size": "20px",
            "font-weight": "bold",
            "text-align": "center",
            "margin": "0px",
            "color": "black",
            "--hover-color": "#c9c9ff",
        },
        "nav-link-selected": {
            "background-color": "#007BFF",
            "color": "white",
        },
    },
)
    if choice == "HOME":
        home()
    elif choice == "ABOUT":
        about()
    elif choice == "CLOUD":
        cloud()
    elif choice == "APP1":
        app1()
    elif choice == "APP2":
        app2()


if __name__ == "__main__":
    main()

