#Importing Require Libraries
import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, AutoProcessor, BlipForConditionalGeneration
import openai
import torch
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm
import itertools
from nltk.corpus import stopwords
import nltk
import easyocr
import numpy as np
import random
nltk.download('stopwords')

st.title(":blue[Image #HashTag Generator AI Web App]")

def vit_model():
    # Object creation model, tokenizer and processor from HuggingFace
    vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    vit_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return vit_model, feature_extractor, vit_tokenizer


def blip_model():
    # Object creation model, tokenizer and processor from HuggingFace
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, blip_model, blip_tokenizer

def openai_model():
    load_dotenv()
    #Getting the key from env
    openai.api_key = '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' ## you Openai key
    openai_model = "text-davinci-002" # OpenAI model
    return openai_model

def blip_prediction(processor, blip_model, blip_tokenizer, openai_model):
    global predict
    predict = ""
    with st.form("BLIP Image Hastags Uploader"):
        # Image input
        image_list2 = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png", "jpeg"], key="BLIP")
        # Generate button
        submit = st.form_submit_button("Generate")

        if submit:  # submit condition
            max_length = 30
            num_beams = 4
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

            blip_img = []

            for image in tqdm(image_list2):

                i_image = Image.open(image)  # Storing of Image

                if i_image.mode != "RGB":  # Check if the image is in RGB mode
                    i_image = i_image.convert(mode="RGB")

                blip_img.append(i_image)  # Add image to the list

            # Image data to pixel values
            pixel_val = processor(images=blip_img, return_tensors="pt").pixel_values
            pixel_val = pixel_val.to(device)

            # Using model to generate output from the pixel values of Image
            output = blip_model.generate(pixel_val, **gen_kwargs)

            # To convert output to text
            predict = blip_tokenizer.batch_decode(output, skip_special_tokens=True)
            predict = [pred.strip() for pred in predict]

            for index in range(len(image_list2)):
                img = Image.open(image_list2[index])
                st.image(img, width=400)
                st.header("BLIP Model Caption For Image: " + str(predict[index]))
                hashtag_generator(predict[index], openai_model)
                #NLP_OCR(img, predict[index])

def vit_prediction(vit_model, feature_extractor, vit_tokenizer, openai_model):
    global preds
    preds = ""
    with st.form("VIT Image Hastags Uploader"):
        # Image input
        image_list = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png", "jpeg"], key="VIT")
        # Generate button
        submit = st.form_submit_button("Generate")

        if submit:  # submit condition
            max_length = 16
            num_beams = 4
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
            image = []
            for image_path in image_list:
                img = Image.open(image_path)
                if img.mode != "RGB":
                    img = img.convert(mode="RGB")

                image.append(img)

            pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            output_ids = vit_model.generate(pixel_values, **gen_kwargs)

            preds = vit_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]

            for index in range(len(image_list)):
                img = Image.open(image_list[index])
                st.image(img, width=400)
                st.header("VIT Model Caption For Image: " + str(preds[index]))
                hashtag_generator(preds[index], openai_model)
                #NLP_OCR(img, preds[index])

def hashtag_generator(des, openai_model):
    # Prompt
    hashtag_prompt = ('''Generate ten relevant and accurate hashtags that will help the photo 
        reach a larger audience on Instagram and Twitter for a photo that shows ''' + des + '''. The hashtag
        can be funny and creative. Please also provide in this format.
        Hashtags:
        #[Hashtag1] #[Hashtag2] #[Hashtag3] #[Hashtag4] #[Hashtag5] #[Hashtag6] #[Hashtag7] #[Hashtag8] #[Hashtag9] #[Hashtag10]
        ''')

    # Hashtag Generation
    response = openai.Completion.create(
        engine=openai_model,
        prompt=hashtag_prompt,
        max_tokens=(20 * 10),
        n=1,
        stop=None,
        temperature=0.7,
    )

    hashtags = response.choices[0].text.strip().split("\n")
    st.markdown("Openai Hastags for this image")
    for hashtag in hashtags:
        st.markdown(hashtag)

def NLP_OCR(image, caption_words):
    #Setting up OCR
    reader = easyocr.Reader(['en'])
    # remove stop words from caption words
    stop_words = set(stopwords.words('english'))
    caption_words = [word for word in caption_words if word not in stop_words]

    # use easyocr to extract text from the image
    text = reader.readtext(np.array(image))
    detected_text = " ".join([item[1] for item in text])

    # combine caption words and detected text
    all_words = caption_words + detected_text.split()

    # generate combinations of words for hashtags
    hashtags = []
    for n in range(1, 4):
        word_combinations = list(itertools.combinations(all_words, n))
        for combination in word_combinations:
            hashtag = "#" + "".join(combination)
            hashtags.append(hashtag)

    # return top 10 hashtags by frequency
    top_hashtags = [tag for tag in sorted(set(hashtags), key=hashtags.count, reverse=True) if tag != "#"]
    st.markdown("traditional NLP&OCR methods Hastags for this image")
    for i in top_hashtags[:10]:
        st.markdown(i)


vit_model, feature_extractor, vit_tokenizer = vit_model()
processor, blip_model, blip_tokenizer = blip_model()
openai_model = openai_model()

# Setting for the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model.to(device)
blip_model.to(device)


tab1, tab2 = st.tabs(["BLIP Model", "VIT Model"])
with tab1:
   st.header("BLIP: Bootstrapping Language-Image Pre-training Model")
   st.markdown('''BLIP is a new pre-training framework for unified vision-language understanding and generation, which 
   achieves state-of-the-art results on a wide range of vision-language tasks.''')
   blip_prediction(processor, blip_model, blip_tokenizer, openai_model)


with tab2:
   st.header("Vision Transformer (ViT) Model")
   st.markdown('''Transformer, an attention-based encoder-decoder architecture, has not only revolutionized the field of
    natural language processing (NLP), but has also done some pioneering work in the field of computer vision (CV). 
    Compared to convolutional neural networks (CNNs), the Vision Transformer (ViT) relies on excellent modeling capabilities 
    to achieve very good performance on several benchmarks such as ImageNet, COCO, and ADE20k. ViT is inspired by the 
    self-attention mechanism in natural language processing, where word embeddings are replaced with patch embeddings.''')
   vit_prediction(vit_model, feature_extractor, vit_tokenizer, openai_model)
