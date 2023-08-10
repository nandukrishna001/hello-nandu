import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
# from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances


# Load the pre-trained BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# def add_bg_from_url():
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("https://smallbizclub.com/wp-content/uploads/2019/07/Successfully-Deal-with-Customer-Complaints.jpg");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )
#
#
# add_bg_from_url()

nav = st.sidebar.radio("Menu",["Home","Principal's message","Contact Us"])
if nav == "Principal's message":
    st.header("Principal's Message")
    st.write("Education is a journey, and together, we embark on this journey with a shared vision of creating an inclusive and inspiring learning environment. Our dedicated team of educators is committed to fostering a love for learning, encouraging critical thinking, and cultivating creativity among our students. We understand that every child is unique, and we strive to provide personalized support to ensure their growth and success. "
             "We also recognize the significance of a strong partnership between school, parents, and the community. Your active involvement and support are invaluable in shaping the future of our students. Together, we can create an environment that nurtures talents, celebrates achievements, and embraces challenges."

"As we navigate the ever-changing world of education, we are committed to staying at the forefront of innovative teaching practices and technology integration. Our aim is to equip our students with the skills they need to thrive in the 21st-century global landscape."

"I encourage all students to dream big, set ambitious goals, and be willing to persevere in the face of challenges. Together, we will create a legacy of excellence that will endure for generations to come."

"Let us join hands and work collaboratively to build a community that is respectful, caring, and compassionate—a community where each individual is valued and respected.")

if nav == "Contact Us":
    st.header("Contact Us")
    st.write("e-mail:  abcenquiry@gmail.com")
    st.write("Phone:  00-0000000")


# Function to calculate BERT embeddings
def get_bert_embeddings(text):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor(tokens).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        embeddings = model(input_ids)[0]
    return embeddings.squeeze().mean(dim=0).numpy()


# Function to prioritize complaints
def prioritize_complaints(complaints):
    embeddings = [get_bert_embeddings(complaint) for complaint in complaints]
    embeddings = torch.tensor(embeddings)

    # Convert 1D tensor to 2D tensor
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)

    # Calculate the cosine similarity between complaint pairs based on BERT embeddings
    cosine_sim = 1 - pairwise_distances(embeddings, metric='cosine')
    priority_scores = cosine_sim.sum(axis=1)
    return priority_scores


# Streamlit App

if nav == "Home":
    def add_bg_from_url():
        st.markdown(
            f"""
             <style>
             .stApp {{
                 background-image: url("https://smallbizclub.com/wp-content/uploads/2019/07/Successfully-Deal-with-Customer-Complaints.jpg");
                 background-attachment: fixed;
                 background-size: cover
             }}
             </style>
             """,
            unsafe_allow_html=True
        )


    add_bg_from_url()
    st.title("Complaint Prioritization Portal")
    st.write("Enter complaints from different students:")

    # Input text box to enter complaints
    complaints_text = st.text_area("Enter complaints (one per line):")

    if st.button("Prioritize"):
        if complaints_text:
            # Process complaints
            complaints = complaints_text.strip().split('\n')
            priority_scores = prioritize_complaints(complaints)

            # Create a DataFrame with complaints and corresponding priorities
            df_result = pd.DataFrame({'Complaints': complaints, 'Priority': priority_scores})

            # Sort the DataFrame by priority scores (higher score, higher priority)
            df_result = df_result.sort_values(by='Priority', ascending=False)

            # Display the result
            st.subheader("Complaints Prioritized:")
            st.table(df_result)

            # Create a download link for the CSV file
            # csv = df_result.to_csv(index=False)
            # b64 = base64.b64encode(csv.encode()).decode()  # Encode CSV data as base64
            # href = f'<a href="data:file/csv;base64,{b64}" download="complaints_prioritized.csv">Download CSV file</a>'
            # st.markdown(href, unsafe_allow_html=True)
            csv = df_result.to_csv(index=False)
            st.download_button(label="Download CSV file", data=csv, file_name="complaints_prioritized.csv",
                               mime="text/csv")





