import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

st.set_page_config(
    page_title='Authenticity of Job Posting Predictions - Exploratory Data Analysis',
    layout='wide',
    initial_sidebar_state='expanded')

def run():
    # create title/header
    st.title('Authenticity of Job Posting Predictions')
    # create subheader
    st.subheader('Exploratory Data Analysis for Authenticity of Job Posting')
    # add image
    st.image('https://resources.workable.com/wp-content/uploads/2019/02/job-boards-list.png',
             caption='Authenticity of Job Posting')
    # add description
    st.write('### Problem Statement')
    st.write('In this digital era, companies often use online platforms to post job vacancies and search for prospective employees. However, the increasing number of fake job vacancies on various platforms is not only harming job seekers looking for job opportunities but also harming the credibility of the companies being imitated.')
    st.write('### Objective')
    st.write('To minimize the risk of job vacancy scams by developing more efficient ways to identify and filter out fake job postings.')
    st.markdown('---')

    # show dataframe
    df = pd.read_csv('fake_job_postings.csv')
    df['text'] = df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements']
    df['text'] = df['text'].astype(str)

    # create submenu
    submenu = st.sidebar.selectbox('Exploratory Data Analysis Navigation',['Dataset','Percentage of Job Postings','Most Employment Types of Real Jobs and Fake Jobs','Most Required Experience of Real Jobs and Fake Jobs','Highest 5 Countries of Real Jobs and Fake Jobs','Word Cloud for Real Jobs and Fake Jobs'])
    if submenu == 'Dataset':
        st.write('### Dataset')
        st.write('Simple exploration using the Job Posting Predictions Dataset taken from [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction/data)')
        st.dataframe(df)
        st.markdown('---')

    elif submenu == 'Percentage of Job Postings':
        # create visualization 1
        st.write('### Percentage of Job Postings')
        fig = plt.figure(figsize=(5,5))
        df['fraudulent'].value_counts().plot(kind='pie',autopct='%.2f%%')
        plt.legend(labels=['0 = Real Jobs, 1 = Fake Jobs'])
        st.pyplot(fig)
        st.write('There is a significant difference between genuine job vacancies and scams, with fraudulent job postings accounting for only 2.14%. This indicates an imbalanced distribution.')

    elif submenu == 'Most Employment Types of Real Jobs and Fake Jobs':
        # definition of fakejobs and realjobs
        fakejobs = df[df['fraudulent'] == 1]
        realjobs = df[df['fraudulent'] == 0]
        # create visualization 2
        st.write('### Most Employment Types of Real Jobs and Fake Jobs')
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ## fakejobs
        sns.countplot(x='employment_type', data=fakejobs, palette='Set2', order=fakejobs['employment_type'].value_counts().index[:5], ax=axes[0])
        axes[0].set_title('Most Employment Types of Fake Jobs')
        axes[0].set_xlabel('Employment Types')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        ## realjobs
        sns.countplot(x='employment_type', data=realjobs, palette='Set2', order=realjobs['employment_type'].value_counts().index[:5], ax=axes[1])
        axes[1].set_title('Most Employment Types of Real Jobs')
        axes[1].set_xlabel('Employment Types')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        st.write('There is no significant difference between genuine job vacancies and scams based on the type of work. This indicates that it is difficult to determine whether a job posting is genuine or a scam based solely on the type of work.')

    elif submenu == 'Most Required Experience of Real Jobs and Fake Jobs':
        # definition of fakejobs and realjobs
        fakejobs = df[df['fraudulent'] == 1]
        realjobs = df[df['fraudulent'] == 0]
        # create visualization 3
        st.write('### Most Required Experience of Real Jobs and Fake Jobs')
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ## fakejobs
        sns.countplot(x='required_experience', data=fakejobs, palette='Set2', order=fakejobs['required_experience'].value_counts().index[:5], ax=axes[0])
        axes[0].set_title('Most Required Experience of Fake Jobs')
        axes[0].set_xlabel('Required Experience')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        ## realjobs
        sns.countplot(x='required_experience', data=realjobs, palette='Set2', order=realjobs['required_experience'].value_counts().index[:5], ax=axes[1])
        axes[1].set_title('Most Required Experience of Real Jobs')
        axes[1].set_xlabel('Required Experience')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        st.write('There is no significant difference between genuine job vacancies and scams based on work experience. This indicates that it is difficult to understand whether a job posting is genuine or a scam solely based on work experience.')

    elif submenu == 'Highest 5 Countries of Real Jobs and Fake Jobs':
        # definition of fakejobs and realjobs
        fakejobs = df[df['fraudulent'] == 1]
        realjobs = df[df['fraudulent'] == 0]
        # create visualization 4
        st.write('### Highest 5 Countries of Real Jobs and Fake Jobs')
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ## fakejobs
        sns.countplot(x='location', data=fakejobs, palette='Set2', order=fakejobs['location'].value_counts().index[:5], ax=axes[0])
        axes[0].set_title('Highest 5 Countries of Fake Jobs')
        axes[0].set_xlabel('Location')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        ## realjobs
        sns.countplot(x='location', data=realjobs, palette='Set2', order=realjobs['location'].value_counts().index[:5], ax=axes[1])
        axes[1].set_title('Highest 5 Countries of Real Jobs')
        axes[1].set_xlabel('Location')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        st.write("The primary target of job vacancy scams is located in the United States, especially in the states of Texas and California. This indicates that job seekers should exercise greater caution when the company's location is in Texas and California.")

    elif submenu == 'Word Cloud for Real Jobs and Fake Jobs':
        # text visualization for fake jobs
        st.write('### Word Cloud for Fake Jobs')
        fake_wordcloud = WordCloud().generate(" ".join(df[df['fraudulent'] == 1]['text']))
        fig = plt.figure(figsize=(5,5))
        plt.imshow(fake_wordcloud, interpolation='bilinear')
        st.pyplot(fig)
        # text visualization for real jobs
        st.write('### Word Cloud for Real Jobs')
        real_wordcloud = WordCloud().generate(" ".join(df[df['fraudulent'] == 0]['text']))
        fig = plt.figure(figsize=(5,5))
        plt.imshow(real_wordcloud, interpolation='bilinear')
        st.pyplot(fig)

if __name__ == '__main__':
    run()