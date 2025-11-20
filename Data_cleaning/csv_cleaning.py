import pandas as pd
import numpy as np

data = pd.read_csv('../data/postings.csv')

data = data.drop(columns=['job_id', 'company_name', 'max_salary',
       'pay_period', 'company_id', 'views', 'med_salary',
       'min_salary', 'formatted_work_type', 'applies', 'original_listed_time',
       'remote_allowed', 'job_posting_url', 'application_url',
       'application_type', 'expiry', 'closed_time',
       'formatted_experience_level', 'listed_time',
       'posting_domain', 'sponsored', 'work_type', 'currency',
       'compensation_type', 'normalized_salary', 'zip_code', 'fips'])



data['full_desk'] = data['description'].fillna('') + ' ' + data['skills_desc'].fillna('')
data = data.drop(columns=['description', 'skills_desc'])

# print(data.columns)
# print(data['full_desk'].head())
# print(data['location'].head())

data.to_csv('../data/cleaned_postings.csv', index=False)
